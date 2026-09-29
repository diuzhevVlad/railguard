"""Kalman-style temporal filtering for rail centerline predictions.

The state is expressed in the local LiDAR frame.  When a LiDAR-to-odometry
pose is supplied, the predict step transforms the previous state into the
current LiDAR frame before associating it with the current model prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from .obstacle_filter import RailPrediction


@dataclass(frozen=True)
class CenterlineFilterConfig:
    """Scalar covariance settings shared by every XYZ centerline sample."""

    initial_variance_m2: float = 0.25
    process_variance_m2: float = 0.02
    measurement_variance_m2: float = 0.09
    measurement_variance_growth_m2_per_m: float = 0.0
    max_measurement_variance_m2: float = 0.50
    use_model_uncertainty: bool = True
    max_correspondence_distance_m: float = 2.0

    def __post_init__(self) -> None:
        for name in ("initial_variance_m2", "measurement_variance_m2",
                     "max_measurement_variance_m2",
                     "max_correspondence_distance_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not np.isfinite(self.process_variance_m2)
                or self.process_variance_m2 < 0):
            raise ValueError("process_variance_m2 must be finite and nonnegative")
        if (not np.isfinite(self.measurement_variance_growth_m2_per_m)
                or self.measurement_variance_growth_m2_per_m < 0):
            raise ValueError(
                "measurement_variance_growth_m2_per_m must be finite and nonnegative")
        if self.max_measurement_variance_m2 < self.measurement_variance_m2:
            raise ValueError(
                "max_measurement_variance_m2 must be >= measurement_variance_m2")
        if not isinstance(self.use_model_uncertainty, bool):
            raise ValueError("use_model_uncertainty must be a boolean")


@dataclass(frozen=True)
class CenterlineEstimate:
    """Filtered centerline and posterior variance in the current LiDAR frame."""

    prediction: RailPrediction
    variances_m2: np.ndarray
    odometry_used: bool = False


class BayesianCenterlineFilter:
    """Independent-point Kalman filter with geometric data association.

    Each dense centerline sample carries one isotropic XYZ variance.  The
    model centerline is the measurement; the transformed previous centerline
    is the prior.  Prior influence fades smoothly near the association gate,
    preventing a hard spatial jump between fused and measurement-only
    sections of the line.
    """

    def __init__(self, config: CenterlineFilterConfig = CenterlineFilterConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._center = np.empty((0, 3), dtype=np.float32)
        self._variances_m2 = np.empty(0, dtype=np.float64)
        self._pose: np.ndarray | None = None

    @staticmethod
    def _validated_center(prediction: RailPrediction) -> np.ndarray:
        center = np.asarray(prediction.center, dtype=np.float32)
        distances = np.asarray(prediction.distance_along_m)
        if (center.ndim != 2 or center.shape[1:] != (3,)
                or distances.shape != (len(center),)
                or not np.isfinite(center).all()
                or not np.isfinite(distances).all()):
            raise ValueError("RailPrediction must contain finite centerline samples")
        return center

    def _measurement_variances(self, prediction: RailPrediction) -> np.ndarray:
        """Return range- and model-uncertainty-aware variance per sample."""
        distance = np.maximum(
            np.asarray(prediction.distance_along_m, dtype=np.float64), 0.0)
        variance = (self.config.measurement_variance_m2
                    + self.config.measurement_variance_growth_m2_per_m * distance)
        sigma_values = ((prediction.details or {}).get("sigma_m")
                        if self.config.use_model_uncertainty else None)
        if sigma_values is not None:
            try:
                sigma = np.asarray(sigma_values, dtype=np.float64)
            except (TypeError, ValueError):
                sigma = np.empty(0, dtype=np.float64)
            if (sigma.ndim == 1 and len(sigma)
                    and np.isfinite(sigma).all() and np.all(sigma >= 0)):
                # Model sigma is calibrated at the raw d=1..N metre rows.
                sigma_distance = np.arange(1, len(sigma) + 1, dtype=np.float64)
                interpolated = np.interp(distance, sigma_distance, sigma)
                variance = np.maximum(variance, interpolated ** 2)
        return np.minimum(variance, self.config.max_measurement_variance_m2)

    def update(self, prediction: RailPrediction, pose: np.ndarray | None = None,
               odometry_used: bool = False) -> CenterlineEstimate:
        """Predict the prior into this frame and fuse the current centerline."""
        measurement = self._validated_center(prediction)
        current_pose = (np.eye(4, dtype=np.float64) if pose is None
                        else np.asarray(pose, dtype=np.float64))
        if current_pose.shape != (4, 4) or not np.isfinite(current_pose).all():
            raise ValueError("Expected a finite 4x4 LiDAR pose")
        if not len(measurement):
            self.reset()
            return CenterlineEstimate(prediction, np.empty(0, dtype=np.float32),
                                      odometry_used)

        filtered = measurement.astype(np.float64, copy=True)
        measurement_variance = self._measurement_variances(prediction)
        variances = np.full(len(measurement), self.config.initial_variance_m2,
                            dtype=np.float64)
        matched_count = 0
        if len(self._center) and self._pose is not None:
            previous_to_current = np.linalg.inv(current_pose) @ self._pose
            prior_center = (self._center @ previous_to_current[:3, :3].T
                            + previous_to_current[:3, 3])
            separation, nearest = cKDTree(prior_center).query(measurement)
            predicted_variance = (
                self._variances_m2[nearest] + self.config.process_variance_m2)
            gain = predicted_variance / (
                predicted_variance + measurement_variance)
            gate = self.config.max_correspondence_distance_m
            fade_start = 0.5 * gate
            transition = np.clip(
                (separation - fade_start) / max(gate - fade_start, 1e-9), 0., 1.)
            association_weight = 1. - transition ** 2 * (3. - 2. * transition)
            if len(association_weight) >= 5:
                kernel = np.array([1., 4., 6., 4., 1.]) / 16.
                association_weight = np.convolve(
                    np.pad(association_weight, 2, mode="edge"), kernel, mode="valid")
            prior_weight = association_weight * (1. - gain)
            prior = prior_center[nearest]
            filtered = measurement + prior_weight[:, None] * (prior - measurement)
            posterior_variance = (1. - gain) * predicted_variance
            variances = (
                association_weight * posterior_variance
                + (1. - association_weight) * self.config.initial_variance_m2)
            matched_count = int(np.count_nonzero(association_weight > 0))

        filtered = filtered.astype(np.float32)
        variances_f32 = variances.astype(np.float32)
        details = prediction.details
        if details is not None:
            details = {
                **details,
                "centerline_filter_enabled": True,
                "centerline_filter_odometry_used": bool(odometry_used),
                "centerline_filter_matched_samples": matched_count,
                "centerline_filter_mean_std_m": float(np.sqrt(variances).mean()),
                "centerline_filter_mean_measurement_std_m": float(
                    np.sqrt(measurement_variance).mean()),
            }
        estimate_prediction = replace(prediction, center=filtered, details=details)
        self._center = filtered.copy()
        self._variances_m2 = variances.copy()
        self._pose = current_pose.copy()
        return CenterlineEstimate(estimate_prediction, variances_f32, odometry_used)
