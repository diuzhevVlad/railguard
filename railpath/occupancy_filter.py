"""Stateful Bayesian obstacle filtering on a 1D rail occupancy grid.

The grid is rebuilt on the current predicted rail centreline for every LiDAR
scan. KISS-ICP poses, when enabled, transform the previous grid from the
previous LiDAR frame into the current one before the measurement update. The
module is shared by the browser application and ROS node and imports KISS-ICP
only when odometry is actually enabled.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from .centerline_filter import (BayesianCenterlineFilter, CenterlineEstimate,
                                CenterlineFilterConfig)
from .obstacle_filter import (ObstacleDetection, ObstacleFilterConfig,
                              RailPrediction, detect_obstacle_points)


def _logit(probability: float | np.ndarray) -> float | np.ndarray:
    probability = np.asarray(probability)
    value = np.log(probability / (1.0 - probability))
    return float(value) if value.ndim == 0 else value


def _probability(log_odds: np.ndarray) -> np.ndarray:
    # Stored log odds are bounded, but this split form also remains stable if a
    # caller supplies a much larger finite value in a test or saved state.
    out = np.empty_like(log_odds, dtype=np.float64)
    positive = log_odds >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-log_odds[positive]))
    exponential = np.exp(log_odds[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


@dataclass(frozen=True)
class BayesianFilterConfig:
    """Inverse sensor model and grid settings for temporal filtering."""

    grid_resolution_m: float = 1.0
    prior_probability: float = 0.4
    point_hit_probability: float = 0.56
    miss_probability: float = 0.1
    detection_probability: float = 0.65
    prediction_decay: float = 0.98
    prediction_diffusion: float = 0.10
    neighbor_hit_weight: float = 0.25
    max_scan_hit_probability: float = 0.62
    immediate_close_distance_m: float = 0.0
    immediate_near_distance_m: float = 0.0
    immediate_near_probability: float = 0.999
    immediate_far_distance_m: float = 1e9
    max_point_evidence: int = 10
    max_log_odds: float = 8.0
    max_time_gap_s: float = 1.0
    odometry_scan_stride: int = 8

    def __post_init__(self):
        for name in ("grid_resolution_m", "max_log_odds", "max_time_gap_s"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("prior_probability", "point_hit_probability",
                     "miss_probability", "detection_probability",
                     "max_scan_hit_probability", "immediate_near_probability"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be strictly between zero and one")
        if self.point_hit_probability <= self.prior_probability:
            raise ValueError("point_hit_probability must be greater than the prior")
        if self.max_scan_hit_probability < self.point_hit_probability:
            raise ValueError("max_scan_hit_probability must be >= point_hit_probability")
        for name in ("immediate_close_distance_m", "immediate_near_distance_m",
                     "immediate_far_distance_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.immediate_close_distance_m > self.immediate_near_distance_m:
            raise ValueError(
                "immediate_close_distance_m must be <= immediate_near_distance_m")
        if self.immediate_near_distance_m > self.immediate_far_distance_m:
            raise ValueError(
                "immediate_near_distance_m must be <= immediate_far_distance_m")
        if self.miss_probability >= self.prior_probability:
            raise ValueError("miss_probability must be less than the prior")
        for name in ("prediction_decay", "prediction_diffusion", "neighbor_hit_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        for name in ("max_point_evidence", "odometry_scan_stride"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class RailOccupancy:
    """Posterior occupancy samples expressed in the current LiDAR frame."""

    center: np.ndarray
    distance_along_m: np.ndarray
    probabilities: np.ndarray
    point_counts: np.ndarray
    detection_probability: float
    odometry_used: bool = False

    @classmethod
    def empty(cls, threshold: float, odometry_used: bool = False) -> "RailOccupancy":
        return cls(np.empty((0, 3), dtype=np.float32),
                   np.empty(0, dtype=np.float32),
                   np.empty(0, dtype=np.float32),
                   np.empty(0, dtype=np.int32), threshold, odometry_used)

    @property
    def occupied(self) -> np.ndarray:
        return self.probabilities >= self.detection_probability

    @property
    def detected(self) -> bool:
        return bool(np.any(self.occupied))

    @property
    def distance_m(self) -> float | None:
        occupied = self.occupied
        return float(np.min(self.distance_along_m[occupied])) if np.any(occupied) else None


class KissICPOdometry:
    """Small lazy KISS-ICP adapter returning LiDAR-to-odometry poses."""

    def __init__(self, scan_stride: int = 8, engine_factory=None):
        if not isinstance(scan_stride, int) or isinstance(scan_stride, bool) or scan_stride < 1:
            raise ValueError("scan_stride must be a positive integer")
        self.scan_stride = scan_stride
        self.engine_factory = engine_factory
        self.engine = None

    def _new_engine(self):
        if self.engine_factory is not None:
            return self.engine_factory()
        try:
            from kiss_icp.config import load_config
            from kiss_icp.kiss_icp import KissICP
        except ImportError as exc:
            raise RuntimeError(
                "KISS-ICP is required when odometry_enabled is true; install kiss-icp "
                "or disable odometry"
            ) from exc
        config = load_config(None)
        config.data.deskew = False
        return KissICP(config)

    def reset(self) -> None:
        self.engine = None

    def update(self, xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(xyz)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Expected Nx3 points for odometry")
        if not np.isfinite(points).all():
            raise ValueError("Odometry points must be finite")
        if self.engine is None:
            self.engine = self._new_engine()
        if len(points):
            self.engine.register_frame(points[::self.scan_stride].astype(np.float64, copy=False),
                                       np.empty(0, dtype=np.float64))
        pose = np.asarray(getattr(self.engine, "last_pose", np.eye(4)), dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("KISS-ICP returned an invalid pose")
        return pose.copy()


class BayesianObstacleFilter:
    """Carry a 1D occupancy posterior between local LiDAR frames."""

    def __init__(self, config: BayesianFilterConfig = BayesianFilterConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._center = np.empty((0, 3), dtype=np.float32)
        self._log_odds = np.empty(0, dtype=np.float64)
        self._pose: np.ndarray | None = None

    def _grid(self, prediction: RailPrediction, min_distance_m: float,
              max_distance_m: float | None) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(prediction.center)
        distance = np.asarray(prediction.distance_along_m)
        if (center.ndim != 2 or center.shape[1:] != (3,) or distance.shape != (len(center),)
                or not np.isfinite(center).all() or not np.isfinite(distance).all()):
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty(0, dtype=np.float32))
        increasing = np.r_[True, np.diff(distance) > 1e-6]
        center, distance = center[increasing], distance[increasing]
        if len(center) < 2:
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty(0, dtype=np.float32))
        upper = float(distance[-1])
        if prediction.max_forward_m is not None:
            upper = min(upper, float(prediction.max_forward_m))
        if max_distance_m is not None:
            upper = min(upper, float(max_distance_m))
        resolution = self.config.grid_resolution_m
        first = np.ceil(max(min_distance_m, float(distance[0])) / resolution) * resolution
        last = np.floor(upper / resolution) * resolution
        if not np.isfinite(first) or not np.isfinite(last) or last < first:
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty(0, dtype=np.float32))
        samples = np.arange(first, last + resolution * 0.5, resolution, dtype=np.float64)
        points = np.column_stack([
            np.interp(samples, distance, center[:, axis]) for axis in range(3)
        ]).astype(np.float32)
        return points, samples.astype(np.float32)

    def update(self, prediction: RailPrediction, obstacle_points: np.ndarray,
               pose: np.ndarray | None = None, min_distance_m: float = 0.0,
               max_distance_m: float | None = None,
               odometry_used: bool = False) -> RailOccupancy:
        """Predict into the current frame and apply point-count evidence."""
        points = np.asarray(obstacle_points)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite Nx3 obstacle points")
        current_pose = np.eye(4, dtype=np.float64) if pose is None else np.asarray(pose, dtype=np.float64)
        if current_pose.shape != (4, 4) or not np.isfinite(current_pose).all():
            raise ValueError("Expected a finite 4x4 LiDAR pose")
        center, distances = self._grid(prediction, min_distance_m, max_distance_m)
        if not len(center):
            self.reset()
            return RailOccupancy.empty(self.config.detection_probability, odometry_used)

        prior = np.full(len(center), _logit(self.config.prior_probability), dtype=np.float64)
        if len(self._center) and self._pose is not None:
            previous_to_current = np.linalg.inv(current_pose) @ self._pose
            transformed = (self._center @ previous_to_current[:3, :3].T
                           + previous_to_current[:3, 3])
            nearest_distance, nearest = cKDTree(transformed).query(center)
            supported = nearest_distance <= self.config.grid_resolution_m * 1.5
            prior[supported] = self._log_odds[nearest[supported]]
        prior_log_odds = _logit(self.config.prior_probability)
        offset = prior - prior_log_odds
        diffusion = self.config.prediction_diffusion
        if diffusion and len(offset) > 1:
            spread = (1.0 - diffusion) * offset
            spread[1:] += 0.5 * diffusion * offset[:-1]
            spread[:-1] += 0.5 * diffusion * offset[1:]
            offset = spread
        prior = prior_log_odds + self.config.prediction_decay * offset

        counts = np.zeros(len(center), dtype=np.int32)
        if len(points):
            _, point_bins = cKDTree(center[:, :2]).query(points[:, :2])
            counts = np.bincount(point_bins, minlength=len(center)).astype(np.int32, copy=False)
        weighted_counts = counts.astype(np.float64)
        neighbor_weight = self.config.neighbor_hit_weight
        if neighbor_weight and len(counts) > 1:
            weighted_counts[1:] += neighbor_weight * counts[:-1]
            weighted_counts[:-1] += neighbor_weight * counts[1:]
        hit = weighted_counts > 0
        posterior = prior.copy()
        posterior[~hit] += (_logit(self.config.miss_probability) - prior_log_odds)
        evidence = np.minimum(weighted_counts[hit], self.config.max_point_evidence)
        hit_update = evidence * (_logit(self.config.point_hit_probability) - prior_log_odds)
        max_hit_update = (_logit(self.config.max_scan_hit_probability)
                          - prior_log_odds)
        posterior[hit] += np.minimum(hit_update, max_hit_update)
        posterior = np.clip(posterior, -self.config.max_log_odds, self.config.max_log_odds)

        self._center = center.copy()
        self._log_odds = posterior
        self._pose = current_pose.copy()
        return RailOccupancy(center, distances, _probability(posterior).astype(np.float32),
                             counts, self.config.detection_probability, odometry_used)


class ObstacleDetectionPipeline:
    """Shared single-scan detection plus optional temporal filter/odometry."""

    def __init__(self, obstacle_config: ObstacleFilterConfig = ObstacleFilterConfig(),
                 bayesian_config: BayesianFilterConfig = BayesianFilterConfig(),
                 bayesian_filter_enabled: bool = False,
                 odometry_enabled: bool = False,
                 odometry: KissICPOdometry | None = None,
                 centerline_config: CenterlineFilterConfig = CenterlineFilterConfig(),
                 centerline_filter_enabled: bool = False):
        self.obstacle_config = obstacle_config
        self.bayesian_config = bayesian_config
        self.bayesian_filter_enabled = bool(bayesian_filter_enabled)
        self.centerline_config = centerline_config
        self.centerline_filter_enabled = bool(centerline_filter_enabled)
        self.odometry_enabled = bool(odometry_enabled)
        self.filter = BayesianObstacleFilter(bayesian_config)
        self.centerline_filter = BayesianCenterlineFilter(centerline_config)
        self.odometry = odometry or (KissICPOdometry(bayesian_config.odometry_scan_stride)
                                     if self.odometry_enabled else None)
        self.last_timestamp_ns: int | None = None

    def reset(self) -> None:
        self.filter.reset()
        self.centerline_filter.reset()
        if self.odometry is not None:
            self.odometry.reset()
        self.last_timestamp_ns = None

    def update(self, timestamp_ns: int, xyz: np.ndarray,
               prediction: RailPrediction) -> ObstacleDetection:
        points = np.asarray(xyz)
        timestamp_ns = int(timestamp_ns)
        if self.last_timestamp_ns is not None:
            elapsed_s = (timestamp_ns - self.last_timestamp_ns) / 1e9
            if elapsed_s <= 0 or elapsed_s > self.bayesian_config.max_time_gap_s:
                self.reset()
        self.last_timestamp_ns = timestamp_ns

        pose = np.eye(4, dtype=np.float64)
        if self.odometry_enabled:
            if self.odometry is None:
                self.odometry = KissICPOdometry(self.bayesian_config.odometry_scan_stride)
            pose = self.odometry.update(points)
        centerline_estimate: CenterlineEstimate | None = None
        effective_prediction = prediction
        if self.centerline_filter_enabled:
            centerline_estimate = self.centerline_filter.update(
                prediction, pose, odometry_used=self.odometry_enabled)
            effective_prediction = centerline_estimate.prediction

        raw = detect_obstacle_points(
            points, effective_prediction, self.obstacle_config)
        if not self.bayesian_filter_enabled:
            return replace(
                raw, centerline_estimate=centerline_estimate,
                centerline_filtered=self.centerline_filter_enabled,
                odometry_used=self.odometry_enabled and self.centerline_filter_enabled)

        occupancy = self.filter.update(
            effective_prediction, raw.points, pose,
            min_distance_m=self.obstacle_config.min_forward_m,
            max_distance_m=effective_prediction.max_forward_m,
            odometry_used=self.odometry_enabled,
        )
        maximum_probability = float(np.max(occupancy.probabilities, initial=0.0))
        immediate = bool(raw.detected and raw.distance_m is not None and (
            raw.distance_m <= self.bayesian_config.immediate_close_distance_m
            or (raw.distance_m <= self.bayesian_config.immediate_near_distance_m
             and maximum_probability >= self.bayesian_config.immediate_near_probability)
            or raw.distance_m >= self.bayesian_config.immediate_far_distance_m
        ))
        confirmed = bool(raw.detected and (occupancy.detected or immediate))
        kept = raw.points if confirmed else raw.points[:0]
        rejected = raw.rejected_points
        if not confirmed and len(raw.points):
            rejected = (raw.points if rejected is None or not len(rejected)
                        else np.concatenate((rejected, raw.points)))
        # Mid-range hits need temporal confirmation; strong near hits and sparse
        # far hits may be immediate. Every alert still needs a current cluster,
        # so stale occupied cells cannot alert by themselves.
        return replace(raw, detected=confirmed, point_count=len(kept),
                       distance_m=raw.distance_m if confirmed else None, points=kept,
                       rejected_points=rejected, occupancy=occupancy,
                       bayesian_filtered=True, odometry_used=self.odometry_enabled,
                       centerline_estimate=centerline_estimate,
                       centerline_filtered=self.centerline_filter_enabled)
