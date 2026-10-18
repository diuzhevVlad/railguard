"""RailPrediction adapter for the forward-window rolling NURBS filter.

The XY state and update policy match ``nurbs_filter.demo_dataset``. Runtime
output is resampled at the detector's forward coordinates so downstream 3-D
obstacle detection retains its dense grid and measured rail height.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from nurbs_filter.forward_filter import (detector_lateral_change,
                                         enforce_forward_reach,
                                         initialize_forward_window,
                                         sample_forward_curve)
from nurbs_filter.rolling_nurbs import Noise, RollingNURBSKF

from .centerline_filter import CenterlineEstimate, CenterlineFilterConfig
from .obstacle_filter import RailPrediction


class RollingNURBSCenterlineFilter:
    """Track one NURBS curve in the current LiDAR frame."""

    def __init__(self, config: CenterlineFilterConfig = CenterlineFilterConfig()):
        self.config = config
        self.noise = Noise(
            white=config.nurbs_white_m,
            frame_translation=config.nurbs_frame_translation_m,
            smooth_shape=config.nurbs_smooth_shape_m,
            correlation_length=config.nurbs_correlation_length_m,
            process_per_sqrt_second=config.nurbs_process_per_sqrt_second_m,
            birth=config.nurbs_birth_m,
        )
        self.reset()

    def reset(self) -> None:
        self._filter: RollingNURBSKF | None = None
        self._pose: np.ndarray | None = None
        self._timestamp_ns: int | None = None
        self._low_accept_streak = 0
        self._history: list[tuple[np.ndarray, np.ndarray]] = []

    def _initialize(self, measured: np.ndarray) -> None:
        self._filter = initialize_forward_window(
            measured, self.config.nurbs_control_points,
            self.config.nurbs_behind_spans, self.noise)
        self._low_accept_streak = 0

    def _remember(self, center: np.ndarray, pose: np.ndarray) -> None:
        self._history.append((center.copy(), pose.copy()))
        if len(self._history) > 5:
            self._history.pop(0)

    def _stable_detector(self, center: np.ndarray, pose: np.ndarray,
                         max_forward_m: float) -> bool:
        if len(self._history) < 5:
            return False
        targets = np.arange(10., max_forward_m - 5., 5.)
        for lag, limit in ((1, .5), (3, 2.5), (5, 3.)):
            old_center, old_pose = self._history[-lag]
            if detector_lateral_change(center, old_center, pose, old_pose,
                                       targets) >= limit:
                return False
        return True

    def _output(self, prediction: RailPrediction, status: str, accepted: int,
                odometry_used: bool, path_accepted: int,
                endpoint_correction: float,
                max_forward_m: float) -> CenterlineEstimate:
        filt = self._filter
        assert filt is not None
        center = np.asarray(prediction.center, dtype=np.float32)
        sample_u = np.linspace(*filt.domain, 500)
        curve = filt.evaluate(sample_u)
        forward = -curve[:, 1]
        # A forward-only rail has increasing -Y. Ignore tiny local reversals
        # without allowing np.interp to silently use an unsorted coordinate.
        previous_max = np.r_[forward[0] - 1e-6,
                             np.maximum.accumulate(forward)[:-1]]
        keep = forward > previous_max + 1e-7
        if keep.sum() < 2:
            raise RuntimeError("NURBS curve has no forward reach")
        forward, curve_x, sample_u = forward[keep], curve[keep, 0], sample_u[keep]
        target = -center[:, 1].astype(np.float64)
        valid = ((target >= forward[0]) & (target <= forward[-1]) &
                 (target <= max_forward_m))
        filtered = center.copy()
        u = np.empty(len(center), dtype=np.float64)
        u.fill(np.nan)
        u[valid] = np.interp(target[valid], forward, sample_u)
        x = np.interp(target[valid], forward, curve_x)
        strength = self.config.nurbs_output_strength
        filtered[valid, 0] = ((1 - strength) * center[valid, 0] +
                              strength * x).astype(np.float32)
        detector_sigma = np.sqrt(self.noise.white**2 +
                                 self.noise.frame_translation**2 +
                                 self.noise.smooth_shape**2)
        variance = np.full(len(center), detector_sigma**2, dtype=np.float64)
        if np.any(valid):
            basis = filt.basis(u[valid])
            covariance_x = filt.P[::2, ::2]
            curve_variance = np.maximum(
                np.einsum("ij,jk,ik->i", basis, covariance_x, basis), 0.)
            # The detector and posterior share data; this is a conservative
            # positive-correlation bound for a partial output blend.
            variance[valid] = ((1 - strength) * detector_sigma +
                               strength * np.sqrt(curve_variance))**2
        details = {**(prediction.details or {}),
                   "centerline_filter_enabled": True,
                   "centerline_filter_type": "rolling_nurbs",
                   "centerline_filter_status": status,
                   "centerline_filter_odometry_used": bool(odometry_used),
                   "centerline_filter_matched_samples": int(accepted),
                   "centerline_filter_path_accepted": bool(path_accepted),
                   "centerline_filter_endpoint_correction_m": float(endpoint_correction),
                   "centerline_filter_mean_std_m": float(np.sqrt(variance).mean())}
        distances = np.asarray(prediction.distance_along_m, dtype=np.float64)
        filtered_distance = np.r_[
            distances[0],
            distances[0] + np.cumsum(
                np.linalg.norm(np.diff(filtered.astype(np.float64), axis=0), axis=1))
        ].astype(np.float32)
        return CenterlineEstimate(replace(prediction, center=filtered,
                                          distance_along_m=filtered_distance,
                                          details=details),
                                  variance.astype(np.float32), odometry_used)

    def update(self, prediction: RailPrediction, pose: np.ndarray | None = None,
               odometry_used: bool = False,
               timestamp_ns: int | None = None) -> CenterlineEstimate:
        center = np.asarray(prediction.center, dtype=np.float32)
        distance = np.asarray(prediction.distance_along_m)
        current_pose = np.eye(4) if pose is None else np.asarray(pose, dtype=float)
        if (center.ndim != 2 or center.shape[1:] != (3,)
                or distance.shape != (len(center),)
                or not np.isfinite(center).all() or not np.isfinite(distance).all()
                or current_pose.shape != (4, 4)
                or not np.isfinite(current_pose).all()):
            raise ValueError("Expected finite XYZ centerline, distances and 4x4 pose")
        if not len(center):
            self.reset()
            return CenterlineEstimate(prediction, np.empty(0, dtype=np.float32),
                                      odometry_used)
        if timestamp_ns is not None:
            timestamp_ns = int(timestamp_ns)
            if self._timestamp_ns is not None and timestamp_ns <= self._timestamp_ns:
                self.reset()
        max_forward_m = self.config.nurbs_max_forward_m
        if prediction.max_forward_m is not None:
            max_forward_m = min(max_forward_m, float(prediction.max_forward_m))
        targets = np.arange(self.config.nurbs_measurement_spacing_m,
                            max_forward_m + 1e-9,
                            self.config.nurbs_measurement_spacing_m)
        measured = sample_forward_curve(center, targets)
        measured = measured[np.isfinite(measured).all(axis=1)]
        enough = len(measured) >= (self.config.nurbs_control_points -
                                   self.config.nurbs_behind_spans)
        accepted = 0
        status = "no_measurement"
        endpoint_correction = 0.
        path_accepted = 0
        try:
            if enough and self._filter is None:
                self._initialize(measured)
                status = "initialized"
            elif self._filter is not None:
                assert self._pose is not None
                old_rotation = self._pose[:3, :3]
                relative_rotation = old_rotation.T @ current_pose[:3, :3]
                translation = old_rotation.T @ (current_pose[:3, 3] -
                                                self._pose[:3, 3])
                yaw = float(np.arctan2(relative_rotation[1, 0],
                                      relative_rotation[0, 0]))
                dt = (max(0., (timestamp_ns - self._timestamp_ns) / 1e9)
                      if timestamp_ns is not None and self._timestamp_ns is not None
                      else 0.1)
                self._filter.predict(translation[:2], yaw, dt=dt)
                if enough:
                    try:
                        self._filter.recenter(
                            anchor_fraction=self.config.nurbs_anchor_fraction)
                        endpoint_correction += enforce_forward_reach(
                            self._filter, max_forward_m + .01)
                        update = self._filter.update_points(
                            measured,
                            max_distance=self.config.nurbs_max_distance_m,
                            gate=self.config.nurbs_gate,
                            robust_threshold=self.config.nurbs_robust_threshold_m)
                        accepted = update["accepted"]
                        status = "updated" if accepted else "no_update"
                        if accepted < (self.config.nurbs_reinit_accept_fraction *
                                       len(measured)):
                            self._low_accept_streak += 1
                        else:
                            self._low_accept_streak = 0
                        if (self.config.nurbs_reinit_accept_fraction > 0 and
                                self._low_accept_streak >=
                                self.config.nurbs_reinit_patience and
                                self._stable_detector(center, current_pose,
                                                      max_forward_m)):
                            self._initialize(measured)
                            endpoint_correction = 0.
                            accepted = 0
                            status = "reset_low_accept"
                    except (RuntimeError, np.linalg.LinAlgError):
                        self._initialize(measured)
                        endpoint_correction = 0.
                        accepted = 0
                        status = "reset"
                else:
                    self._low_accept_streak = 0
            if self._filter is not None:
                if (self.config.nurbs_path_measurement_enabled and
                        odometry_used):
                    path = self._filter.update_path_position(
                        lateral_sigma=self.config.nurbs_path_lateral_sigma_m)
                    path_accepted = path["accepted"]
                endpoint_correction += enforce_forward_reach(
                    self._filter, max_forward_m + .01)
                result = self._output(prediction, status, accepted,
                                      odometry_used, path_accepted,
                                      endpoint_correction, max_forward_m)
            else:
                result = CenterlineEstimate(
                    prediction, np.full(len(center), self.config.initial_variance_m2,
                                        dtype=np.float32), odometry_used)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            # A degenerate detector/curve must never break obstacle detection.
            self.reset()
            result = CenterlineEstimate(
                prediction, np.full(len(center), self.config.initial_variance_m2,
                                    dtype=np.float32), odometry_used)
        self._pose = current_pose.copy()
        self._timestamp_ns = timestamp_ns
        self._remember(center, current_pose)
        return result
