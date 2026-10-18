"""Detection on the current LiDAR scan using a replaceable rail prediction source."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import threading

import numpy as np

from .obstacle_filter import (BayesianFilterConfig, CenterlineFilterConfig,
                              ObstacleDetectionPipeline, ObstacleFilterConfig)
from .cloud import valid_xyz
from .detection import (ObstacleDetection, PrecomputedMergedRailProvider,
                        RailPrediction, ScanRailPredictionProvider)


class ScanDetector:
    """Keep a small cache so paused refreshes do not reprocess a full scan."""

    def __init__(self, predictions: ScanRailPredictionProvider, cache_frames: int = 12,
                 filter_config: ObstacleFilterConfig = ObstacleFilterConfig(),
                 max_forward_m: float | None = None,
                 bayesian_filter_enabled: bool = False,
                 odometry_enabled: bool = False,
                 bayesian_config: BayesianFilterConfig = BayesianFilterConfig(),
                 centerline_filter_enabled: bool = False,
                 centerline_config: CenterlineFilterConfig = CenterlineFilterConfig()):
        self.max_forward_m = max_forward_m
        self.predictions = predictions
        self.filter_config = filter_config
        self.bayesian_config = bayesian_config
        self.centerline_config = centerline_config
        self.bayesian_filter_enabled = bayesian_filter_enabled
        self.centerline_filter_enabled = centerline_filter_enabled
        self.odometry_enabled = odometry_enabled
        self.pipeline = ObstacleDetectionPipeline(
            filter_config, bayesian_config, bayesian_filter_enabled, odometry_enabled,
            centerline_config=centerline_config,
            centerline_filter_enabled=centerline_filter_enabled)
        self.cache_frames = cache_frames
        self.cache: OrderedDict[int, tuple[RailPrediction, ObstacleDetection]] = OrderedDict()
        self.lock = threading.Lock()
        self.compute_lock = threading.Lock()

    def configure(self, filter_config: ObstacleFilterConfig, max_forward_m: float | None,
                  bayesian_filter_enabled: bool = False,
                  odometry_enabled: bool = False,
                  bayesian_config: BayesianFilterConfig | None = None,
                  centerline_filter_enabled: bool = False,
                  centerline_config: CenterlineFilterConfig | None = None) -> None:
        """Apply one job's options and clear incompatible temporal/cache state."""
        self.filter_config = filter_config
        self.max_forward_m = max_forward_m
        if bayesian_config is not None:
            self.bayesian_config = bayesian_config
        self.bayesian_filter_enabled = bool(bayesian_filter_enabled)
        if centerline_config is not None:
            self.centerline_config = centerline_config
        self.odometry_enabled = bool(odometry_enabled)
        self.centerline_filter_enabled = bool(centerline_filter_enabled)
        self.pipeline = ObstacleDetectionPipeline(
            filter_config, self.bayesian_config,
            self.bayesian_filter_enabled, self.odometry_enabled,
            centerline_config=self.centerline_config,
            centerline_filter_enabled=self.centerline_filter_enabled)
        with self.lock:
            self.cache.clear()

    @classmethod
    def from_map(cls, path: Path) -> "ScanDetector":
        return cls(PrecomputedMergedRailProvider(path))

    def detect(self, timestamp_ns: int, point_cloud: object
               ) -> tuple[RailPrediction, ObstacleDetection]:
        with self.compute_lock:
            cached = self.cached(timestamp_ns)
            if cached is not None:
                return cached
            prediction = self.predictions.predict(timestamp_ns, point_cloud)
            if self.max_forward_m is not None:
                prediction = replace(prediction, max_forward_m=self.max_forward_m)
            if prediction.details is not None:
                sigma = prediction.details.get("sigma_m")
                checked_sigma = ([value for i, value in enumerate(sigma)
                                  if value is not None and i + 1 <= (prediction.max_forward_m or float('inf'))]
                                 if sigma is not None else [])
                prediction = replace(prediction, details={**prediction.details,
                    "checked_min_forward_m": self.filter_config.min_forward_m,
                    "train_width_m": self.filter_config.train_width_m,
                    "train_height_m": self.filter_config.train_height_m,
                    "lookahead_limit_m": prediction.max_forward_m,
                    "bayesian_filter_enabled": self.bayesian_filter_enabled,
                    "centerline_filter_enabled": self.centerline_filter_enabled,
                    "centerline_filter_algorithm": self.centerline_config.algorithm,
                    "odometry_enabled": self.odometry_enabled,
                    "sigma_max_checked_m": max(checked_sigma) if checked_sigma else None,
                    "checked_forward_m": min(prediction.max_forward_m or float('inf'), float(np.max(-prediction.center[:, 1]))) if len(prediction.center) else 0.})
            detection = self.pipeline.update(timestamp_ns, valid_xyz(point_cloud), prediction)
            with self.lock:
                self.cache[timestamp_ns] = (prediction, detection)
                self.cache.move_to_end(timestamp_ns)
                if len(self.cache) > self.cache_frames:
                    self.cache.popitem(last=False)
            return prediction, detection

    def cached(self, timestamp_ns: int) -> tuple[RailPrediction, ObstacleDetection] | None:
        with self.lock:
            result = self.cache.get(timestamp_ns)
            if result is not None:
                self.cache.move_to_end(timestamp_ns)
            return result


def display_points(points: np.ndarray, max_points: int = 20_000) -> np.ndarray:
    """Bound the number of red points sent to the browser."""
    stride = max(1, (len(points) + max_points - 1) // max_points)
    return points[::stride]


def rejected_payload(detection):
    points = detection.rejected_points if detection is not None else None
    return {"excluded_rail_points": detection.excluded_rail_count if detection is not None else 0,
            "rejected_points": len(points) if points is not None else 0,
            "rejected_xyz": display_points(points).tolist() if points is not None else []}


def occupancy_payload(detection):
    occupancy = detection.occupancy if detection is not None else None
    if occupancy is None:
        return {"rail_occupancy": [], "occupancy_threshold": None,
                "bayesian_filter_enabled": False, "odometry_enabled": False}
    rows = np.column_stack((occupancy.center, occupancy.probabilities,
                            occupancy.point_counts))
    return {
        "rail_occupancy": rows.tolist(),
        "occupancy_threshold": occupancy.detection_probability,
        "bayesian_filter_enabled": detection.bayesian_filtered,
        "odometry_enabled": detection.odometry_used,
    }


def centerline_payload(detection):
    estimate = detection.centerline_estimate if detection is not None else None
    if estimate is None:
        return {
            "filtered_rail_center": [],
            "centerline_variance_m2": [],
            "centerline_filter_enabled": False,
            "centerline_filter_odometry_used": False,
        }
    return {
        "filtered_rail_center": estimate.prediction.center.tolist(),
        "centerline_variance_m2": estimate.variances_m2.tolist(),
        "centerline_filter_enabled": detection.centerline_filtered,
        "centerline_filter_odometry_used": estimate.odometry_used,
    }
