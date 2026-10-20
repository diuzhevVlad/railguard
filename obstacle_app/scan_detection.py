"""Detection on the current LiDAR scan using a replaceable rail prediction source."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import threading

import numpy as np

from .obstacle_filter import ObstacleFilterConfig
from .cloud import valid_xyz
from .detection import (ObstacleDetection, PrecomputedMergedRailProvider,
                        RailPrediction, ScanRailPredictionProvider,
                        detect_obstacle_points)


class ScanDetector:
    """Keep a small cache so paused refreshes do not reprocess a full scan."""

    def __init__(self, predictions: ScanRailPredictionProvider, cache_frames: int = 12,
                 filter_config: ObstacleFilterConfig = ObstacleFilterConfig(verify_wall_geometry=True, reject_trackbed_strips=True), max_forward_m: float | None = None,
                 candidate_detector=detect_obstacle_points):
        self.max_forward_m = max_forward_m
        self.predictions = predictions
        self.filter_config = filter_config
        self.candidate_detector = candidate_detector
        self.cache_frames = cache_frames
        self.cache: OrderedDict[int, tuple[RailPrediction, ObstacleDetection]] = OrderedDict()
        self.lock = threading.Lock()
        self.compute_lock = threading.Lock()

    @classmethod
    def from_map(cls, path: Path) -> ScanDetector:
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
                    "sigma_max_checked_m": max(checked_sigma) if checked_sigma else None,
                    "checked_forward_m": min(prediction.max_forward_m or float('inf'), float(np.max(-prediction.center[:, 1]))) if len(prediction.center) else 0.})
            detection = self.candidate_detector(valid_xyz(point_cloud), prediction, self.filter_config)
            # Draw exactly the envelope used for classification, including its
            # observed-rail correction; keep the raw model itself unchanged.
            if detection.prediction is not None:
                prediction=detection.prediction
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


def tracking_payload(detection):
    pending = detection.unconfirmed_points if detection is not None else None
    candidates = detection.candidate_points if detection is not None else None
    return dict(raw_candidate_points=len(candidates) if candidates is not None else 0,
                unconfirmed_xyz=display_points(pending).tolist() if pending is not None else [],
                tracks=detection.tracks or [] if detection is not None else [])
