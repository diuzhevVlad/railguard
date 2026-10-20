"""Single-scan proposals and odometry confirmation, without neural/ROS imports."""
from collections import deque
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from .obstacle_filter import (
    ObstacleFilterConfig, ObstacleDetection, envelope_candidates,
    filter_obstacle_candidates,
)
from .obstacle_tracking import TRACKING_SETTINGS, confirm_window


CANDIDATE_CONFIG = ObstacleFilterConfig(
    min_cluster_points=4, count_unique_xyz=True, near_height_distance_m=15.,
    stabilize_envelope=False, min_near_component_extent_m=0.,
)


def detection_from_points(points, prediction, **kwargs):
    """Describe the current returns; distance follows the predicted centerline."""
    distance = nearest_point = None
    if len(points):
        _, nearest = cKDTree(prediction.center[:, :2]).query(points[:, :2])
        distances = prediction.distance_along_m[nearest]
        i = int(np.argmin(distances))
        distance, nearest_point = float(distances[i]), points[i].copy()
    return ObstacleDetection(bool(len(points)), len(points), distance, points,
                             prediction=prediction, nearest_point=nearest_point, **kwargs)


def detect_candidates(xyz, prediction, config=CANDIDATE_CONFIG):
    """Editor rules: raised envelope, predicted-rail exclusion, unique XYZ support.

    The clearance shifts both floor and roof. It is selected by each point's
    forward depth (20 cm through 15 m; 10 cm afterwards), before clustering.
    Neither the predicted centerline nor the source cloud is modified.
    """
    xyz = np.asarray(xyz)
    center = prediction.center
    selected, excluded = [], 0
    if len(center):
        maximum = min(prediction.max_forward_m or np.inf, float(np.max(-center[:, 1])))
        depth = -xyz[:, 1]
        valid = (np.isfinite(xyz).all(1) & (depth >= config.min_forward_m)
                 & (depth <= maximum))
        near = depth <= config.near_height_distance_m
        for region, offset in ((near, config.near_min_height_above_track_m),
                               (~near, config.min_height_above_track_m)):
            shifted = center.copy()
            shifted[:, 2] += offset
            pts, _ = envelope_candidates(
                xyz[valid & region], replace(prediction, center=shifted),
                config.train_width_m, config.train_height_m)
            selected.append(pts)
    candidates = np.concatenate(selected) if selected else np.empty((0, 3), np.float32)
    rails = prediction.rail_points
    if len(candidates) and rails is not None and len(rails):
        distance = cKDTree(rails).query(candidates, distance_upper_bound=np.nextafter(1e-5, np.inf))[0]
        is_rail = distance <= 1e-5
        excluded = int(is_rail.sum())
        candidates = candidates[~is_rail]
    keep = filter_obstacle_candidates(candidates, config)
    details = dict(prediction.details or {}, envelope_offset_m=config.min_height_above_track_m,
                   near_envelope_offset_m=config.near_min_height_above_track_m,
                   near_envelope_depth_m=config.near_height_distance_m)
    prediction = replace(prediction, details=details)
    return detection_from_points(candidates[keep], prediction,
                                 rejected_points=candidates[~keep],
                                 excluded_rail_count=excluded)


def confirm_detection(candidate, scans, settings=TRACKING_SETTINGS):
    keep, tracks = confirm_window(scans, settings)
    return detection_from_points(
        candidate.points[keep], candidate.prediction,
        rejected_points=candidate.rejected_points,
        excluded_rail_count=candidate.excluded_rail_count,
        candidate_points=candidate.points, unconfirmed_points=candidate.points[~keep],
        tracks=tracks)


class StreamingConfirmation:
    """Bounded history for a live subscriber; one observation per processed scan."""
    def __init__(self, settings=None):
        self.settings = dict(TRACKING_SETTINGS if settings is None else settings)
        for key in ("window_scans", "minimum_hits", "immediate_unique_points_gt"):
            value = self.settings[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.settings["minimum_hits"] > self.settings["window_scans"]:
            raise ValueError("minimum_hits must not exceed window_scans")
        for key in ("match_distance_m", "max_gap_s", "grouping_voxel_m"):
            if not np.isfinite(self.settings[key]) or self.settings[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        self.scans = deque(maxlen=self.settings["window_scans"])
        self.frame = 0

    def reset(self):
        self.scans.clear()
        self.frame = 0

    def update(self, stamp_ns, candidate, pose):
        self.scans.append((self.frame, stamp_ns, candidate.points, pose.copy()))
        self.frame += 1
        return confirm_detection(candidate, list(self.scans), self.settings)
