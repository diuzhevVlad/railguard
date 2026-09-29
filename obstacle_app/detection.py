"""Merged-rail prediction and train-envelope obstacle detection."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from railpath.obstacle_filter import (
    RailPrediction, ObstacleDetection, ObstacleFilterConfig,
    detect_obstacle_points, envelope_candidates,
)
from .cloud import valid_xyz
from .core import Progress
from .merged_rails import MergedRailMap, merge_rails, world_rail_observation
from .rails import detect_rail_points
from .rail_guard_detector import summarize


INITIAL_SPEED_MPS = {
    "roundT_doubleT": 10.0,
    "roundT_pressureGate_roundT": 5.0,
    "roundT_squareT_pressureGate_squareT": 10.0,
    "squareT_platform_squareT_switch": 10.0,
    "doubleT_obstacle": 0.0,
    "new_data": 10.0,
    "cloud_with_fake_obj": 3.0,
}
TRAIN_WIDTH_M = 2.1
TRAIN_HEIGHT_M = 3.0
LOOKAHEAD_M = 100.0


class RailPredictor(Protocol):
    def predict(self, pose: np.ndarray, lookahead_m: float = LOOKAHEAD_M) -> RailPrediction:
        """Return the predicted local centerline for one pose."""


class ScanRailPredictionProvider(Protocol):
    def predict(self, timestamp_ns: int, point_cloud: object | None = None) -> RailPrediction:
        """Return rails in the coordinates of the scan at this exact timestamp."""


class MergedRailPredictor:
    """Temporary predictor using rails merged from the recorded bag."""

    def __init__(self, rail_map: MergedRailMap):
        self.rail_map = rail_map
        self.tree = cKDTree(rail_map.center[:, :2])

    def predict(self, pose: np.ndarray, lookahead_m: float = LOOKAHEAD_M) -> RailPrediction:
        nearest = int(self.tree.query(pose[:2, 3])[1])
        lengths = self.rail_map.lengths
        end = int(np.searchsorted(lengths, lengths[nearest] + lookahead_m, side="right"))
        world = self.rail_map.center[nearest:max(nearest + 1, end)]
        local = (world - pose[:3, 3]) @ pose[:3, :3]
        return RailPrediction(local.astype(np.float32),
                              (lengths[nearest:nearest + len(world)] - lengths[nearest]).astype(np.float32))


class PrecomputedMergedRailProvider:
    """Temporary scan predictor backed by the merged route and recorded poses."""

    def __init__(self, path: Path):
        with np.load(path) as data:
            self.predictor = MergedRailPredictor(MergedRailMap(data["center"], data["lengths"]))
            self.poses = data["poses"]
            self.timestamps = data["timestamps_ns"]

    def predict(self, timestamp_ns: int, point_cloud: object | None = None) -> RailPrediction:
        index = int(np.searchsorted(self.timestamps, timestamp_ns))
        if index >= len(self.timestamps) or self.timestamps[index] != timestamp_ns:
            raise ValueError("Rail prediction does not match this bag frame")
        return self.predictor.predict(self.poses[index])


def _rail_anchor(rails: dict[str, np.ndarray], pose: np.ndarray) -> np.ndarray | None:
    """Center of paired rails 2–5 m ahead, transformed into the odometry map."""
    nearby = []
    for side in ("left", "right"):
        points = rails[side]
        forward = -points[:, 1]
        selected = points[(forward >= 2) & (forward <= 5)]
        if len(selected) < 5:
            return None
        nearby.append(np.median(selected, axis=0))
    center = (nearby[0] + nearby[1]) / 2
    return center @ pose[:3, :3].T + pose[:3, 3]


def _last_scan_tail(rails: dict[str, np.ndarray], pose: np.ndarray) -> np.ndarray:
    sides = {side: summarize(rails[side]) for side in ("left", "right")}
    end = min(float(-side[-1, 1]) for side in sides.values())
    forward = np.arange(6.0, end, 1.0)
    if not len(forward):
        return np.empty((0, 3), dtype=np.float64)
    paths = []
    for side in sides.values():
        distance = -side[:, 1]
        paths.append(np.column_stack((np.interp(forward, distance, side[:, 0]),
                                      -forward,
                                      np.interp(forward, distance, side[:, 2]))))
    center = (paths[0] + paths[1]) / 2
    return center @ pose[:3, :3].T + pose[:3, 3]


def merged_trail_from_anchors(anchors: list[np.ndarray], tail: np.ndarray) -> MergedRailMap:
    """Follow odometry-ordered rail centers when global -Y binning truncates a run."""
    if len(anchors) < 10:
        raise ValueError("Too few paired rail anchors for a merged trail")
    smooth = gaussian_filter1d(np.asarray(anchors, dtype=np.float64), sigma=3, axis=0)
    center = [smooth[0]]
    for point in smooth[1:]:
        if np.linalg.norm(point - center[-1]) >= 0.3:
            center.append(point)
    for point in tail:
        if np.linalg.norm(point - center[-1]) >= 0.3:
            center.append(point)
    center = np.asarray(center, dtype=np.float32)
    if len(center) < 10:
        raise ValueError("Merged trail is too short")
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
    return MergedRailMap(center, lengths)


def build_merged_rail_map(bag: Path, topic: str, initial_speed_m_s: float,
                          output_path: Path, progress: Progress) -> Path:
    """Estimate poses and merge detected rails for all frames in a source bag."""
    try:
        from kiss_icp.config import load_config
        from kiss_icp.kiss_icp import KissICP
    except ImportError as exc:
        raise RuntimeError(
            "KISS-ICP is required to build the merged obstacle-placement map"
        ) from exc
    config = load_config(None)
    config.data.deskew = False
    odometry = KissICP(config)
    poses = []
    timestamps = []
    observations = []
    anchors = []
    last_rails = None
    last_rail_pose = None
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag], default_typestore=typestore) as reader:
        matches = [c for c in reader.connections if c.topic == topic]
        if len(matches) != 1 or matches[0].msgtype != "sensor_msgs/msg/PointCloud2":
            raise ValueError("Select one PointCloud2 topic")
        total = max(1, matches[0].msgcount)
        for _, bag_ns, raw in reader.messages(connections=matches):
            xyz = valid_xyz(reader.deserialize(raw, matches[0].msgtype))
            if len(poses) == 1:
                odometry.last_delta[1, 3] = -initial_speed_m_s * 0.1
            odometry.register_frame(xyz[::8].astype(np.float64),
                                    np.empty(0, dtype=np.float64))
            pose = odometry.last_pose.copy()
            poses.append(pose)
            timestamps.append(bag_ns)
            rails = detect_rail_points(xyz)
            if len(rails) == 2:
                observations.append(world_rail_observation(rails, pose))
                anchor = _rail_anchor(rails, pose)
                if anchor is not None:
                    anchors.append(anchor)
                    last_rails = rails
                    last_rail_pose = pose
            if len(poses) % 10 == 0:
                progress(min(0.9, len(poses) / total * 0.9),
                         f"Mapped {len(poses)}/{total} LiDAR frames")
    if not poses:
        raise ValueError("No LiDAR frames found")
    route_progress = abs(float(poses[-1][1, 3] - poses[0][1, 3]))
    try:
        merged = merge_rails(observations)
    except ValueError:
        merged = None
    method = "global_rail_merge"
    if merged is None or (route_progress > 20 and merged.length_m < route_progress * 0.8):
        last_tail = (_last_scan_tail(last_rails, last_rail_pose) if last_rails is not None
                     else np.empty((0, 3), dtype=np.float64))
        merged = merged_trail_from_anchors(anchors, last_tail)
        method = "pose_anchored_rail_trail"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, center=merged.center, lengths=merged.lengths,
                        poses=np.asarray(poses), timestamps_ns=np.asarray(timestamps),
                        initial_speed_m_s=initial_speed_m_s, map_method=method)
    progress(1.0, f"Ready: {merged.length_m:.1f} m of merged rails from {len(poses)} frames")
    return output_path
