"""Merge rail detections with KISS-ICP poses and place by observed track length."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .rail_guard_detector import summarize


@dataclass(frozen=True)
class RailPlacement:
    position: np.ndarray
    direction: np.ndarray
    extrapolated_m: float = 0.0


def _point_at_length(center: np.ndarray, lengths: np.ndarray, distance: float) -> np.ndarray:
    return np.array([np.interp(distance, lengths, center[:, axis]) for axis in range(3)])


@dataclass(frozen=True)
class MergedRailMap:
    center: np.ndarray
    lengths: np.ndarray

    @property
    def length_m(self) -> float:
        return float(self.lengths[-1])

    def locate(self, pose: np.ndarray, ahead_m: float,
               extrapolate: bool = False) -> RailPlacement:
        """Return a rail point ahead of this LiDAR pose."""
        nearest = cKDTree(self.center[:, :2]).query(pose[:2, 3])[1]
        target = float(self.lengths[nearest] + ahead_m)
        extra = max(0.0, target - self.length_m)
        if target > self.length_m:
            if not extrapolate:
                available = max(0.0, self.length_m - float(self.lengths[nearest]))
                raise ValueError(f"Merged rails extend only {available:.1f} m ahead; "
                                 f"requested {ahead_m:.1f} m")
            before = _point_at_length(self.center, self.lengths, max(0, self.length_m - 5))
            tangent = self.center[-1] - before
            tangent /= np.linalg.norm(tangent)
            world = self.center[-1] + tangent * extra
        else:
            world = _point_at_length(self.center, self.lengths, target)
            before = _point_at_length(self.center, self.lengths, max(0, target - 1))
            after = _point_at_length(self.center, self.lengths, min(self.length_m, target + 1))
            tangent = after - before
        rotation = pose[:3, :3]
        position = rotation.T @ (world - pose[:3, 3])
        direction = (rotation.T @ tangent)[:2]
        direction /= np.linalg.norm(direction)
        return RailPlacement(position.astype(np.float32), direction.astype(np.float32), extra)


def merge_rails(observations: list[dict[str, np.ndarray]]) -> MergedRailMap:
    """Bin left/right world-space detections into one continuous rail centerline."""
    sides = {}
    for side in ("left", "right"):
        clouds = [item[side] for item in observations if side in item]
        if not clouds:
            raise ValueError("No paired rails were detected for the merged map")
        points = np.concatenate(clouds)
        bins = np.floor(-points[:, 1] / 0.5).astype(np.int32)
        ids = np.unique(bins)
        sides[side] = np.array([np.median(points[bins == index], axis=0)
                                for index in ids], dtype=np.float64)
        sides[side] = sides[side][np.argsort(-sides[side][:, 1])]

    last_y = min(float(-points[-1, 1]) for points in sides.values())
    if last_y < 4:
        raise ValueError("Merged rails are too short to establish a centerline")
    forward = np.r_[np.arange(0, last_y, 0.5), last_y]
    interpolated = {}
    for side, points in sides.items():
        positions = -points[:, 1]
        if len(positions) > 1 and np.max(np.diff(positions)) > 3:
            raise ValueError("Merged rail map has a gap larger than 3 m")
        interpolated[side] = np.column_stack((
            np.interp(forward, positions, points[:, 0]),
            -forward,
            np.interp(forward, positions, points[:, 2]),
        ))
    gauge = np.linalg.norm(interpolated["right"][:, :2] - interpolated["left"][:, :2], axis=1)
    bad = np.flatnonzero((gauge < 1.15) | (gauge > 1.9))
    usable = int(bad[0]) if len(bad) else len(forward)
    if usable < 9:
        raise ValueError("Merged rail separation is inconsistent")
    center = (interpolated["left"][:usable] + interpolated["right"][:usable]) * 0.5
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
    return MergedRailMap(center.astype(np.float32), lengths)


def world_rail_observation(rails: dict[str, np.ndarray], pose: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize local rails and transform them into the odometry map frame."""
    return {side: (summarize(points).astype(np.float64) @ pose[:3, :3].T
                   + pose[:3, 3]).astype(np.float32)
            for side, points in rails.items() if len(points) >= 10}
