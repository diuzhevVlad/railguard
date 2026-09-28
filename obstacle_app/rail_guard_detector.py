"""RailGuard_rail_det's point-cloud detector, without its Rerun UI imports.

Adapted from RailGuard_rail_det/ros2_ws/scripts/detect_rails.py at a60ba2d.
The seed ROI, DBSCAN clustering, and forward propagation are unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


SIDES = ("left", "right")
STEP = 1.0


def detect_rails(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    roi = xyz[(z > -1.2) & (z < -1.0) & (x > -1.1) & (x < 1.1) & (y > -5) & (y < -2)]
    if len(roi) == 0:
        return {}

    radius_x = 0.10
    radius_y = 0.50
    scaled_xy = roi[:, :2] / np.array([radius_x, radius_y])
    labels = DBSCAN(eps=1.0, min_samples=10).fit_predict(scaled_xy)
    cluster_ids, counts = np.unique(labels[labels != -1], return_counts=True)
    seed_ids = cluster_ids[np.argsort(counts)[-2:]]
    seeds = [roi[labels == cluster_id] for cluster_id in seed_ids]

    lateral = []
    for points in seeds:
        forward = -points[:, 1]
        near = points[forward <= forward.min() + 2.0]
        lateral.append(float(np.median(near[:, 0])))

    sides = ["left", "right"]
    return {side: seeds[index] for side, index in zip(sides, np.argsort(lateral))}


def summarize(points):
    bins = np.floor(-points[:, 1] / 0.5).astype(int)
    return np.array([np.median(points[bins == b], axis=0) for b in np.unique(bins)])


def propagate_rails(xyz, seeds):
    if any(side not in seeds for side in SIDES):
        return seeds
    rails = {side: seeds[side].copy() for side in SIDES}
    end = min((-p[:, 1]).max() for p in rails.values())
    rails = {side: p[-p[:, 1] <= end] for side, p in rails.items()}
    distance = -xyz[:, 1]

    while True:
        lines = {}
        for side in SIDES:
            rail_points = rails[side]
            forward = -rail_points[:, 1]
            centers = summarize(rail_points[forward >= end - 6])
            distance_from_end = -centers[:, 1] - end
            design = np.column_stack([np.ones(len(centers)), distance_from_end])
            lines[side] = np.linalg.lstsq(design, centers[:, [0, 2]], rcond=None)[0]

        for advance in (STEP, 2 * STEP):
            chunk = xyz[(distance > end) & (distance <= end + advance)]
            ahead = -chunk[:, 1] - end
            candidates = {}
            for side in SIDES:
                position_at_end, slope = lines[side]
                predicted_x = position_at_end[0] + ahead * slope[0]
                predicted_z = position_at_end[1] + ahead * slope[1]
                near_line = (
                    (np.abs(chunk[:, 0] - predicted_x) < 0.12)
                    & (np.abs(chunk[:, 2] - predicted_z) < 0.10)
                )
                candidates[side] = chunk[near_line]
            if all(len(p) >= 5 for p in candidates.values()):
                break
        else:
            break

        for side in SIDES:
            rails[side] = np.concatenate([rails[side], candidates[side]])
        end += advance

    return rails
