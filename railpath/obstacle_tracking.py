"""Bounded, deterministic world-coordinate tracking shared by the editor, web app, and ROS node.

Rebuild only the trailing five scans: seeking, replay, and revisiting a frame
produce the same answer. These tracks are local to that window, not object IDs
that persist across the recording.
"""
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


TRACKING_SETTINGS = dict(window_scans=5, minimum_hits=4, match_distance_m=1.,
                         immediate_unique_points_gt=50,
                         max_gap_s=2., grouping_voxel_m=.25,
                         grouping="26-connected occupied voxels",
                         association="one-to-one nearest cluster surface; last observed geometry",
                         alarm="observed now and either >50 unique XYZ points or four-of-five spatial support")


@dataclass
class Cluster:
    indices: np.ndarray
    world: np.ndarray


@dataclass
class Track:
    world: np.ndarray
    observations: list[int] = field(default_factory=list)


def group_candidates(points, pose, voxel_m=.25):
    """Group already-filtered candidates with a bounded-degree voxel graph.

    This adds no point-count rejection. Centroids represent occupied cells for
    association only; returned indices retain every original candidate point.
    """
    points = np.asarray(points)
    if not len(points):
        return []
    if not np.isfinite(points).all() or not np.isfinite(pose).all():
        raise ValueError("Tracking requires finite points and odometry poses")
    cells, inverse, counts = np.unique(np.floor(points / voxel_m).astype(np.int64),
                                      axis=0, return_inverse=True, return_counts=True)
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    row, col = [], []
    offsets = [offset for offset in product((-1, 0, 1), repeat=3) if offset > (0, 0, 0)]
    for i, cell in enumerate(cells):
        for offset in offsets:
            j = lookup.get(tuple(cell + offset))
            if j is not None:
                row.append(i)
                col.append(j)
    graph = coo_matrix((np.ones(len(row), dtype=np.uint8), (row, col)),
                       shape=(len(cells), len(cells))).tocsr()
    _, labels = connected_components(graph, directed=False)
    centroids = np.column_stack([np.bincount(inverse, weights=points[:, axis]) / counts
                                for axis in range(3)])
    world = centroids @ pose[:3, :3].T + pose[:3, 3]
    # Sort once rather than scanning the whole cloud for every small cluster.
    point_labels = labels[inverse]
    order = np.argsort(point_labels, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(point_labels[order])) + 1)
    cell_order = np.argsort(labels, kind="stable")
    cell_groups = np.split(cell_order, np.flatnonzero(np.diff(labels[cell_order])) + 1)
    return [Cluster(indices, world[cell_indices]) for indices, cell_indices in zip(groups, cell_groups)]


def confirm_window(scans, config=TRACKING_SETTINGS):
    """Return a current-point confirmation mask and currently visible tracks.

    Each scan is (frame_index, timestamp_ns, candidate_xyz, lidar_to_world_pose).
    Missing detections still occupy a scan in the five-scan window. Timestamp
    gaps, duplicate/reversed timestamps, and non-contiguous indices reset history.
    """
    if not scans:
        return np.zeros(0, dtype=bool), []
    scans = scans[-config["window_scans"]:]
    tracks = []
    previous = None
    visible = []
    mask = np.zeros(len(scans[-1][2]), dtype=bool)
    for scan_index, (frame, stamp, points, pose) in enumerate(scans):
        if previous is not None and (frame != previous[0] + 1
                or not 0 < stamp - previous[1] <= config["max_gap_s"] * 1e9):
            tracks = []
        previous = (frame, stamp)
        clusters = group_candidates(points, pose, config["grouping_voxel_m"])
        trees = [cKDTree(track.world) for track in tracks]
        # Cheap AABB distance rejects distant pairs before nearest-point queries.
        pairs = []
        limit = config["match_distance_m"]
        for ci, cluster in enumerate(clusters):
            lo, hi = cluster.world.min(axis=0), cluster.world.max(axis=0)
            for ti, (track, tree) in enumerate(zip(tracks, trees)):
                separation = np.maximum(0, np.maximum(lo - track.world.max(axis=0),
                                                      track.world.min(axis=0) - hi))
                if np.linalg.norm(separation) > limit:
                    continue
                distance = float(np.min(tree.query(cluster.world, distance_upper_bound=limit + 1e-9)[0]))
                if distance <= limit + 1e-9:
                    pairs.append((distance, ci, ti))
        matched, used_tracks = {}, set()
        for _, ci, ti in sorted(pairs):
            if ci not in matched and ti not in used_tracks:
                matched[ci] = ti
                used_tracks.add(ti)
        visible = []
        for ci, cluster in enumerate(clusters):
            if ci in matched:
                track = tracks[matched[ci]]
                track.world = cluster.world
                track.observations.append(frame)
            else:
                track = Track(cluster.world, [frame])
                tracks.append(track)
            canonical = np.ascontiguousarray(points[cluster.indices] + 0)
            keys = canonical.view(np.dtype((np.void, canonical.dtype.itemsize * 3))).reshape(-1)
            unique_count = len(np.unique(keys))
            immediate = unique_count > config["immediate_unique_points_gt"]
            temporal = len(track.observations) >= config["minimum_hits"]
            confirmed = immediate or temporal
            if scan_index == len(scans) - 1 and confirmed:
                mask[cluster.indices] = True
            visible.append(dict(position_world=cluster.world.mean(axis=0).tolist(),
                                min_world=cluster.world.min(axis=0).tolist(),
                                max_world=cluster.world.max(axis=0).tolist(),
                                hits=len(track.observations), observation_frames=track.observations.copy(),
                                confirmed=confirmed, point_count=len(cluster.indices),
                                unique_point_count=unique_count,
                                immediate_confirmation=immediate, temporal_confirmation=temporal))
    return mask, visible
