"""Shared obstacle filtering for the ROS node and web app.

Only NumPy/SciPy are required. Coordinates: metres, forward -Y, up +Z.
The neural model and ROS transport are independent of this module.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class ObstacleFilterConfig:
    train_width_m: float = 2.1
    train_height_m: float = 3.0
    min_forward_m: float = 1.5
    min_height_above_track_m: float = 0.10
    near_min_height_above_track_m: float = 0.20
    near_height_distance_m: float = 10.0
    min_cluster_points: int = 5
    distance_knots_m: tuple[float, ...] = (0., 30., 80., 150.)
    radius_knots_m: tuple[float, ...] = (.2, .2, .35, .6)

    def __post_init__(self):
        for value in (self.train_width_m, self.train_height_m):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Train dimensions must be finite and positive")
        if not np.isfinite(self.min_forward_m) or self.min_forward_m < 0:
            raise ValueError("min_forward_m must be finite and nonnegative")
        if not np.isfinite(self.min_height_above_track_m) or self.min_height_above_track_m < 0:
            raise ValueError("min_height_above_track_m must be finite and nonnegative")
        if not np.isfinite(self.near_min_height_above_track_m) or self.near_min_height_above_track_m < 0:
            raise ValueError("near_min_height_above_track_m must be finite and nonnegative")
        if not np.isfinite(self.near_height_distance_m) or self.near_height_distance_m < 0:
            raise ValueError("near_height_distance_m must be finite and nonnegative")
        if not isinstance(self.min_cluster_points, int) or self.min_cluster_points < 1:
            raise ValueError('min_cluster_points must be a positive integer')
        d, r = np.asarray(self.distance_knots_m), np.asarray(self.radius_knots_m)
        if (d.ndim != 1 or r.shape != d.shape or len(d) < 2
                or not np.isfinite(d).all() or not np.isfinite(r).all()
                or np.any(np.diff(d) <= 0) or d[0] < 0 or np.any(r <= 0)):
            raise ValueError('Distances must increase and radii must be positive and finite')


def filter_obstacle_candidates(points, config=ObstacleFilterConfig()):
    """Return a keep mask without modifying points.

    Radius interpolates smoothly with forward depth (-Y). Two points connect
    only if their 3D distance fits BOTH radii, making adjacency symmetric.
    Components use total point count, not DBSCAN's local core density, so thin
    chains can survive. Retain at most min_cluster_points neighbors per point:
    small components retain all edges; a truncated neighborhood already proves
    its component reaches the minimum size. Thus the keep mask is unchanged,
    without constructing all pairwise edges in dense contact scans.
    """
    xyz = np.asarray(points)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite Nx3 candidate points')
    keep = np.zeros(len(xyz), dtype=bool)
    if not len(xyz):
        return keep
    radius = np.interp(np.maximum(0., -xyz[:, 1]),
                       config.distance_knots_m, config.radius_knots_m)
    minimum = config.min_cluster_points
    if minimum == 1:
        return np.ones(len(xyz), dtype=bool)
    if len(xyz) < minimum:
        return keep
    # Points in a cell whose diagonal fits the smallest radius are mutually
    # connected. Dense cells prove acceptance without any neighbor queries.
    cell_size = radius.min() / np.sqrt(3.) * (1. - 1e-12)
    cells = np.floor((xyz - xyz.min(axis=0)) / cell_size)
    shape = cells.max(axis=0) + 1
    dense = np.zeros(len(xyz), dtype=bool)
    if np.all(shape < 1e6) and float(np.prod(shape)) < np.iinfo(np.int64).max:
        cells = cells.astype(np.int64)
        shape = shape.astype(np.int64)
        keys = (cells[:,0] * shape[1] + cells[:,1]) * shape[2] + cells[:,2]
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        dense = counts[inverse] >= minimum
    pending = np.flatnonzero(~dense)
    if not len(pending):
        return dense
    tree = cKDTree(xyz)
    k = minimum
    distance, neighbors = tree.query(xyz[pending], k=list(range(1, k + 1)),
                                    distance_upper_bound=np.nextafter(radius.max(), np.inf),
                                    workers=4 if len(pending) >= 10000 else 1)
    safe = np.minimum(neighbors, len(xyz)-1)
    valid = (neighbors < len(xyz)) & (distance <= radius[pending, None]) & (distance <= radius[safe])
    # A farther neighbor may have a larger radius than a closer rejected one.
    # Only these boundary cases need a full radius query; keep at most k edges.
    ambiguous = (distance[:, -1] <= radius[pending]) & ~valid.all(axis=1)
    for row in np.flatnonzero(ambiguous):
        i = pending[row]
        ids = np.asarray(tree.query_ball_point(xyz[i], radius[i]), dtype=int)
        delta = xyz[ids] - xyz[i]
        ids = ids[np.einsum('ij,ij->i', delta, delta)
                  <= np.minimum(radius[i], radius[ids]) ** 2][:k]
        valid[row] = False
        valid[row, :len(ids)] = True
        neighbors[row, :len(ids)] = ids
    rows = np.broadcast_to(pending[:, None], neighbors.shape)[valid]
    graph = coo_matrix((np.ones(len(rows), dtype=bool), (rows, neighbors[valid])),
                       shape=(len(xyz), len(xyz))).tocsr()
    _, labels = connected_components(graph, directed=False)
    accepted = np.bincount(labels) >= minimum
    accepted[labels[dense]] = True
    return accepted[labels]


@dataclass(frozen=True)
class RailPrediction:
    center: np.ndarray
    distance_along_m: np.ndarray
    continuation: np.ndarray | None = None
    details: dict | None = None
    rail_points: np.ndarray | None = None
    max_forward_m: float | None = None


@dataclass(frozen=True)
class ObstacleDetection:
    detected: bool
    point_count: int
    distance_m: float | None
    points: np.ndarray
    rejected_points: np.ndarray | None = None
    excluded_rail_count: int = 0
    nearest_point: np.ndarray | None = None


def envelope_candidates(xyz: np.ndarray, prediction: RailPrediction,
                        width_m: float = 2.1, height_m: float = 3.0
                        ) -> tuple[np.ndarray, float | None]:
    """Return every point inside the forward envelope and its closest distance."""
    points, nearest = _envelope_candidates(xyz, prediction, width_m, height_m)
    distance = float(np.min(prediction.distance_along_m[nearest])) if len(points) else None
    return points, distance


def _envelope_candidates(xyz, prediction, width_m, height_m):
    """Keep nearest-center indices for reuse by height and distance filtering."""
    empty = (np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.intp))
    if len(prediction.center) == 0 or len(xyz) == 0:
        return empty
    center = prediction.center
    if prediction.max_forward_m is not None:
        xyz = xyz[(-xyz[:, 1] >= 0) & (-xyz[:, 1] <= prediction.max_forward_m)]
    # Restrict the expensive nearest-path query to a broad sensor-frame ROI.
    lo = center.min(axis=0) - np.array([width_m, width_m, 0.0])
    hi = center.max(axis=0) + np.array([width_m, width_m, height_m])
    candidates = xyz[np.all((xyz >= lo) & (xyz <= hi), axis=1)]
    if not len(candidates):
        return empty
    lateral, nearest = cKDTree(center[:, :2]).query(
        candidates[:, :2], distance_upper_bound=width_m / 2,
        workers=4 if len(candidates) >= 1000 else 1)
    near = lateral <= width_m / 2
    candidates = candidates[near]
    nearest = nearest[near]
    if not len(candidates):
        return empty
    height = candidates[:, 2] - center[nearest, 2]
    inside = (height > 0) & (height <= height_m)
    points = candidates[inside].astype(np.float32, copy=False)
    return points, nearest[inside]


def detect_obstacle_points(xyz: np.ndarray, prediction: RailPrediction,
                           filter_config: ObstacleFilterConfig = ObstacleFilterConfig()) -> ObstacleDetection:
    """Filter small components after the forward cutoff and envelope check."""
    candidates, nearest = _envelope_candidates(xyz, prediction, filter_config.train_width_m, filter_config.train_height_m)
    if not len(candidates):
        return ObstacleDetection(False, 0, None, candidates, candidates.copy())
    is_rail = np.zeros(len(candidates), dtype=bool)
    if prediction.rail_points is not None and len(prediction.rail_points):
        # Match the model's returned XYZ to original scan returns. This tiny
        # tolerance is only numerical; do not erase nearby non-rail objects.
        rail_distance, _ = cKDTree(prediction.rail_points).query(
            candidates, distance_upper_bound=np.nextafter(1e-5, np.inf),
            workers=4 if len(candidates) >= 10000 else 1)
        is_rail = rail_distance <= 1e-5
    heights = candidates[:, 2] - prediction.center[nearest, 2]
    keep = ((~is_rail) & (heights > filter_config.min_height_above_track_m)
            & (-candidates[:, 1] >= filter_config.min_forward_m))
    active = np.flatnonzero(keep)
    depth = -candidates[active, 1]
    boundary = filter_config.near_height_distance_m
    near = depth < boundary
    low_near = near & (heights[active] <= filter_config.near_min_height_above_track_m)
    if (filter_config.near_min_height_above_track_m > filter_config.min_height_above_track_m
            and low_near.any()):
        # A size-k component has a connected witness of at most k points
        # starting at any member. k maximum-radius steps provide a conservative
        # halo, so each pass only needs its region plus boundary neighbors.
        # Far points retain baseline 10 cm decisions; near points use 20 cm.
        # The old full baseline pass before near filtering was redundant:
        # removing points cannot make a previously small component qualify.
        halo = filter_config.min_cluster_points * max(filter_config.radius_knots_m)
        keep[active] = False
        for context, target in ((depth >= boundary - halo, ~near),
                                ((depth <= boundary + halo) & ~low_near, near)):
            if not np.any(context & target):
                continue
            indices = np.flatnonzero(context)
            accepted = filter_obstacle_candidates(candidates[active[indices]], filter_config)
            selected = indices[accepted & target[indices]]
            keep[active[selected]] = True
    else:
        keep[active] = filter_obstacle_candidates(candidates[active], filter_config)
    points = candidates[keep]
    distance = None
    nearest_point = None
    if len(points):
        distances = prediction.distance_along_m[nearest[keep]]
        closest = int(np.argmin(distances))
        distance = float(distances[closest])
        nearest_point = points[closest].copy()
    return ObstacleDetection(bool(len(points)), len(points), distance, points,
                             candidates[~keep & ~is_rail], int(is_rail.sum()), nearest_point)



def detect_obstacles(points, centerline, rail_points=None,
                     config=ObstacleFilterConfig(), max_forward_m=150.0):
    """Filter one scan using the model's raw 1 m centerline samples.

    Returns ObstacleDetection with retained XYZ and nearest arc distance in
    metres (None when there are no retained points). Invalid/truncated curves
    use only their initial finite section; no continuation is invented here.
    Inputs are never modified. The returned point order follows the input.
    """
    if not np.isfinite(max_forward_m) or max_forward_m < config.min_forward_m:
        raise ValueError('max_forward_m must be finite and >= min_forward_m')
    xyz = np.asarray(points)
    curve = np.asarray(centerline, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or curve.ndim != 2 or curve.shape[1] != 3:
        raise ValueError('Expected Nx3 points and Mx3 centerline')
    xyz = xyz[np.isfinite(xyz).all(axis=1) & (np.einsum('ij,ij->i', xyz, xyz) >= .09)]
    invalid = np.flatnonzero(~np.isfinite(curve).all(axis=1))
    path = curve[:int(invalid[0])] if len(invalid) else curve
    center, distances = np.empty((0, 3), np.float32), np.empty(0, np.float32)
    if len(path) >= 2:
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        unique = np.r_[True, np.diff(lengths) > 1e-6]
        path, lengths = path[unique], lengths[unique]
        if len(path) >= 2:
            sample = np.r_[np.arange(0., lengths[-1], .25), lengths[-1]]
            center = np.column_stack([np.interp(sample, lengths, path[:, a]) for a in range(3)]).astype(np.float32)
            first_step = np.linalg.norm(path[1] - path[0])
            first_depth_step = float(-path[1, 1] + path[0, 1])
            origin_offset = max(0., float(-path[0, 1])) * first_step / max(first_depth_step, 1e-6)
            distances = (sample + origin_offset).astype(np.float32)
    rails = None
    if rail_points is not None:
        rails = np.asarray(rail_points, dtype=np.float32)
        if rails.ndim != 2 or rails.shape[1] != 3:
            raise ValueError('Expected Nx3 rail points')
        rails = rails[np.isfinite(rails).all(axis=1)]
    return detect_obstacle_points(xyz, RailPrediction(center, distances, rail_points=rails, max_forward_m=max_forward_m), config)
