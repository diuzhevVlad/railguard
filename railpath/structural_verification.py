"""Bounded scan-only verification of obstacle proposals.

Wall constraints require lower/upper surface agreement and a smooth curve.
Track-bed rejection requires a whole thin group to lie below a rail-bounded
floor estimate. No bag identity, map, GT labels, odometry, or history is used.
The shared detector opts into each check through ObstacleFilterConfig.
"""

from dataclasses import replace
from functools import lru_cache
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .envelope_stabilization import _segmented_anchors
from .obstacle_filter import ObstacleFilterConfig


def _replace_center(prediction, center, details):
    original = prediction.center
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
    distance += prediction.distance_along_m[0]
    continuation = prediction.continuation
    if continuation is not None and len(continuation):
        continuation = np.asarray(continuation).copy()
        for axis in (0, 2):
            continuation[:, axis] += np.interp(
                -continuation[:, 1],
                -original[:, 1],
                center[:, axis] - original[:, axis],
            )
    return replace(
        prediction,
        center=center.astype(np.float32),
        distance_along_m=distance.astype(np.float32),
        continuation=continuation,
        details={**(prediction.details or {}), **details},
    )


def _wall_modes(points, depths, heights, middle, slope=0.0, height_band=(0.65, 1.65)):
    good = (
        (depths >= middle - 2.5)
        & (depths < middle + 2.5)
        & (heights > height_band[0])
        & (heights < height_band[1])
    )
    p = points[good]
    h = heights[good]
    if len(p) < 8:
        return []
    projected_x = p[:, 0] - slope * (depths[good] - middle)
    bins = np.arange(
        np.floor(projected_x.min() / 0.1) * 0.1 - 0.1, projected_x.max() + 0.21, 0.1
    )
    counts, edges = np.histogram(projected_x, bins)
    smooth = gaussian_filter1d(counts.astype(float), 1.0)
    peaks, _ = find_peaks(np.r_[0.0, smooth, 0.0], height=2.0)
    modes = []
    for peak in peaks - 1:
        x = (edges[peak] + edges[peak + 1]) / 2
        near = abs(projected_x - x) < 0.2
        if near.sum() < 6 or np.ptp(h[near]) < 0.25:
            continue
        modes.append(
            (float(np.median(projected_x[near])), float(smooth[peak]), int(near.sum()))
        )
    return modes


def _wall_constraint(
    xyz,
    prediction,
    mode="both",
    strength=1.0,
    project_depth=False,
    seed_consistent=False,
    height_band=(0.65, 1.65),
    clearance_target=1.6,
):
    c = prediction.center
    if len(c) < 2 or prediction.rail_points is None:
        return prediction, {}
    anchors = _segmented_anchors(c, prediction.rail_points)
    if len(anchors) < 5:
        return prediction, {}
    cd = -c[:, 1]
    grid = np.arange(7.5, min(202.5, -c[-1, 1]), 5.0)
    d = -xyz[:, 1]
    h = xyz[:, 2] - np.interp(d, cd, c[:, 2])
    use = (d >= 5) & (d <= 205) & (h > height_band[0]) & (h < height_band[1])
    p = xyz[use]
    depth = d[use]
    height = h[use]
    cx = np.interp(grid, cd, c[:, 0])
    slope = np.interp(grid, cd, np.gradient(c[:, 0], cd))
    modes = [
        _wall_modes(
            p, depth, height, s, slope[i] if project_depth else 0.0, height_band
        )
        for i, s in enumerate(grid)
    ]
    profiles = []
    offsets = []
    supports = []
    deficits = []
    for side in (-1, 1):
        seed = []
        pool = []
        for i in range(min(4, len(grid))):
            candidates = [m for m in modes[i] if 1.0 < side * (m[0] - cx[i]) < 6.0]
            if not candidates:
                continue
            # Prefer a populated continuous enclosure surface over a lone pole.
            candidates.sort(key=lambda m: m[1], reverse=True)
            value = candidates[0][0]
            seed.append((i, value, (value - cx[i]) / np.sqrt(1 + slope[i] ** 2)))
            for value, weight, _ in candidates:
                pool.append(
                    (i, value, (value - cx[i]) / np.sqrt(1 + slope[i] ** 2), weight)
                )
        if seed_consistent and pool:
            options = []
            for candidate in pool:
                near = [a for a in pool if abs(a[2] - candidate[2]) < 0.3]
                bins = {a[0] for a in near}
                if len(bins) < 3:
                    continue
                chosen = [
                    min(
                        (a for a in near if a[0] == i),
                        key=lambda a: abs(a[2] - candidate[2]),
                    )
                    for i in sorted(bins)
                ]
                options.append(
                    (
                        (-len(bins), abs(candidate[2]), -sum(a[3] for a in chosen)),
                        chosen,
                    )
                )
            if not options:
                continue
            seed = [a[:3] for a in min(options, key=lambda a: a[0])[1]]
        if len(seed) < 3:
            continue
        offset = float(np.median([a[2] for a in seed]))
        if np.median(abs(np.array([a[2] for a in seed]) - offset)) > 0.3:
            continue
        xs = np.array([a[1] for a in seed])
        ds = np.array([grid[a[0]] for a in seed])
        fit = np.polyfit(ds - ds[-1], xs, 1)
        state = np.array([xs[-1], fit[0]])
        cov = np.diag([0.1**2, 0.01**2])
        observed = np.full(len(grid), np.nan)
        vel = np.full(len(grid), np.nan)
        last = ds[-1]
        for i, x, _ in seed:
            observed[i] = x
            vel[i] = fit[0]
        last_supported = seed[-1][0]
        for i in range(seed[-1][0] + 1, len(grid)):
            delta = grid[i] - last
            f = np.array([[1.0, delta], [0.0, 1.0]])
            estimate = f @ state
            unc = f @ cov @ f.T + 1e-5 * np.array(
                [[delta**4 / 4, delta**3 / 2], [delta**3 / 2, delta**2]]
            )
            candidates = [
                m
                for m in modes[i]
                if abs(m[0] - estimate[0]) < min(1.2, 0.5 + 0.1 * (i - last_supported))
            ]
            if candidates:
                value = min(candidates, key=lambda m: abs(m[0] - estimate[0]))[0]
                gain = unc[:, 0] / (unc[0, 0] + 0.08**2)
                state = estimate + gain * (value - estimate[0])
                cov = unc - gain[:, None] * unc[0:1, :]
                observed[i] = state[0]
                vel[i] = state[1]
                last_supported = i
            else:
                state = estimate
                cov = unc
            last = grid[i]
            if i - last_supported > 4:
                break
        valid = np.isfinite(observed)
        if valid.sum() < 6:
            continue
        # Short gaps are interpolation, not evidence extending the horizon.
        profile = np.interp(grid, grid[valid], observed[valid])
        v = np.interp(grid, grid[valid], vel[valid])
        distance = np.min(abs(grid[:, None] - grid[valid][None]), axis=1)
        supported = (
            (grid >= grid[valid][0]) & (grid <= grid[valid][-1]) & (distance <= 7.5)
        )
        profiles.append(profile - offset * np.sqrt(1 + v * v))
        offsets.append(offset)
        supports.append(supported)
        clearance = np.sign(offset) * (profile - cx) / np.sqrt(1 + v * v)
        deficits.append(abs(offset) - clearance)
    if not profiles:
        return prediction, {}
    estimate = np.full(len(grid), np.nan)
    if len(profiles) == 2:
        both = supports[0] & supports[1] & (abs(profiles[0] - profiles[1]) < 0.5)
        estimate[both] = (profiles[0][both] + profiles[1][both]) / 2
        if mode in ("single", "gated", "clearance"):
            for i in (0, 1):
                only = supports[i] & ~supports[1 - i]
                estimate[only] = profiles[i][only]
    elif mode in ("single", "gated", "clearance"):
        estimate[supports[0]] = profiles[0][supports[0]]
    valid = np.isfinite(estimate)
    if valid.sum() < 4:
        return prediction, dict(
            offsets=offsets, profiles=profiles, grid=grid, supported=False
        )
    residual = estimate[valid] - cx[valid]
    if mode in ("gated", "clearance"):
        # Correct only a loss of the clearance learned from nearby rails.
        # Do not pull an already clear path toward a wall merely to impose a
        # constant tunnel offset. A trusted wall must push the estimate AWAY
        # from itself; disagreement between visible walls remains unsupported.
        gate = np.zeros(len(grid))
        for offset, deficit, supported in zip(offsets, deficits, supports):
            away = (estimate - cx) * (-np.sign(offset)) > 0
            gate = np.maximum(
                gate,
                np.where(supported & away, np.clip((deficit - 0.25) / 0.5, 0, 1), 0),
            )
        if mode == "clearance":
            # Use walls as a feasibility constraint, not an assumption that
            # tunnel width remains constant. Only repair an actual loss of
            # clearance; wider sections must not pull the path sideways.
            constraints = []
            for offset, deficit, supported in zip(offsets, deficits, supports):
                clearance = abs(offset) - deficit
                target = min(abs(offset), clearance_target)
                push = -np.sign(offset) * np.maximum(0.0, target - clearance)
                active = supported & (clearance < 1.25) & ((estimate - cx) * push > 0)
                constraints.append(np.where(active, push, 0.0))
            correction = np.zeros(len(grid))
            for push in constraints:
                select = abs(push) > abs(correction)
                correction[select] = push[select]
            residual = correction[valid]
        else:
            residual *= gate[valid]
    # Robust local smoothing suppresses a single narrow object/pillar bin.
    for i in range(1, len(residual) - 1):
        if abs(residual[i] - (residual[i - 1] + residual[i + 1]) / 2) > 0.4:
            residual[i] = (residual[i - 1] + residual[i + 1]) / 2
    residual = gaussian_filter1d(residual, 0.8)
    gap = np.min(abs(cd[:, None] - grid[valid][None]), axis=1)
    protected_end = (
        min(60.0, max(25.0, float(-anchors[-1, 1])))
        if mode in ("gated", "clearance")
        else 25.0
    )
    weight = (
        np.clip((cd - protected_end) / 15, 0, 1)
        * np.clip((grid[valid][-1] + 10 - cd) / 10, 0, 1)
        * (gap < 10)
    )
    center = c.copy()
    center[:, 0] += (
        strength * weight * np.interp(cd, grid[valid], np.clip(residual, -5, 5))
    )
    return _replace_center(
        prediction,
        center,
        {
            "wall_tracking": dict(
                offsets=offsets,
                mode=mode,
                strength=strength,
                project_depth=project_depth,
                seed_consistent=seed_consistent,
                height_band=height_band,
                clearance_target=clearance_target,
                supported_end_m=float(grid[valid][-1]),
            )
        },
    ), dict(
        offsets=offsets,
        profiles=profiles,
        grid=grid,
        supported=valid,
        estimate=estimate,
    )


def _validated_wall(xyz, prediction, clearance_target=2.0):
    """Validate a lower-wall constraint with an independent upper-wall profile.

    Unlike dual-wall intersection, an upper wall need not itself be colliding:
    its inferred track position must support the lower wall's proposed change.
    """
    lower, ld = _wall_constraint(
        xyz,
        prediction,
        "clearance",
        project_depth=True,
        seed_consistent=True,
        clearance_target=clearance_target,
    )
    upper, ud = _wall_constraint(
        xyz,
        prediction,
        "clearance",
        project_depth=True,
        seed_consistent=True,
        height_band=(1.8, 3.5),
        clearance_target=clearance_target,
    )
    if "estimate" not in ld or "estimate" not in ud:
        return prediction, {}
    grid = ld["grid"]
    a = ld["estimate"]
    b = ud["estimate"]
    cx = np.interp(grid, -prediction.center[:, 1], prediction.center[:, 0])
    valid = (
        np.isfinite(a) & np.isfinite(b) & (abs(a - b) < 0.5) & ((a - cx) * (b - cx) > 0)
    )
    # Agreement needs a run of at least three stations (10 m), not one pole.
    keep = np.zeros(len(valid), bool)
    for chain in np.split(
        np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1) + 1
    ):
        if len(chain) >= 3:
            keep[chain] = True
    if not keep.any():
        return prediction, {}
    confidence = np.interp(
        -prediction.center[:, 1], grid, keep.astype(float), left=0, right=0
    )
    c = prediction.center.copy()
    c[:, 0] += confidence * (lower.center[:, 0] - c[:, 0])
    return _replace_center(
        prediction,
        c,
        {
            "validated_wall": dict(
                clearance_target=clearance_target,
                agreement_stations=int(keep.sum()),
                lower=lower.details.get("wall_tracking"),
                upper=upper.details.get("wall_tracking"),
            )
        },
    ), dict(lower=ld, upper=ud, agreement=keep)


@lru_cache(maxsize=1)
def _curve_design():
    depth = np.arange(1.0, 201.0)
    knots = np.arange(0.0, 201.0, 25.0)
    grid = np.arange(0.0, 201.0)
    cols = [np.ones(len(grid)), grid]
    for k in range(len(knots)):
        hat = np.zeros(len(knots))
        hat[k] = 1.0
        curvature = np.interp(grid, knots, hat)
        heading = np.r_[0.0, np.cumsum((curvature[1:] + curvature[:-1]) / 2)]
        cols.append(np.r_[0.0, np.cumsum((heading[1:] + heading[:-1]) / 2)])
    design = np.column_stack(cols)[1:]
    return depth, knots, design


def _smooth_wall_constraint(base, candidate):
    """Fit a smooth, bounded-curvature path to the model and wall constraint.

    Curvature knots and the 200 m minimum radius follow the model decoder's
    existing prior. This is spatial fitting within the current scan.
    """
    from scipy.optimize import lsq_linear

    delta = candidate.center[:, 0] - base.center[:, 0]
    if np.max(abs(delta), initial=0) < 0.05:
        return base
    depth, knots, design = _curve_design()
    cd = -base.center[:, 1]
    original = np.interp(depth, cd, base.center[:, 0])
    target = np.interp(depth, cd, candidate.center[:, 0])
    active = abs(target - original) > 0.05
    weight = np.where(active, 12.5, 2.0)
    weight[depth <= 25] = 50.0
    low = np.r_[-2.0, -0.3, np.full(len(knots), -0.005)]
    high = -low
    fit = lsq_linear(
        design * weight[:, None],
        target * weight,
        bounds=(low, high),
        method="bvls",
        tol=1e-8,
    )
    if not fit.success:
        return base
    measured = design @ fit.x
    center = base.center.copy()
    blend = np.clip((cd - 25) / 15, 0, 1)
    center[:, 0] += blend * (np.interp(cd, depth, measured) - center[:, 0])
    return _replace_center(
        base,
        center,
        {
            "_smooth_wall_constraint": dict(
                curvature_limit=0.005, knots_m=25.0, fit_cost=float(fit.cost)
            )
        },
    )


def _terrain_clearance(xyz, selected, center, half_window=6.0):
    """Deterministic local ground-plane fits, with bilateral / long support.

    A compact object cannot alone establish a plane: inliers must span the
    track width and at least 5 m longitudinally. Unsupported cells return NaN.
    """
    result = np.full(len(selected), np.nan, np.float32)
    if not len(selected) or len(center) < 2:
        return result
    depth = -xyz[:, 1]
    cd = -center[:, 1]
    cx = np.interp(depth, cd, center[:, 0])
    cz = np.interp(depth, cd, center[:, 2])
    mask = (
        (depth >= 0)
        & (depth <= 210)
        & (abs(xyz[:, 0] - cx) < 2)
        & (xyz[:, 2] > cz - 0.55)
        & (xyz[:, 2] < cz + 0.55)
    )
    p = xyz[mask]
    d = depth[mask]
    _, idx = np.unique(
        np.floor(p / [0.08, 0.15, 0.04]).astype(np.int32), axis=0, return_index=True
    )
    p = p[idx]
    d = d[idx]
    order = np.argsort(d)
    p = p[order]
    d = d[order]
    for bucket in np.unique(np.floor(-selected[:, 1] / 5).astype(int)):
        mid = bucket * 5 + 2.5
        ids = np.flatnonzero(np.floor(-selected[:, 1] / 5).astype(int) == bucket)
        a = p[
            np.searchsorted(d, mid - half_window) : np.searchsorted(
                d, mid + half_window
            )
        ]
        if len(a) < 30:
            continue
        # Fixed deterministic samples ensure reproducible fitting and bounded work.
        a = a[:: max(1, len(a) // 2000)]
        rng = np.random.default_rng(8192 + int(bucket))
        triples = rng.integers(0, len(a), (80, 3))
        v = a[triples]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        valid = abs(n[:, 2]) > 1e-5
        n = n[valid]
        v = v[valid]
        coeff = -n[:, :2] / n[:, 2, None]
        intercept = v[:, 0, 2] - np.sum(coeff * v[:, 0, :2], axis=1)
        valid = (abs(coeff[:, 0]) < 0.15) & (abs(coeff[:, 1]) < 0.15)
        coeff = coeff[valid]
        intercept = intercept[valid]
        if not len(coeff):
            continue
        error = abs(a[:, 2, None] - (a[:, :2] @ coeff.T + intercept))
        inlier = error < 0.035
        count = inlier.sum(0)
        for index in np.argsort(count)[::-1][:8]:
            good = a[inlier[:, index]]
            localx = good[:, 0] - np.interp(-good[:, 1], cd, center[:, 0])
            if (
                len(good) < 25
                or np.ptp(good[:, 1]) < 5
                or localx.min() > -0.5
                or localx.max() < 0.5
            ):
                continue
            # Least squares uses only the consensus plane, not obstacle tops.
            mat = np.column_stack([good[:, :2], np.ones(len(good))])
            fit = np.linalg.lstsq(mat, good[:, 2], rcond=None)[0]
            result[ids] = selected[ids, 2] - (selected[ids, :2] @ fit[:2] + fit[2])
            break
    return result


def _floor_candidate(xyz, prediction, half_window=6.0):
    c = prediction.center
    if len(c) < 2 or prediction.rail_points is None:
        return prediction, {}
    anchors = _segmented_anchors(c, prediction.rail_points)
    if len(anchors) < 5:
        return prediction, {}
    cd = -c[:, 1]
    d = np.arange(7.5, min(200.0, cd[-1]), 5.0)
    query = np.column_stack([np.interp(d, cd, c[:, 0]), -d, np.interp(d, cd, c[:, 2])])
    clearance = _terrain_clearance(xyz, query, c, half_window=half_window)
    near = (
        np.isfinite(clearance) & (d <= 27.5) & (clearance >= 0.02) & (clearance <= 0.4)
    )
    if near.sum() < 3:
        return prediction, {}
    reference = float(np.median(clearance[near]))
    if np.median(abs(clearance[near] - reference)) > 0.04:
        return prediction, {}
    valid = np.isfinite(clearance) & (abs(clearance - reference) < 0.4)
    shift = reference - clearance
    # Only a clearly too-low rail level, with direct local floor evidence.
    valid &= (d > 25) & (shift > 0.08)
    if not valid.any():
        return prediction, dict(reference_clearance=reference)
    shift = np.where(valid, np.clip(shift, 0, 0.4), 0.0)
    shift = gaussian_filter1d(shift, 0.7)
    center = c.copy()
    center[:, 2] += np.interp(cd, d, shift, left=0, right=0)
    return _replace_center(
        prediction,
        center,
        {
            "floor_height": dict(
                reference_clearance=reference, stations=int(valid.sum())
            )
        },
    ), dict(depth=d, clearance=clearance, shift=shift)


def _bound_floor_to_rails(prediction, floor_prediction, min_correction=0.0):
    """Do not lift the rail level above the locally observed rail-top band."""
    c = prediction.center
    rails = prediction.rail_points
    if rails is None or len(rails) < 3:
        return prediction
    rails = np.unique(rails, axis=0)
    d = -rails[:, 1]
    cd = -c[:, 1]
    x = rails[:, 0] - np.interp(d, cd, c[:, 0])
    h = rails[:, 2] - np.interp(d, cd, c[:, 2])
    good = (abs(x) > 0.35) & (abs(x) < 1.2) & (abs(h) < 0.5) & np.isfinite(rails).all(1)
    d = d[good]
    h = h[good]
    cap = np.zeros(len(c))
    order = np.argsort(d)
    d = d[order]
    h = h[order]
    for i, station in enumerate(cd):
        lo, hi = np.searchsorted(d, [station - 7.5, station + 7.5])
        if hi - lo < 3 or d[hi - 1] - d[lo] < 2:
            continue
        cap[i] = max(0.0, float(np.quantile(h[lo:hi], 0.75)) + 0.02)
    shift = np.minimum(np.maximum(0.0, floor_prediction.center[:, 2] - c[:, 2]), cap)
    shift[cap < 0.03] = 0.0
    if min_correction > 0:
        accepted = shift >= min_correction
        supported = np.zeros(len(shift), bool)
        ids = np.flatnonzero(accepted)
        for chain in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1):
            if len(chain) and cd[chain[-1]] - cd[chain[0]] >= 5.0:
                supported[chain] = True
        shift[~supported] = 0.0
    center = c.copy()
    center[:, 2] += shift
    return _replace_center(
        prediction,
        center,
        {
            "supported_floor": dict(
                max_correction_m=float(shift.max()),
                rail_top_quantile=0.75,
                rail_tolerance_m=0.02,
            )
        },
    )


def _supported_floor(xyz, prediction, min_correction=0.0):
    floor, info = _floor_candidate(xyz, prediction)
    return _bound_floor_to_rails(prediction, floor, min_correction=min_correction), info


def _component_labels(points, config=ObstacleFilterConfig(), pair_budget=100000):
    """Exact adaptive-radius components, or None for deliberately skipped dense input."""
    if not len(points):
        return np.empty(0, dtype=int)
    # Optional rejection must not add a dense-neighborhood cost at contact.
    # Keeping all proposals is the conservative fallback on very large sets.
    if len(points) > 20_000:
        return None
    canonical = np.ascontiguousarray(points + 0, dtype=np.float32)
    _, first, inverse = np.unique(
        canonical.view("V12").ravel(), return_index=True, return_inverse=True
    )
    p = points[first]
    radius = np.interp(-p[:, 1], config.distance_knots_m, config.radius_knots_m)
    tree = cKDTree(p)
    if tree.count_neighbors(tree, radius.max()) > 2 * pair_budget + len(p):
        return None
    pairs = tree.query_pairs(radius.max(), output_type="ndarray")
    delta = p[pairs[:, 0]] - p[pairs[:, 1]]
    pairs = pairs[
        np.einsum("ij,ij->i", delta, delta)
        <= np.minimum(radius[pairs[:, 0]], radius[pairs[:, 1]]) ** 2
    ]
    graph = coo_matrix(
        (np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(len(p), len(p))
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    return labels[inverse]


def _reject_ground_components(
    points,
    corrected_center,
    strip_only=True,
    config=ObstacleFilterConfig(),
    labels=None,
):
    if labels is None:
        labels = _component_labels(points, config)
    if labels is None or not len(points):
        return np.ones(len(points), bool)
    _, near = cKDTree(corrected_center[:, :2]).query(points[:, :2])
    threshold = np.where(
        -points[:, 1] < config.near_height_distance_m,
        config.near_min_height_above_track_m,
        config.min_height_above_track_m,
    )
    above = points[:, 2] - corrected_center[near, 2] > threshold
    n = labels.max() + 1
    has_above = np.zeros(n, bool)
    np.logical_or.at(has_above, labels, above)
    reject = ~has_above
    if strip_only:
        lo = np.full((n, 3), np.inf)
        hi = np.full((n, 3), -np.inf)
        np.minimum.at(lo, labels, points)
        np.maximum.at(hi, labels, points)
        size = hi - lo
        reject &= (size[:, 0] > 0.15) & (size[:, 1] < 0.2) & (size[:, 2] < 0.02)
    return ~reject[labels]


def verification_geometry(xyz, prediction, proposals, config):
    """Return an independent wall estimate and a whole-component floor mask.

    The caller must classify the full scan under the returned wall geometry
    before intersecting its detections with the initial proposals. Clustering
    the proposal subset alone would tighten the existing 10 m boundary rule.
    """
    keep = np.ones(len(proposals), bool)
    center = prediction.center
    if (
        not len(proposals)
        or len(center) < 2
        or prediction.rail_points is None
        or not np.isfinite(center).all()
        or np.any(np.diff(-center[:, 1]) <= 0)
    ):
        return prediction, keep
    wall = prediction
    if config.verify_wall_geometry:
        candidate = _validated_wall(xyz, prediction)[0]
        wall = _smooth_wall_constraint(prediction, candidate)
    if config.reject_trackbed_strips:
        # Ground fitting cannot reject a non-strip component. Check this cheap
        # necessary condition first, especially on dense nearby obstacles.
        labels = _component_labels(proposals, config)
        if labels is not None and len(labels):
            n = labels.max() + 1
            lo, hi = np.full((n, 3), np.inf), np.full((n, 3), -np.inf)
            np.minimum.at(lo, labels, proposals)
            np.maximum.at(hi, labels, proposals)
            extent = hi - lo
            possible = (
                (extent[:, 0] > 0.15) & (extent[:, 1] < 0.2) & (extent[:, 2] < 0.02)
            )
            if possible.any():
                floor = _supported_floor(xyz, prediction)[0]
                if not np.array_equal(floor.center, prediction.center):
                    keep = _reject_ground_components(
                        proposals, floor.center, True, config, labels
                    )
    return wall, keep
