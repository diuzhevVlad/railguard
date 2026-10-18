"""Fit RollingNURBSKF's detector noise from exported centerline NPZ files.

Samples the detected curve at fixed arc distances, projects them onto the
ground-truth polyline, and fits the covariance of signed normal errors.
Only the detector fields of ``Noise`` are identifiable from this dataset;
process, birth, and odometry uncertainty require different reference data.

Example:
    python calibrate_noise.py centerline_dataset_all_generated --output detector_noise.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def sample_detector(xyz: np.ndarray, distance: np.ndarray,
                    targets: np.ndarray) -> np.ndarray:
    """Interpolate sensor XY on the detector's exported arc-distance grid."""
    good = np.isfinite(distance) & np.isfinite(xyz[:, :2]).all(axis=1)
    distance, xy = distance[good], xyz[good, :2]
    if len(distance) < 2:
        return np.empty((0, 2))
    keep = np.r_[True, np.diff(distance) > 1e-6]
    distance, xy = distance[keep], xy[keep]
    return np.column_stack([np.interp(targets, distance, xy[:, axis],
                                      left=np.nan, right=np.nan)
                            for axis in range(2)])


def project_to_gt(points: np.ndarray, gt_xyz: np.ndarray,
                  endpoint_margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed normal errors, normals, and valid interior projections."""
    gt = np.asarray(gt_xyz[:, :2], float)
    steps = np.diff(gt, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    if len(gt) < 3 or not np.isfinite(gt).all() or not np.isfinite(lengths).all():
        return np.zeros(len(points)), np.zeros((len(points), 2)), np.zeros(len(points), bool)
    arc = np.r_[0., np.cumsum(lengths)]
    # A closest vertex has one of its adjacent segments as a candidate. The
    # export uses approximately 0.5 m GT spacing, so this is inexpensive.
    _, vertex = cKDTree(gt).query(points)
    candidates = np.stack([np.clip(vertex - 1, 0, len(steps) - 1),
                           np.clip(vertex, 0, len(steps) - 1)], axis=1)
    start = gt[candidates]
    direction = steps[candidates]
    length2 = np.sum(direction**2, axis=2)
    fraction = np.clip(np.sum((points[:, None, :] - start)*direction, axis=2)
                       / np.maximum(length2, 1e-12), 0., 1.)
    closest = start + fraction[:, :, None]*direction
    squared = np.sum((points[:, None, :] - closest)**2, axis=2)
    choice = np.argmin(squared, axis=1)
    row = np.arange(len(points))
    seg = candidates[row, choice]
    t = fraction[row, choice]
    chosen = closest[row, choice]
    tangent = steps[seg] / np.maximum(lengths[seg, None], 1e-12)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    error = np.sum((points - chosen)*normal, axis=1)
    along = arc[seg] + t*lengths[seg]
    valid = ((lengths[seg] > 1e-6) & (along >= endpoint_margin) &
             (along <= arc[-1] - endpoint_margin) & np.isfinite(error))
    return error, normal, valid


def collect_moments(paths: list[Path], targets: np.ndarray, max_error: float,
                    endpoint_margin: float) -> dict:
    """Collect distance-wise moments and one compact error record per frame."""
    n = len(targets)
    count = np.zeros(n, dtype=np.int64)
    first = np.zeros(n)
    diagonal = np.zeros(n)
    pair_count = np.zeros(n, dtype=np.int64)
    pair_product = np.zeros(n)
    pair_dot = np.zeros(n)
    records = []
    frames = 0
    used_frames = 0
    rejected = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            needed = ("detected_xyz_sensor_m", "detected_distance_along_m",
                      "detected_offsets", "gt_xyz_sensor_m", "gt_offsets")
            missing = set(needed) - set(data.files)
            if missing:
                raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")
            det, dist = data[needed[0]], data[needed[1]]
            det_offsets, gt, gt_offsets = data[needed[2]], data[needed[3]], data[needed[4]]
            if len(det_offsets) != len(gt_offsets) or det_offsets[-1] != len(det) or gt_offsets[-1] != len(gt):
                raise ValueError(f"{path}: inconsistent offsets")
            for frame in range(len(det_offsets) - 1):
                frames += 1
                a, b = det_offsets[frame:frame + 2]
                c, d = gt_offsets[frame:frame + 2]
                if b - a < 2 or d - c < 3:
                    continue
                points = sample_detector(det[a:b], dist[a:b], targets)
                if len(points) != n:
                    continue
                present = np.isfinite(points).all(axis=1)
                if not np.any(present):
                    continue
                idx = np.flatnonzero(present)
                error, normal, valid = project_to_gt(points[idx], gt[c:d], endpoint_margin)
                rejected += int(np.sum(valid & (np.abs(error) > max_error)))
                valid &= np.abs(error) <= max_error
                idx, error, normal = idx[valid], error[valid], normal[valid]
                if len(idx) < 2:
                    continue
                used_frames += 1
                records.append((idx, error, normal))
                count[idx] += 1
                first[idx] += error
                diagonal[idx] += error**2
    mean = np.divide(first, count, out=np.zeros_like(first), where=count > 0)
    for idx, error, normal in records:
        i, j = np.triu_indices(len(idx), 1)
        lag = idx[j] - idx[i]
        centered = error - mean[idx]
        pair_count += np.bincount(lag, minlength=n)
        pair_product += np.bincount(lag, weights=centered[i]*centered[j], minlength=n)
        pair_dot += np.bincount(lag, weights=np.sum(normal[i]*normal[j], axis=1), minlength=n)
    return dict(frames=frames, used_frames=used_frames, rejected=rejected,
                count=count, first=first, diagonal=diagonal,
                pair_count=pair_count, pair_product=pair_product, pair_dot=pair_dot)


def fit_noise(moments: dict, spacing: float) -> dict:
    """Fit the filter's white + shared translation + SE shape covariance."""
    count = moments["count"]
    pairs = moments["pair_count"]
    if count.sum() < 100 or np.count_nonzero(pairs) < 5:
        raise ValueError("Too few matched samples to calibrate the detector")
    # Remove a global mean at each forward distance. This avoids attributing
    # a systematic range bias to a random frame offset.
    mean = np.divide(moments["first"], count, out=np.zeros_like(moments["first"]), where=count > 0)
    variance = np.divide(moments["diagonal"], count, out=np.zeros_like(moments["diagonal"]), where=count > 0) - mean**2
    # Pair products were also centered at these distance-specific means.
    lag = np.arange(len(count))
    valid = (lag > 0) & (pairs >= 50)
    if np.count_nonzero(valid) < 5 or np.count_nonzero(count >= 20) < 2:
        raise ValueError("Too few matched pairs at distinct distances to fit the covariance")
    distance = lag[valid]*spacing
    empirical = moments["pair_product"][valid] / pairs[valid]
    dot = moments["pair_dot"][valid] / pairs[valid]
    diagonal_target = np.average(variance[count >= 20], weights=count[count >= 20])
    scale = max(float(diagonal_target), 1e-6)
    weights = np.sqrt(np.minimum(pairs[valid], 10000) / 10000)

    def residual(x: np.ndarray) -> np.ndarray:
        white2, translation2, shape2, length = x
        predicted = dot*(translation2 + shape2*np.exp(-.5*(distance/length)**2))
        return np.r_[2*(white2 + translation2 + shape2 - diagonal_target)/scale,
                     weights*(predicted - empirical)/scale]

    result = least_squares(residual, [scale*.2, scale*.4, scale*.4, max(2*spacing, 4.)],
                           bounds=([0., 0., 0., spacing*.5],
                                   [np.inf, np.inf, np.inf, max(distance)*4]),
                           max_nfev=1000)
    if not result.success:
        raise RuntimeError(f"Noise fit did not converge: {result.message}")
    white2, translation2, shape2, length = result.x
    return dict(white=float(np.sqrt(white2)),
                frame_translation=float(np.sqrt(translation2)),
                smooth_shape=float(np.sqrt(shape2)),
                correlation_length=float(length),
                normal_bias_m=float(np.average(mean[count >= 20], weights=count[count >= 20])),
                normal_rms_m=float(np.sqrt(diagonal_target)),
                fit_success=bool(result.success), fit_cost=float(result.cost))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("centerline_dataset_all_generated"),
                        help="Dataset directory or one NPZ file")
    parser.add_argument("--output", type=Path, default=Path("detector_noise.json"))
    parser.add_argument("--spacing-m", type=float, default=.5, help="Detector sample spacing")
    parser.add_argument("--max-forward-m", type=float, default=150.)
    parser.add_argument("--max-error-m", type=float, default=2., help="Outlier cutoff for matched normal errors")
    parser.add_argument("--endpoint-margin-m", type=float, default=1.)
    args = parser.parse_args()
    if args.spacing_m <= 0 or args.max_forward_m <= args.spacing_m or args.max_error_m <= 0 or args.endpoint_margin_m < 0:
        parser.error("Use positive spacing, forward distance and error cutoff; endpoint margin must be nonnegative")
    paths = sorted(args.dataset.glob("*.npz")) if args.dataset.is_dir() else [args.dataset]
    if not paths or any(not path.is_file() for path in paths):
        parser.error(f"No NPZ files found at {args.dataset}")
    targets = np.arange(args.spacing_m, args.max_forward_m + 1e-9, args.spacing_m)
    moments = collect_moments(paths, targets, args.max_error_m, args.endpoint_margin_m)
    fitted = fit_noise(moments, args.spacing_m)
    report = {
        "model": "rolling_nurbs.Noise detector covariance, sensor XY normal residuals",
        "noise": {key: fitted[key] for key in ("white", "frame_translation", "smooth_shape", "correlation_length")},
        "diagnostics": {
            "files": len(paths), "frames": moments["frames"], "used_frames": moments["used_frames"],
            "matched_samples": int(moments["count"].sum()), "rejected_large_errors": moments["rejected"],
            "normal_bias_m": fitted["normal_bias_m"], "normal_rms_m": fitted["normal_rms_m"],
            "fit_cost": fitted["fit_cost"], "fit_success": fitted["fit_success"],
        },
        "settings": {"spacing_m": args.spacing_m, "max_forward_m": args.max_forward_m,
                     "max_error_m": args.max_error_m, "endpoint_margin_m": args.endpoint_margin_m},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
