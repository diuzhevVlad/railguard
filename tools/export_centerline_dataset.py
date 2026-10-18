"""Export cached model centerlines, merged-rail labels, and relative sensor poses.

The source bags are used only to validate the existing prediction caches; scan
payloads are never read. Each output NPZ contains one bag's complete frame set.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from obstacle_app.detection import MergedRailPredictor
from obstacle_app.merged_rails import MergedRailMap
from obstacle_app.prediction_cache import load_prediction_cache
from railpath.detection_config import load_obstacle_detection_config


def pack_centerlines(centerlines: list) -> dict[str, np.ndarray]:
    """Store variable-length sensor-frame XYZ curves and their arc distances."""
    counts = np.asarray([len(item.center) for item in centerlines], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    centers = [np.asarray(item.center, dtype=np.float32) for item in centerlines]
    distances = [np.asarray(item.distance_along_m, dtype=np.float32)
                 for item in centerlines]
    for center, distance in zip(centers, distances):
        if (center.shape != (len(distance), 3)
                or not np.isfinite(center).all()
                or not np.isfinite(distance).all()
                or np.any(np.diff(distance) < 0)):
            raise ValueError("Invalid centerline or arc-distance samples")
    return {
        "xyz": np.concatenate(centers) if centers else np.empty((0, 3), np.float32),
        "distance_along_m": (
            np.concatenate(distances) if distances else np.empty(0, np.float32)),
        "offsets": offsets,
    }


def relative_sensor_poses(poses: np.ndarray) -> np.ndarray:
    """Return T_start_sensor for each saved T_odometry_sensor pose."""
    poses = np.asarray(poses, dtype=np.float64)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses)
            or not np.isfinite(poses).all()
            or not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-6)):
        raise ValueError("Expected finite homogeneous 4x4 poses")
    return np.linalg.inv(poses[0]) @ poses


def build_bag_arrays(bag: Path, cache_dir: Path, weights: Path,
                     max_forward_m: float) -> tuple[dict[str, np.ndarray], dict]:
    scenario = json.loads((bag / "obstacle_scenario.json").read_text(encoding="utf-8"))
    topic = scenario.get("source_topic", "/lidar_points")
    with np.load(bag / "merged_rails.npz", allow_pickle=False) as saved:
        timestamps = saved["timestamps_ns"].copy()
        poses = saved["poses"].copy()
        rail_map = MergedRailMap(saved["center"].copy(), saved["lengths"].copy())
    if (timestamps.ndim != 1 or not len(timestamps)
            or np.any(np.diff(timestamps) <= 0)
            or poses.shape != (len(timestamps), 4, 4)):
        raise ValueError(f"Invalid timestamp/pose sequence in {bag}")

    cached = load_prediction_cache(cache_dir, bag, topic, weights)
    if cached is None:
        raise FileNotFoundError(f"No valid prediction cache for {bag} on {topic}")
    if set(cached) != set(map(int, timestamps)):
        raise ValueError(f"Prediction cache timestamps differ from saved poses: {bag}")

    detected = pack_centerlines([cached[int(t)] for t in timestamps])
    predictor = MergedRailPredictor(rail_map)
    ground_truth = pack_centerlines(
        [predictor.predict(pose, max_forward_m) for pose in poses])
    relative = relative_sensor_poses(poses)
    arrays = {
        "timestamps_ns": timestamps.astype(np.int64),
        "sensor_pose_start": relative,
        "sensor_position_start_m": relative[:, :3, 3].copy(),
        "detected_xyz_sensor_m": detected["xyz"],
        "detected_distance_along_m": detected["distance_along_m"],
        "detected_offsets": detected["offsets"],
        "gt_xyz_sensor_m": ground_truth["xyz"],
        "gt_distance_along_m": ground_truth["distance_along_m"],
        "gt_offsets": ground_truth["offsets"],
    }
    summary = {
        "bag": str(bag),
        "topic": topic,
        "frames": len(timestamps),
        "detected_points": len(detected["xyz"]),
        "gt_points": len(ground_truth["xyz"]),
    }
    return arrays, summary


def export_dataset(bags: list[Path], output_dir: Path, cache_dir: Path,
                   weights: Path, max_forward_m: float) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if len({bag.name for bag in bags}) != len(bags):
        raise ValueError("Bag names must be unique for per-bag output filenames")
    if not np.isfinite(max_forward_m) or max_forward_m <= 0:
        raise ValueError("max_forward_m must be finite and positive")

    # Validate all sources before writing the new dataset.
    prepared = [build_bag_arrays(bag, cache_dir, weights, max_forward_m)
                for bag in bags]
    output_dir.mkdir(parents=True)
    summaries = []
    for bag, (arrays, summary) in zip(bags, prepared):
        filename = f"{bag.name}.npz"
        np.savez_compressed(output_dir / filename, **arrays)
        summaries.append({**summary, "file": filename})
        print(f"{bag.name}: {summary['frames']} frames -> {filename}", flush=True)

    manifest = {
        "schema": "rail-centerline-dataset-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "units": "metres; nanosecond ROS bag timestamps",
        "sensor_coordinates": "LiDAR XYZ; -Y forward, +Z up",
        "detected_centerline": "Cached learned RailGuard prediction for that scan",
        "ground_truth_centerline": (
            "Merged paired-rail centerline transformed by the saved KISS-ICP "
            "pose into the current sensor frame; label quality depends on the "
            "merged rail map and odometry."),
        "sensor_pose_start": (
            "T_start_sensor = inverse(T_odometry_sensor_first) @ "
            "T_odometry_sensor_current; maps current sensor coordinates into "
            "the first frame's sensor coordinates."),
        "sensor_position_start_m": "Translation column of T_start_sensor",
        "ragged_centerlines": (
            "Frame i uses xyz[offsets[i]:offsets[i+1]] and matching "
            "distance_along_m[offsets[i]:offsets[i+1]]."),
        "max_gt_forward_m": max_forward_m,
        "bags": summaries,
        "total_frames": sum(item["frames"] for item in summaries),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="*", type=Path,
                        help="Generated bag directories; defaults to the filter-matrix report")
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/filter_matrix_all_generated_bags.json"))
    parser.add_argument("--prediction-cache", type=Path,
                        default=Path("outputs/eval_corpus_all_real_bags/prediction_cache"))
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--config", type=Path,
                        default=Path("railguard/config/railguard.yaml"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/centerline_dataset_all_generated"))
    args = parser.parse_args()
    bags = args.bags or [Path(item["bag"]) for item in json.loads(
        args.report.read_text(encoding="utf-8"))["bags"]]
    config = load_obstacle_detection_config(args.config)
    manifest = export_dataset(
        bags, args.output_dir, args.prediction_cache, args.weights,
        config.max_forward_m)
    print(f"Exported {manifest['total_frames']} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
