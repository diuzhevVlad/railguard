"""Compare obstacle detection and centerline error with temporal filtering off/on."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial import cKDTree

from obstacle_app.cloud import valid_xyz
from obstacle_app.detection import MergedRailPredictor
from obstacle_app.learned_detection import RailGuardRuntime
from obstacle_app.merged_rails import MergedRailMap
from obstacle_app.prediction_cache import (load_prediction_cache,
                                           save_prediction_cache)
from railpath.detection_config import load_obstacle_detection_config
from railpath.occupancy_filter import ObstacleDetectionPipeline


TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


class ReplayOdometry:
    """Replay the KISS-ICP poses stored with the labeled merged rail map."""

    def __init__(self, poses: np.ndarray):
        self.poses = np.asarray(poses, dtype=np.float64)
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def update(self, _xyz: np.ndarray) -> np.ndarray:
        if self.index >= len(self.poses):
            raise ValueError("Stored KISS-ICP pose sequence is too short")
        pose = self.poses[self.index]
        self.index += 1
        return pose.copy()


def classification_metrics(labels: list[bool], predictions: list[bool]) -> dict:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    tn = sum(not label and not prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def centerline_errors(prediction, ground_truth) -> tuple[np.ndarray, np.ndarray]:
    """Return 3D and XY errors at matching labeled arc distances."""
    distance = np.asarray(prediction.distance_along_m, dtype=np.float64)
    center = np.asarray(prediction.center, dtype=np.float64)
    gt_distance = np.asarray(ground_truth.distance_along_m, dtype=np.float64)
    gt_center = np.asarray(ground_truth.center, dtype=np.float64)
    if not len(distance) or len(gt_distance) < 2:
        return np.empty(0), np.empty(0)
    valid = ((distance >= gt_distance[0]) & (distance <= gt_distance[-1])
             & np.isfinite(center).all(axis=1))
    distance, center = distance[valid], center[valid]
    if not len(distance):
        return np.empty(0), np.empty(0)
    target = np.column_stack([
        np.interp(distance, gt_distance, gt_center[:, axis]) for axis in range(3)
    ])
    difference = center - target
    return np.linalg.norm(difference, axis=1), np.linalg.norm(difference[:, :2], axis=1)


def temporal_jitter(current, previous, current_pose: np.ndarray,
                    previous_pose: np.ndarray) -> np.ndarray:
    """Motion-compensated separation from the preceding centerline."""
    center = np.asarray(current.center, dtype=np.float64)
    prior = np.asarray(previous.center, dtype=np.float64)
    if not len(center) or not len(prior):
        return np.empty(0)
    previous_to_current = np.linalg.inv(current_pose) @ previous_pose
    prior = (prior @ previous_to_current[:3, :3].T
             + previous_to_current[:3, 3])
    separation, _ = cKDTree(prior).query(center)
    # Ignore path sections that appeared or disappeared between frames.
    return separation[np.isfinite(separation) & (separation <= 2.0)]


def error_metrics(errors: list[np.ndarray]) -> dict:
    values = np.concatenate(errors) if errors else np.empty(0)
    if not len(values):
        return {"samples": 0, "mae_m": None, "rmse_m": None, "p95_m": None}
    return {
        "samples": len(values),
        "mae_m": float(np.mean(values)),
        "rmse_m": float(np.sqrt(np.mean(values ** 2))),
        "p95_m": float(np.percentile(values, 95)),
    }


def evaluate_bag(bag: Path, runtime: RailGuardRuntime, config, min_points: int,
                 topic_override: str | None, weights: Path,
                 prediction_cache: Path | None) -> tuple[dict, list[dict], dict]:
    scenario = json.loads((bag / "obstacle_scenario.json").read_text())
    frame_counts = scenario["frame_counts"]
    topic = topic_override or scenario.get("source_topic", "/lidar_points")
    with np.load(bag / "merged_rails.npz", allow_pickle=False) as saved:
        poses = saved["poses"].copy()
        timestamps = saved["timestamps_ns"].copy()
        gt_predictor = MergedRailPredictor(
            MergedRailMap(saved["center"].copy(), saved["lengths"].copy()))
    if poses.shape != (len(timestamps), 4, 4):
        raise ValueError(f"Invalid labeled pose sequence in {bag}")

    without_filter = ObstacleDetectionPipeline(
        config.obstacle, config.bayesian, True, True,
        odometry=ReplayOdometry(poses), centerline_config=config.centerline,
        centerline_filter_enabled=False)
    with_filter = ObstacleDetectionPipeline(
        config.obstacle, config.bayesian, True, True,
        odometry=ReplayOdometry(poses), centerline_config=config.centerline,
        centerline_filter_enabled=True)
    cached = (load_prediction_cache(prediction_cache, bag, topic, weights)
              if prediction_cache is not None else None)
    computed_predictions = []
    frames = []
    raw_3d, filtered_3d, raw_xy, filtered_xy = [], [], [], []
    raw_jitter, filtered_jitter = [], []
    previous_raw = previous_filtered = None

    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        connections = [connection for connection in reader.connections
                       if connection.topic == topic
                       and connection.msgtype == "sensor_msgs/msg/PointCloud2"]
        if len(connections) != 1:
            raise ValueError(f"Expected one PointCloud2 topic {topic!r} in {bag}")
        total = connections[0].msgcount
        for index, (connection, timestamp_ns, rawdata) in enumerate(
                reader.messages(connections=connections)):
            if index >= len(timestamps) or timestamp_ns != timestamps[index]:
                raise ValueError(f"Labeled centerline timestamps do not match {bag}")
            msg = reader.deserialize(rawdata, connection.msgtype)
            if cached is None:
                prediction = runtime.predict(msg)
                computed_predictions.append((timestamp_ns, prediction))
            else:
                try:
                    prediction = cached[timestamp_ns]
                except KeyError as exc:
                    raise ValueError(
                        f"RailPrediction cache misses {timestamp_ns}") from exc
            xyz = valid_xyz(msg)
            baseline = without_filter.update(timestamp_ns, xyz, prediction)
            filtered = with_filter.update(timestamp_ns, xyz, prediction)
            if filtered.centerline_estimate is None:
                raise ValueError("Enabled centerline filter returned no estimate")
            filtered_prediction = filtered.centerline_estimate.prediction
            if index:
                raw_frame_jitter = temporal_jitter(
                    prediction, previous_raw, poses[index], poses[index - 1])
                filtered_frame_jitter = temporal_jitter(
                    filtered_prediction, previous_filtered,
                    poses[index], poses[index - 1])
            else:
                raw_frame_jitter = filtered_frame_jitter = np.empty(0)
            raw_jitter.append(raw_frame_jitter)
            filtered_jitter.append(filtered_frame_jitter)
            previous_raw = prediction
            previous_filtered = filtered_prediction
            ground_truth = gt_predictor.predict(poses[index], config.max_forward_m)
            raw_error, raw_lateral = centerline_errors(prediction, ground_truth)
            filtered_error, filtered_lateral = centerline_errors(
                filtered_prediction, ground_truth)
            raw_3d.append(raw_error)
            filtered_3d.append(filtered_error)
            raw_xy.append(raw_lateral)
            filtered_xy.append(filtered_lateral)
            synthetic_points = int(frame_counts.get(str(timestamp_ns), {}).get(
                "synthetic_points", 0))
            frames.append({
                "timestamp_ns": timestamp_ns,
                "label": synthetic_points >= min_points,
                "synthetic_points": synthetic_points,
                "without_centerline_filter_detected": baseline.detected,
                "with_centerline_filter_detected": filtered.detected,
                "raw_centerline_rmse_m": (float(np.sqrt(np.mean(raw_error ** 2)))
                                            if len(raw_error) else None),
                "filtered_centerline_rmse_m": (
                    float(np.sqrt(np.mean(filtered_error ** 2)))
                    if len(filtered_error) else None),
                "raw_centerline_jitter_rmse_m": (
                    float(np.sqrt(np.mean(raw_frame_jitter ** 2)))
                    if len(raw_frame_jitter) else None),
                "filtered_centerline_jitter_rmse_m": (
                    float(np.sqrt(np.mean(filtered_frame_jitter ** 2)))
                    if len(filtered_frame_jitter) else None),
            })
            completed = index + 1
            if completed == 1 or completed % 25 == 0 or completed == total:
                print(f"{bag.name}: {completed}/{total}", flush=True)

    if len(frames) != len(timestamps):
        raise ValueError(f"Incomplete labeled pose sequence in {bag}")
    if cached is not None and len(cached) != len(frames):
        raise ValueError(f"RailPrediction cache frame count does not match {bag}")
    if prediction_cache is not None and cached is None:
        path = save_prediction_cache(
            prediction_cache, bag, topic, weights, computed_predictions)
        print(f"{bag.name}: cached RailPredictions in {path}", flush=True)

    labels = [frame["label"] for frame in frames]
    summary = {
        "bag": str(bag),
        "frames": len(frames),
        "positive_frames": sum(labels),
        "without_centerline_filter": classification_metrics(
            labels, [frame["without_centerline_filter_detected"] for frame in frames]),
        "with_centerline_filter": classification_metrics(
            labels, [frame["with_centerline_filter_detected"] for frame in frames]),
        "centerline_temporal_jitter": {
            "raw_3d": error_metrics(raw_jitter),
            "filtered_3d": error_metrics(filtered_jitter),
        },
        "centerline_gt": {
            "raw_3d": error_metrics(raw_3d),
            "filtered_3d": error_metrics(filtered_3d),
            "raw_xy": error_metrics(raw_xy),
            "filtered_xy": error_metrics(filtered_xy),
        },
    }
    errors = {"raw_3d": raw_3d, "filtered_3d": filtered_3d,
              "raw_xy": raw_xy, "filtered_xy": filtered_xy,
              "raw_jitter": raw_jitter, "filtered_jitter": filtered_jitter}
    return summary, frames, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--config", type=Path,
                        default=Path("railguard/config/railguard.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topic")
    parser.add_argument("--min-points", type=int)
    parser.add_argument("--centerline-process-variance-m2", type=float)
    parser.add_argument("--centerline-measurement-variance-m2", type=float)
    parser.add_argument(
        "--centerline-measurement-variance-growth-m2-per-m", type=float)
    parser.add_argument("--centerline-max-measurement-variance-m2", type=float)
    parser.add_argument("--centerline-max-correspondence-distance-m", type=float)
    parser.add_argument("--prediction-cache", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/centerline_filter_evaluation.json"))
    args = parser.parse_args()
    config = load_obstacle_detection_config(args.config)
    centerline_overrides = {
        name: value for name, value in {
            "process_variance_m2": args.centerline_process_variance_m2,
            "measurement_variance_m2": args.centerline_measurement_variance_m2,
            "measurement_variance_growth_m2_per_m":
                args.centerline_measurement_variance_growth_m2_per_m,
            "max_measurement_variance_m2":
                args.centerline_max_measurement_variance_m2,
            "max_correspondence_distance_m":
                args.centerline_max_correspondence_distance_m,
        }.items() if value is not None
    }
    if centerline_overrides:
        config = replace(config, centerline=replace(config.centerline, **centerline_overrides))
    min_points = args.min_points or config.obstacle.min_cluster_points
    runtime = RailGuardRuntime(args.weights, args.device)
    summaries, all_frames = [], []
    combined_errors = {name: [] for name in
                       ("raw_3d", "filtered_3d", "raw_xy", "filtered_xy",
                        "raw_jitter", "filtered_jitter")}
    for bag in args.bags:
        summary, frames, errors = evaluate_bag(
            bag, runtime, config, min_points, args.topic, args.weights,
            args.prediction_cache)
        summaries.append(summary)
        all_frames.extend({"bag": bag.name, **frame} for frame in frames)
        for name, values in errors.items():
            combined_errors[name].extend(values)

    labels = [frame["label"] for frame in all_frames]
    without = classification_metrics(
        labels, [frame["without_centerline_filter_detected"] for frame in all_frames])
    filtered = classification_metrics(
        labels, [frame["with_centerline_filter_detected"] for frame in all_frames])
    raw_gt = error_metrics(combined_errors["raw_3d"])
    filtered_gt = error_metrics(combined_errors["filtered_3d"])
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_definition": f"synthetic_points >= {min_points}",
        "centerline_gt_definition": (
            "Per-frame local centerline from merged paired-rail labels and stored "
            "KISS-ICP poses; errors use common arc-distance samples only."),
        "comparison_control": (
            "Bayesian occupancy and replayed KISS-ICP poses are enabled in both "
            "variants; only centerline_filter_enabled changes."),
        "parameters": config.to_parameters(),
        "bags": summaries,
        "combined": {
            "frames": len(all_frames),
            "positive_frames": sum(labels),
            "negative_frames": len(labels) - sum(labels),
            "without_centerline_filter": without,
            "with_centerline_filter": filtered,
            "precision_delta": filtered["precision"] - without["precision"],
            "recall_delta": filtered["recall"] - without["recall"],
            "centerline_temporal_jitter": {
                "raw_3d": error_metrics(combined_errors["raw_jitter"]),
                "filtered_3d": error_metrics(combined_errors["filtered_jitter"]),
            },
            "centerline_gt": {
                "raw_3d": raw_gt,
                "filtered_3d": filtered_gt,
                "raw_xy": error_metrics(combined_errors["raw_xy"]),
                "filtered_xy": error_metrics(combined_errors["filtered_xy"]),
                "rmse_delta_m": (filtered_gt["rmse_m"] - raw_gt["rmse_m"]
                                 if raw_gt["rmse_m"] is not None else None),
            },
        },
        "frame_results": all_frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["combined"], indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
