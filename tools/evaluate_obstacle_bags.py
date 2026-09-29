"""Evaluate raw and Bayesian obstacle detection on generated labeled bags."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from obstacle_app.cloud import valid_xyz, xyz_view
from obstacle_app.learned_detection import RailGuardRuntime
from obstacle_app.prediction_cache import (load_prediction_cache,
                                           save_prediction_cache)
from railpath.detection_config import load_obstacle_detection_config
from railpath.occupancy_filter import ObstacleDetectionPipeline


TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def metrics(labels: list[bool], predictions: list[bool]) -> dict[str, int | float]:
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


def synthetic_distance_m(msg: object, count: int) -> float | None:
    if count <= 0:
        return None
    cloud = xyz_view(msg).reshape(-1)
    tail = cloud[-count:]
    xyz = np.column_stack([tail[name] for name in ("x", "y", "z")])
    forward = -xyz[np.isfinite(xyz).all(axis=1), 1]
    forward = forward[forward > 0]
    return float(np.min(forward)) if len(forward) else None


def distance_bin(distance_m: float | None) -> str:
    if distance_m is None:
        return "unknown"
    if distance_m < 30.0:
        return "0-30m"
    if distance_m < 80.0:
        return "30-80m"
    return "80m+"


def evaluate_bag(bag: Path, runtime: RailGuardRuntime, config,
                 min_points: int, topic_override: str | None, weights: Path,
                 prediction_cache: Path | None) -> tuple[dict, list[dict]]:
    scenario_path = bag / "obstacle_scenario.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    frame_counts = scenario["frame_counts"]
    topic = topic_override or scenario.get("source_topic", "/lidar_points")
    raw = ObstacleDetectionPipeline(config.obstacle, config.bayesian, False, False)
    bayesian = ObstacleDetectionPipeline(config.obstacle, config.bayesian, True, True)
    frames: list[dict] = []
    cached = (load_prediction_cache(prediction_cache, bag, topic, weights)
              if prediction_cache is not None else None)
    computed_predictions = []
    if cached is not None:
        print(f"{bag.name}: loaded {len(cached)} cached RailPredictions", flush=True)

    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        connections = [connection for connection in reader.connections
                       if connection.topic == topic
                       and connection.msgtype == "sensor_msgs/msg/PointCloud2"]
        if not connections:
            raise ValueError(f"No PointCloud2 topic {topic!r} in {bag}")
        total = sum(connection.msgcount for connection in connections)
        for index, (connection, timestamp_ns, rawdata) in enumerate(
                reader.messages(connections=connections), 1):
            msg = reader.deserialize(rawdata, connection.msgtype)
            generated = frame_counts.get(str(timestamp_ns), {})
            synthetic_points = int(generated.get("synthetic_points", 0))
            label = synthetic_points >= min_points
            distance_m = synthetic_distance_m(msg, synthetic_points) if label else None
            if cached is None:
                prediction = runtime.predict(msg)
                computed_predictions.append((timestamp_ns, prediction))
            else:
                try:
                    prediction = cached[timestamp_ns]
                except KeyError as exc:
                    raise ValueError(f"RailPrediction cache misses {timestamp_ns}") from exc
            xyz = valid_xyz(msg)
            raw_detection = raw.update(timestamp_ns, xyz, prediction)
            bayesian_detection = bayesian.update(timestamp_ns, xyz, prediction)
            frames.append({
                "timestamp_ns": timestamp_ns,
                "label": label,
                "synthetic_points": synthetic_points,
                "distance_m": distance_m,
                "distance_bin": distance_bin(distance_m) if label else None,
                "raw_point_count": raw_detection.point_count,
                "raw_distance_m": raw_detection.distance_m,
                "raw_detected": raw_detection.detected,
                "max_cell_point_count": int(np.max(bayesian_detection.occupancy.point_counts))
                    if len(bayesian_detection.occupancy.point_counts) else 0,
                "max_occupancy_probability": float(np.max(
                    bayesian_detection.occupancy.probabilities, initial=0.0)),
                "bayesian_detected": bayesian_detection.detected,
            })
            if index == 1 or index % 10 == 0 or index == total:
                print(f"{bag.name}: {index}/{total}", flush=True)

    if cached is not None and len(cached) != len(frames):
        raise ValueError(f"RailPrediction cache frame count does not match {bag}")
    if prediction_cache is not None and cached is None:
        path = save_prediction_cache(
            prediction_cache, bag, topic, weights, computed_predictions)
        print(f"{bag.name}: cached RailPredictions in {path}", flush=True)
    labels = [frame["label"] for frame in frames]
    raw_predictions = [frame["raw_detected"] for frame in frames]
    bayesian_predictions = [frame["bayesian_detected"] for frame in frames]
    bins = {}
    for name in ("0-30m", "30-80m", "80m+", "unknown"):
        selected = [frame for frame in frames
                    if frame["label"] and frame["distance_bin"] == name]
        if selected:
            bins[name] = {
                "positive_frames": len(selected),
                "raw_recall": sum(frame["raw_detected"] for frame in selected) / len(selected),
                "bayesian_recall": sum(frame["bayesian_detected"] for frame in selected) / len(selected),
            }
    summary = {
        "bag": str(bag),
        "topic": topic,
        "frames": len(frames),
        "positive_frames": sum(labels),
        "negative_frames": len(labels) - sum(labels),
        "raw": metrics(labels, raw_predictions),
        "bayesian_kiss_icp": metrics(labels, bayesian_predictions),
        "distance_bins": bins,
    }
    return summary, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--config", type=Path,
                        default=Path("railguard/config/railguard.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topic")
    parser.add_argument("--min-points", type=int)
    parser.add_argument("--prediction-cache", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/occupancy_precision_recall.json"))
    args = parser.parse_args()
    config = load_obstacle_detection_config(args.config)
    min_points = args.min_points or config.obstacle.min_cluster_points
    runtime = RailGuardRuntime(args.weights, args.device)
    summaries = []
    all_frames = []
    for bag in args.bags:
        summary, frames = evaluate_bag(
            bag, runtime, config, min_points, args.topic, args.weights,
            args.prediction_cache)
        summaries.append(summary)
        all_frames.extend({"bag": bag.name, **frame} for frame in frames)

    labels = [frame["label"] for frame in all_frames]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_definition": f"synthetic_points >= {min_points}",
        "config": str(args.config),
        "parameters": config.to_parameters(),
        "bayesian_odometry_enabled_for_evaluation": True,
        "bags": summaries,
        "combined": {
            "frames": len(all_frames),
            "positive_frames": sum(labels),
            "negative_frames": len(labels) - sum(labels),
            "raw": metrics(labels, [frame["raw_detected"] for frame in all_frames]),
            "bayesian_kiss_icp": metrics(
                labels, [frame["bayesian_detected"] for frame in all_frames]),
        },
        "frame_results": all_frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["combined"], indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
