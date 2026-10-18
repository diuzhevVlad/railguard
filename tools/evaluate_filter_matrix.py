"""Evaluate occupancy and rolling-NURBS centerline filtering on generated bags."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from obstacle_app.cloud import valid_xyz
from obstacle_app.detection import MergedRailPredictor
from obstacle_app.learned_detection import RailGuardRuntime
from obstacle_app.merged_rails import MergedRailMap
from obstacle_app.prediction_cache import (load_prediction_cache,
                                           save_prediction_cache)
from railpath.detection_config import load_obstacle_detection_config
from railpath.centerline_filter import BayesianCenterlineFilter
from railpath.nurbs_centerline_filter import RollingNURBSCenterlineFilter
from railpath.obstacle_filter import detect_obstacle_points
from railpath.occupancy_filter import BayesianObstacleFilter
from tools.evaluate_centerline_filter import (
    centerline_errors, classification_metrics, error_metrics, temporal_jitter,
)
from tools.evaluate_obstacle_bags import distance_bin, synthetic_distance_m


TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
CONFIGURATIONS = {
    "no_filters": (False, False),
    "occupancy_only": (True, False),
    "centerline_only": (False, True),
    "occupancy_and_centerline": (True, True),
}
LEGACY_CONFIGURATIONS = {
    "legacy_centerline_only": (False, True),
    "legacy_occupancy_and_centerline": (True, True),
}


def occupancy_detected(spatial, occupancy, config) -> bool:
    """Apply the pipeline's occupancy confirmation policy."""
    maximum = float(np.max(occupancy.probabilities, initial=0.0))
    immediate = bool(spatial.detected and spatial.distance_m is not None and (
        spatial.distance_m <= config.immediate_close_distance_m
        or (spatial.distance_m <= config.immediate_near_distance_m
            and maximum >= config.immediate_near_probability)
        or spatial.distance_m >= config.immediate_far_distance_m
    ))
    return bool(spatial.detected and (occupancy.detected or immediate))

def detection_result(spatial, detected: bool, occupancy=None) -> dict:
    """Serialize one configuration's frame-level result."""
    return {
        "detected": bool(detected),
        "point_count": spatial.point_count if detected else 0,
        "distance_m": spatial.distance_m if detected else None,
        "max_occupancy_probability": float(np.max(
            occupancy.probabilities, initial=0.0)) if occupancy is not None else None,
    }



def distance_bin_metrics(frames: list[dict], names=CONFIGURATIONS) -> dict:
    result = {}
    for bin_name in ("0-30m", "30-80m", "80m+", "unknown"):
        selected = [
            frame for frame in frames
            if frame["label"] and frame["distance_bin"] == bin_name
        ]
        if selected:
            result[bin_name] = {
                "positive_frames": len(selected),
                "recall": {
                    name: sum(frame["configurations"][name]["detected"]
                              for frame in selected) / len(selected)
                    for name in names
                },
            }
    return result


def evaluate_bag(
    bag: Path,
    runtime: RailGuardRuntime,
    config,
    min_points: int,
    topic_override: str | None,
    weights: Path,
    prediction_cache: Path | None,
    compare_legacy: bool = False,
) -> tuple[dict, list[dict], dict[str, list[np.ndarray]]]:
    scenario = json.loads(
        (bag / "obstacle_scenario.json").read_text(encoding="utf-8"))
    frame_counts = scenario["frame_counts"]
    topic = topic_override or scenario.get("source_topic", "/lidar_points")
    with np.load(bag / "merged_rails.npz", allow_pickle=False) as saved:
        poses = saved["poses"].copy()
        timestamps = saved["timestamps_ns"].copy()
        gt_predictor = MergedRailPredictor(
            MergedRailMap(saved["center"].copy(), saved["lengths"].copy()))
    if poses.shape != (len(timestamps), 4, 4):
        raise ValueError(f"Invalid labeled pose sequence in {bag}")

    filter_class = (RollingNURBSCenterlineFilter
                    if config.centerline.algorithm == "rolling_nurbs"
                    else BayesianCenterlineFilter)
    centerline_filter = filter_class(config.centerline)
    legacy_filter = BayesianCenterlineFilter(config.centerline) if compare_legacy else None
    raw_occupancy_filter = BayesianObstacleFilter(config.bayesian)
    filtered_occupancy_filter = BayesianObstacleFilter(config.bayesian)
    legacy_occupancy_filter = (BayesianObstacleFilter(config.bayesian)
                               if compare_legacy else None)
    names = {**CONFIGURATIONS, **LEGACY_CONFIGURATIONS} if compare_legacy else CONFIGURATIONS
    last_timestamp_ns = None
    cached = (
        load_prediction_cache(prediction_cache, bag, topic, weights)
        if prediction_cache is not None else None
    )
    if cached is not None:
        print(f"{bag.name}: loaded {len(cached)} cached RailPredictions",
              flush=True)
    computed_predictions = []
    frames = []
    errors = {
        name: [] for name in
        ("raw_3d", "filtered_3d", "raw_xy", "filtered_xy",
         "raw_jitter", "filtered_jitter")
    }
    if compare_legacy:
        errors.update(legacy_3d=[], legacy_xy=[])
    previous_raw = previous_filtered = None

    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        connections = [
            connection for connection in reader.connections
            if connection.topic == topic
            and connection.msgtype == "sensor_msgs/msg/PointCloud2"
        ]
        if len(connections) != 1:
            raise ValueError(
                f"Expected one PointCloud2 topic {topic!r} in {bag}")
        total = connections[0].msgcount
        for index, (connection, timestamp_ns, rawdata) in enumerate(
                reader.messages(connections=connections)):
            if index >= len(timestamps) or timestamp_ns != timestamps[index]:
                raise ValueError(
                    f"Labeled centerline timestamps do not match {bag}")
            msg = reader.deserialize(rawdata, connection.msgtype)
            generated = frame_counts.get(str(timestamp_ns), {})
            synthetic_points = int(generated.get("synthetic_points", 0))
            label = synthetic_points >= min_points
            object_distance = (
                synthetic_distance_m(msg, synthetic_points) if label else None)
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
            if last_timestamp_ns is not None:
                elapsed_s = (timestamp_ns - last_timestamp_ns) / 1e9
                if elapsed_s <= 0 or elapsed_s > config.bayesian.max_time_gap_s:
                    centerline_filter.reset()
                    raw_occupancy_filter.reset()
                    filtered_occupancy_filter.reset()
                    if compare_legacy:
                        legacy_filter.reset()
                        legacy_occupancy_filter.reset()
            last_timestamp_ns = timestamp_ns
            pose = poses[index]
            filtered_estimate = centerline_filter.update(
                prediction, pose, odometry_used=True,
                timestamp_ns=timestamp_ns)
            filtered_prediction = filtered_estimate.prediction
            legacy_prediction = (legacy_filter.update(
                prediction, pose, odometry_used=True).prediction
                if compare_legacy else None)
            raw_detection = detect_obstacle_points(
                xyz, prediction, config.obstacle)
            filtered_detection = detect_obstacle_points(
                xyz, filtered_prediction, config.obstacle)
            legacy_detection = (detect_obstacle_points(
                xyz, legacy_prediction, config.obstacle)
                if compare_legacy else None)
            raw_occupancy = raw_occupancy_filter.update(
                prediction, raw_detection.points, pose,
                min_distance_m=config.obstacle.min_forward_m,
                max_distance_m=prediction.max_forward_m,
                odometry_used=True,
            )
            filtered_occupancy = filtered_occupancy_filter.update(
                filtered_prediction, filtered_detection.points, pose,
                min_distance_m=config.obstacle.min_forward_m,
                max_distance_m=filtered_prediction.max_forward_m,
                odometry_used=True,
            )
            legacy_occupancy = (legacy_occupancy_filter.update(
                legacy_prediction, legacy_detection.points, pose,
                min_distance_m=config.obstacle.min_forward_m,
                max_distance_m=legacy_prediction.max_forward_m,
                odometry_used=True) if compare_legacy else None)
            raw_confirmed = occupancy_detected(
                raw_detection, raw_occupancy, config.bayesian)
            filtered_confirmed = occupancy_detected(
                filtered_detection, filtered_occupancy, config.bayesian)
            legacy_confirmed = (occupancy_detected(
                legacy_detection, legacy_occupancy, config.bayesian)
                if compare_legacy else None)

            ground_truth = gt_predictor.predict(
                poses[index], config.max_forward_m)
            raw_error, raw_lateral = centerline_errors(
                prediction, ground_truth)
            filtered_error, filtered_lateral = centerline_errors(
                filtered_prediction, ground_truth)
            errors["raw_3d"].append(raw_error)
            errors["filtered_3d"].append(filtered_error)
            errors["raw_xy"].append(raw_lateral)
            errors["filtered_xy"].append(filtered_lateral)
            if compare_legacy:
                legacy_error, legacy_lateral = centerline_errors(
                    legacy_prediction, ground_truth)
                errors["legacy_3d"].append(legacy_error)
                errors["legacy_xy"].append(legacy_lateral)
            if index:
                raw_frame_jitter = temporal_jitter(
                    prediction, previous_raw, poses[index], poses[index - 1])
                filtered_frame_jitter = temporal_jitter(
                    filtered_prediction, previous_filtered,
                    poses[index], poses[index - 1])
            else:
                raw_frame_jitter = filtered_frame_jitter = np.empty(0)
            errors["raw_jitter"].append(raw_frame_jitter)
            errors["filtered_jitter"].append(filtered_frame_jitter)
            previous_raw = prediction
            previous_filtered = filtered_prediction

            configuration_results = {
                "no_filters": detection_result(raw_detection, raw_detection.detected),
                "occupancy_only": detection_result(
                    raw_detection, raw_confirmed, raw_occupancy),
                "centerline_only": detection_result(
                    filtered_detection, filtered_detection.detected),
                "occupancy_and_centerline": detection_result(
                    filtered_detection, filtered_confirmed, filtered_occupancy),
            }
            if compare_legacy:
                configuration_results.update({
                    "legacy_centerline_only": detection_result(
                        legacy_detection, legacy_detection.detected),
                    "legacy_occupancy_and_centerline": detection_result(
                        legacy_detection, legacy_confirmed, legacy_occupancy),
                })
            frames.append({
                "timestamp_ns": timestamp_ns,
                "label": label,
                "synthetic_points": synthetic_points,
                "distance_m": object_distance,
                "distance_bin": (
                    distance_bin(object_distance) if label else None),
                "configurations": configuration_results,
                "raw_centerline_rmse_m": (
                    float(np.sqrt(np.mean(raw_error ** 2)))
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
        raise ValueError(
            f"RailPrediction cache frame count does not match {bag}")
    if prediction_cache is not None and cached is None:
        cache_path = save_prediction_cache(
            prediction_cache, bag, topic, weights, computed_predictions)
        print(f"{bag.name}: cached RailPredictions in {cache_path}",
              flush=True)

    labels = [frame["label"] for frame in frames]
    summary = {
        "bag": str(bag),
        "topic": topic,
        "frames": len(frames),
        "positive_frames": sum(labels),
        "negative_frames": len(labels) - sum(labels),
        "configurations": {
            name: classification_metrics(
                labels,
                [frame["configurations"][name]["detected"]
                 for frame in frames],
            )
            for name in names
        },
        "distance_bins": distance_bin_metrics(frames, names),
        "centerline_temporal_jitter": {
            "raw_3d": error_metrics(errors["raw_jitter"]),
            "filtered_3d": error_metrics(errors["filtered_jitter"]),
        },
        "centerline_gt": {
            "raw_3d": error_metrics(errors["raw_3d"]),
            "filtered_3d": error_metrics(errors["filtered_3d"]),
            "raw_xy": error_metrics(errors["raw_xy"]),
            "filtered_xy": error_metrics(errors["filtered_xy"]),
            **({"legacy_3d": error_metrics(errors["legacy_3d"]),
                "legacy_xy": error_metrics(errors["legacy_xy"])}
               if compare_legacy else {}),
        },
    }
    return summary, frames, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=Path("railguard/config/railguard.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topic")
    parser.add_argument("--min-points", type=int)
    parser.add_argument("--prediction-cache", type=Path)
    parser.add_argument("--compare-legacy", action="store_true",
                        help="Also run the previous pointwise Kalman filter on the same scans")
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/filter_matrix_evaluation.json"))
    args = parser.parse_args()

    config = load_obstacle_detection_config(args.config)
    if args.compare_legacy and config.centerline.algorithm != "rolling_nurbs":
        parser.error("--compare-legacy requires centerline_filter_algorithm=rolling_nurbs")
    names = {**CONFIGURATIONS, **LEGACY_CONFIGURATIONS} if args.compare_legacy else CONFIGURATIONS
    min_points = args.min_points or config.obstacle.min_cluster_points
    runtime = RailGuardRuntime(args.weights, args.device)
    summaries = []
    all_frames = []
    combined_errors = {
        name: [] for name in
        ("raw_3d", "filtered_3d", "raw_xy", "filtered_xy",
         "raw_jitter", "filtered_jitter")
    }
    if args.compare_legacy:
        combined_errors.update(legacy_3d=[], legacy_xy=[])
    for bag in args.bags:
        summary, frames, errors = evaluate_bag(
            bag, runtime, config, min_points, args.topic, args.weights,
            args.prediction_cache, args.compare_legacy)
        summaries.append(summary)
        all_frames.extend(
            {"bag": bag.name, **frame} for frame in frames)
        for name, values in errors.items():
            combined_errors[name].extend(values)

    labels = [frame["label"] for frame in all_frames]
    configuration_metrics = {
        name: classification_metrics(
            labels,
            [frame["configurations"][name]["detected"]
             for frame in all_frames],
        )
        for name in names
    }
    baseline = configuration_metrics["no_filters"]
    raw_gt = error_metrics(combined_errors["raw_3d"])
    filtered_gt = error_metrics(combined_errors["filtered_3d"])
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_definition": f"synthetic_points >= {min_points}",
        "configuration_definitions": {
            name: {
                "occupancy_filter_enabled": occupancy,
                "centerline_filter_enabled": centerline,
                "centerline_filter_algorithm": (
                    "legacy_pointwise" if name.startswith("legacy_") else
                    config.centerline.algorithm if centerline else None),
                "replayed_kiss_icp_enabled": occupancy or centerline,
            }
            for name, (occupancy, centerline) in names.items()
        },
        "comparison_control": (
            "All configurations use identical scans and cached RailPredictions. "
            "Stored KISS-ICP poses are replayed whenever either temporal filter "
            "is enabled."),
        "parameters": config.to_parameters(),
        "bags": summaries,
        "combined": {
            "bags": len(summaries),
            "frames": len(all_frames),
            "positive_frames": sum(labels),
            "negative_frames": len(labels) - sum(labels),
            "configurations": configuration_metrics,
            "deltas_from_no_filters": {
                name: {
                    "precision": values["precision"] - baseline["precision"],
                    "recall": values["recall"] - baseline["recall"],
                    "false_positive": (
                        values["false_positive"]
                        - baseline["false_positive"]),
                    "false_negative": (
                        values["false_negative"]
                        - baseline["false_negative"]),
                }
                for name, values in configuration_metrics.items()
                if name != "no_filters"
            },
            "centerline_temporal_jitter": {
                "raw_3d": error_metrics(combined_errors["raw_jitter"]),
                "filtered_3d": error_metrics(
                    combined_errors["filtered_jitter"]),
            },
            "centerline_gt": {
                "raw_3d": raw_gt,
                "filtered_3d": filtered_gt,
                "raw_xy": error_metrics(combined_errors["raw_xy"]),
                "filtered_xy": error_metrics(combined_errors["filtered_xy"]),
                **({"legacy_3d": error_metrics(combined_errors["legacy_3d"]),
                    "legacy_xy": error_metrics(combined_errors["legacy_xy"])}
                   if args.compare_legacy else {}),
                "rmse_delta_m": (
                    filtered_gt["rmse_m"] - raw_gt["rmse_m"]
                    if raw_gt["rmse_m"] is not None else None),
            },
        },
        "frame_results": all_frames,
    }
    if args.compare_legacy:
        gt = result["combined"]["centerline_gt"]
        result["combined"]["nurbs_vs_legacy"] = {
            "centerline_only": {
                metric: configuration_metrics["centerline_only"][metric]
                - configuration_metrics["legacy_centerline_only"][metric]
                for metric in ("precision", "recall")
            },
            "occupancy_and_centerline": {
                metric: configuration_metrics["occupancy_and_centerline"][metric]
                - configuration_metrics["legacy_occupancy_and_centerline"][metric]
                for metric in ("precision", "recall")
            },
            "centerline_rmse_delta_m": (
                gt["filtered_3d"]["rmse_m"] - gt["legacy_3d"]["rmse_m"]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["combined"], indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
