"""Tune the causal centerline filter on read-only ranges of a real bag.

The tool writes metrics only.  It never creates or modifies a ROS bag and uses
the last clip as a held-out validation range.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP
from obstacle_app.cloud import valid_xyz
from obstacle_app.detection import (
    INITIAL_SPEED_MPS,
    MergedRailPredictor,
    _last_scan_tail,
    _rail_anchor,
    merged_trail_from_anchors,
)
from obstacle_app.learned_detection import RailGuardRuntime
from obstacle_app.merged_rails import merge_rails, world_rail_observation
from obstacle_app.rails import detect_rail_points
from railpath.centerline_filter import BayesianCenterlineFilter
from railpath.detection_config import load_obstacle_detection_config
from tools.evaluate_centerline_filter import centerline_errors, temporal_jitter


TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


@dataclass(frozen=True)
class ClipSpec:
    start_s: float
    duration_s: float

    @classmethod
    def parse(cls, text: str) -> "ClipSpec":
        try:
            start, duration = (float(value) for value in text.split(":", 1))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "clip must be START_SECONDS:DURATION_SECONDS") from exc
        if start < 0 or duration <= 0:
            raise argparse.ArgumentTypeError(
                "clip start must be nonnegative and duration positive")
        return cls(start, duration)

    @property
    def name(self) -> str:
        return f"{self.start_s:g}:{self.duration_s:g}"


@dataclass
class ClipData:
    spec: ClipSpec
    timestamps_ns: np.ndarray
    poses: np.ndarray
    predictions: list
    ground_truth: list
    map_method: str
    map_length_m: float
    paired_rail_frames: int


def build_clip(
    bag: Path,
    topic: str,
    spec: ClipSpec,
    runtime: RailGuardRuntime,
    initial_speed_m_s: float,
    max_forward_m: float,
) -> ClipData:
    """Read one timestamp range and derive poses, labels, and predictions."""
    odometry_config = load_config(None)
    odometry_config.data.deskew = False
    odometry = KissICP(odometry_config)
    poses: list[np.ndarray] = []
    timestamps: list[int] = []
    predictions = []
    observations = []
    anchors = []
    last_rails = None
    last_rail_pose = None

    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        matches = [
            connection for connection in reader.connections
            if connection.topic == topic
            and connection.msgtype == "sensor_msgs/msg/PointCloud2"
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one PointCloud2 topic {topic!r}")
        connection = matches[0]
        start_ns = reader.start_time + round(spec.start_s * 1e9)
        stop_ns = start_ns + round(spec.duration_s * 1e9)
        for _, timestamp_ns, rawdata in reader.messages(
                connections=matches, start=start_ns, stop=stop_ns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            xyz = valid_xyz(msg)
            if len(poses) == 1:
                odometry.last_delta[1, 3] = -initial_speed_m_s * 0.1
            odometry.register_frame(
                xyz[::8].astype(np.float64), np.empty(0, dtype=np.float64))
            pose = odometry.last_pose.copy()
            poses.append(pose)
            timestamps.append(timestamp_ns)
            predictions.append(runtime.predict(msg))

            rails = detect_rail_points(xyz)
            if len(rails) == 2:
                observations.append(world_rail_observation(rails, pose))
                anchor = _rail_anchor(rails, pose)
                if anchor is not None:
                    anchors.append(anchor)
                    last_rails = rails
                    last_rail_pose = pose
            if len(poses) == 1 or len(poses) % 25 == 0:
                print(f"clip {spec.name}: prepared {len(poses)} frames", flush=True)

    if len(poses) < 10:
        raise ValueError(f"Clip {spec.name} has too few LiDAR frames")
    pose_array = np.asarray(poses)
    try:
        merged = merge_rails(observations)
    except ValueError:
        merged = None
    method = "global_rail_merge"
    if merged is None:
        tail = (
            _last_scan_tail(last_rails, last_rail_pose)
            if last_rails is not None else np.empty((0, 3), dtype=np.float64)
        )
        merged = merged_trail_from_anchors(anchors, tail)
        method = "pose_anchored_rail_trail"

    predictor = MergedRailPredictor(merged)
    ground_truth = [
        predictor.predict(pose, max_forward_m) for pose in pose_array
    ]
    return ClipData(
        spec=spec,
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        poses=pose_array,
        predictions=predictions,
        ground_truth=ground_truth,
        map_method=method,
        map_length_m=merged.length_m,
        paired_rail_frames=len(observations),
    )


def metric(values: list[np.ndarray]) -> dict:
    array = np.concatenate(values) if values else np.empty(0)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "samples": 0, "mae_m": None, "rmse_m": None, "p95_m": None,
            "within_0_25m": None, "within_0_50m": None,
        }
    return {
        "samples": int(len(array)),
        "mae_m": float(np.mean(array)),
        "rmse_m": float(np.sqrt(np.mean(array ** 2))),
        "p95_m": float(np.percentile(array, 95)),
        "within_0_25m": float(np.mean(array <= 0.25)),
        "within_0_50m": float(np.mean(array <= 0.50)),
    }


def evaluate(clips: list[ClipData], centerline_config=None) -> dict:
    errors_3d: list[np.ndarray] = []
    errors_xy: list[np.ndarray] = []
    jitters: list[np.ndarray] = []
    per_clip = []

    for clip in clips:
        tracker = (
            BayesianCenterlineFilter(centerline_config)
            if centerline_config is not None else None
        )
        previous = None
        clip_3d: list[np.ndarray] = []
        clip_xy: list[np.ndarray] = []
        clip_jitter: list[np.ndarray] = []
        for index, (prediction, ground_truth) in enumerate(zip(
                clip.predictions, clip.ground_truth)):
            current = (
                tracker.update(
                    prediction, clip.poses[index], odometry_used=True).prediction
                if tracker is not None else prediction
            )
            error_3d, error_xy = centerline_errors(current, ground_truth)
            clip_3d.append(error_3d)
            clip_xy.append(error_xy)
            if previous is not None:
                clip_jitter.append(temporal_jitter(
                    current, previous, clip.poses[index], clip.poses[index - 1]))
            previous = current
        errors_3d.extend(clip_3d)
        errors_xy.extend(clip_xy)
        jitters.extend(clip_jitter)
        per_clip.append({
            "clip": clip.spec.name,
            "error_3d": metric(clip_3d),
            "error_xy": metric(clip_xy),
            "temporal_jitter": metric(clip_jitter),
        })

    return {
        "error_3d": metric(errors_3d),
        "error_xy": metric(errors_xy),
        "temporal_jitter": metric(jitters),
        "clips": per_clip,
    }


def score(result: dict) -> float:
    """Balance typical error, tail error, and motion-compensated jitter."""
    xy = result["error_xy"]
    jitter = result["temporal_jitter"]
    return (
        xy["mae_m"] + 0.25 * xy["p95_m"]
        + 0.20 * jitter["rmse_m"]
    )


def validation_not_worse(candidate: dict, baseline: dict) -> bool:
    """Require held-out geometry, tolerance precision, and jitter to hold."""
    current = candidate["validation"]
    reference = baseline["validation"]
    if candidate["validation_score"] > baseline["validation_score"]:
        return False
    for group in ("error_xy", "error_3d"):
        for name in ("mae_m", "rmse_m", "p95_m"):
            if current[group][name] > reference[group][name]:
                return False
        for name in ("within_0_25m", "within_0_50m"):
            if current[group][name] < reference[group][name]:
                return False
    for name in ("rmse_m", "p95_m"):
        if (current["temporal_jitter"][name]
                > reference["temporal_jitter"][name]):
            return False
    return True


def candidate_configs(base):
    """Yield a compact gain/association grid, including the configured default."""
    seen = set()
    values = [base]
    for process, measurement, fade, uncertainty in product(
            (0.005, 0.01, 0.02, 0.04, 0.08),
            (0.04, 0.09, 0.18),
            (0.25, 0.50),
            (False, True)):
        values.append(replace(
            base,
            process_variance_m2=process,
            measurement_variance_m2=measurement,
            association_fade_start_ratio=fade,
            use_model_uncertainty=uncertainty,
        ))
    for config in values:
        key = tuple(asdict(config).values())
        if key not in seen:
            seen.add(key)
            yield config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument(
        "--clip", action="append", type=ClipSpec.parse, required=True,
        help="read-only START_SECONDS:DURATION_SECONDS range; repeat at least twice")
    parser.add_argument("--topic", default="/lidar_points")
    parser.add_argument(
        "--config", type=Path, default=Path("railguard/config/railguard.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--initial-speed-m-s", type=float)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/new_data_centerline_tuning.json"))
    args = parser.parse_args()
    if len(args.clip) < 2:
        parser.error("provide at least two clips so the last can be held out")
    if not args.bag.is_dir():
        parser.error(f"bag not found: {args.bag}")

    detection_config = load_obstacle_detection_config(args.config)
    speed = (
        args.initial_speed_m_s
        if args.initial_speed_m_s is not None
        else INITIAL_SPEED_MPS.get(args.bag.name, 0.0)
    )
    runtime = RailGuardRuntime(args.weights, args.device)
    clips = [
        build_clip(
            args.bag, args.topic, spec, runtime, speed,
            detection_config.max_forward_m)
        for spec in args.clip
    ]
    training = clips[:-1]
    validation = clips[-1:]
    raw = {
        "training": evaluate(training),
        "validation": evaluate(validation),
        "all": evaluate(clips),
    }

    candidates = []
    for index, config in enumerate(candidate_configs(detection_config.centerline), 1):
        print(f"evaluating centerline candidate {index}", flush=True)
        results = {
            "training": evaluate(training, config),
            "validation": evaluate(validation, config),
            "all": evaluate(clips, config),
        }
        candidates.append({
            "config": asdict(config),
            "training_score": score(results["training"]),
            "validation_score": score(results["validation"]),
            "all_score": score(results["all"]),
            **results,
        })

    configured = next(
        item for item in candidates
        if item["config"] == asdict(detection_config.centerline)
    )
    eligible = [item for item in candidates
                if validation_not_worse(item, configured)]
    selected = min(
        eligible or [configured],
        key=lambda item: (item["training_score"], item["validation_score"]))
    candidates.sort(key=lambda item: item["training_score"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_bag": str(args.bag),
        "source_bag_modified": False,
        "bag_data_copied": False,
        "topic": args.topic,
        "selection": (
            "Lowest training score among candidates that do not regress held-out "
            "XY/3D MAE, RMSE, p95, 0.25/0.50 m tolerance precision, or "
            "temporal-jitter RMSE/p95 versus configured defaults."
        ),
        "score": "XY_MAE + 0.25*XY_p95 + 0.20*jitter_RMSE",
        "training_clips": [clip.spec.name for clip in training],
        "validation_clips": [clip.spec.name for clip in validation],
        "clip_metadata": [{
            "clip": clip.spec.name,
            "frames": len(clip.timestamps_ns),
            "paired_rail_frames": clip.paired_rail_frames,
            "map_method": clip.map_method,
            "map_length_m": clip.map_length_m,
        } for clip in clips],
        "raw": raw,
        "configured": configured,
        "selected": selected,
        "candidate_count": len(candidates),
        "candidates_by_training_score": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "raw": raw["all"],
        "configured": configured["all"],
        "selected_config": selected["config"],
        "selected": selected["all"],
    }, indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
