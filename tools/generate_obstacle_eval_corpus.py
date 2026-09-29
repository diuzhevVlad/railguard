"""Generate a reproducible labeled obstacle bag from every real source bag."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from obstacle_app.core import ObstacleSpec, Scenario
from obstacle_app.detection import INITIAL_SPEED_MPS
from obstacle_app.sampling_backend import RayCastingBackend


TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
MODEL_SEQUENCE = (
    "person_standing",
    "box_square",
    "niva_car",
    "box_long_thin",
    "person_photo",
    "wire_hanging",
)


def bag_info(path: Path) -> tuple[str, float]:
    with AnyReader([path], default_typestore=TYPESTORE) as reader:
        topics = [connection.topic for connection in reader.connections
                  if connection.msgtype == "sensor_msgs/msg/PointCloud2"]
        if len(topics) != 1:
            raise ValueError(f"Expected one PointCloud2 topic in {path}, got {topics}")
        return topics[0], reader.duration / 1e9


def obstacle_times(duration_s: float) -> tuple[float, ...]:
    if duration_s >= 45:
        return duration_s * 0.32, duration_s * 0.72
    return (duration_s * 0.55,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/eval_corpus_all_real_bags"))
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--new-data-duration", type=float, default=20.0)
    args = parser.parse_args()
    sources = sorted(path for path in args.data_root.iterdir()
                     if path.is_dir() and (path / "metadata.yaml").is_file())
    if not sources:
        parser.error(f"No source bags found in {args.data_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"label": "synthetic_points >= configured min_cluster_points",
                "new_data_clip_seconds": [0.0, args.new_data_duration], "bags": []}
    backend = RayCastingBackend()
    model_index = 0
    for source in sources:
        topic, source_duration = bag_info(source)
        duration = min(source_duration, args.new_data_duration) if source.name == "new_data" else source_duration
        times = obstacle_times(duration)
        obstacles = []
        for item_index, relative_time in enumerate(times, 1):
            model_id = MODEL_SEQUENCE[model_index % len(MODEL_SEQUENCE)]
            model_index += 1
            filename = {
                "person_standing": "person_standing/meshes/person.dae",
                "person_photo": "person_photo/meshes/person.dae",
                "box_square": "box_square/meshes/box.dae",
                "box_long_thin": "box_long_thin/meshes/box.dae",
                "niva_car": "niva_car/meshes/niva.dae",
                "wire_hanging": "wire_hanging/meshes/wire.dae",
            }[model_id]
            obstacles.append(ObstacleSpec(
                id=f"{source.name}-{item_index}",
                model_path=args.models_root / filename,
                model_id=model_id,
                encounter_seconds=relative_time,
                lateral_offset_m=(-0.2, 0.0, 0.2)[model_index % 3],
                height_above_track_m=2.4 if model_id == "wire_hanging" else 0.0,
            ))
        output = args.output_root / f"eval_{source.name}"
        record = {
            "source": str(source), "output": str(output), "topic": topic,
            "start_seconds": 0.0, "duration_seconds": duration,
            "source_duration_seconds": source_duration,
            "initial_speed_m_s": INITIAL_SPEED_MPS.get(source.name, 0.0),
            "obstacles": [{**asdict(item), "model_path": str(item.model_path)}
                          for item in obstacles],
        }
        if source.name == "doubleT_obstacle":
            record["fixed_distance_m"] = 25.0
        if (output / "obstacle_scenario.json").is_file():
            record["status"] = "existing"
            print(f"{source.name}: keeping existing {output}", flush=True)
        else:
            print(f"{source.name}: generating {duration:.3f}s -> {output}", flush=True)
            try:
                if "fixed_distance_m" in record:
                    item = obstacles[0]
                    scenario = Scenario(
                        source_bag=source, topic=topic, start_seconds=0.0,
                        duration_seconds=duration, model_path=item.model_path,
                        model_id=item.model_id, distance_m=record["fixed_distance_m"],
                        initial_speed_m_s=record["initial_speed_m_s"],
                    )
                else:
                    scenario = Scenario(
                        source_bag=source, topic=topic, start_seconds=0.0,
                        duration_seconds=duration, obstacles=tuple(obstacles),
                        initial_speed_m_s=record["initial_speed_m_s"],
                    )
                last_message = None
                def progress(value: float, message: str) -> None:
                    nonlocal last_message
                    if message != last_message:
                        print(f"  {value:5.1%} {message}", flush=True)
                        last_message = message
                backend.run(scenario, output, progress)
                record["status"] = "generated"
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  FAILED: {record['error']}", flush=True)
        manifest["bags"].append(record)
        (args.output_root / "corpus_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    failures = [record for record in manifest["bags"] if record["status"] == "failed"]
    if failures:
        raise SystemExit(f"{len(failures)} bag(s) failed; see corpus_manifest.json")


if __name__ == "__main__":
    main()
