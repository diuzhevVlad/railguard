"""Visualize the obstacle app's merged LiDAR cloud and detected rails in Rerun.

Uses poses/centerline saved by MeshSamplingBackend, so the visualization and
obstacle placement refer to the same world map. No ROS process is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import rerun as rr
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Allow direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obstacle_app.cloud import valid_xyz  # noqa: E402
from obstacle_app.rails import detect_rail_points  # noqa: E402


def side_route(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Summarize world-space rail detections at the map's 0.5 m samples."""
    if not len(points):
        return np.empty((0, 3), dtype=np.float32)
    bins = np.floor(-points[:, 1] / 0.5).astype(np.int32)
    ids = np.unique(bins)
    samples = np.array([np.median(points[bins == index], axis=0) for index in ids])
    samples = samples[np.argsort(-samples[:, 1])]
    forward = -samples[:, 1]
    target = -center[:, 1]
    inside = (target >= forward[0]) & (target <= forward[-1])
    target = target[inside]
    if len(target) < 2:
        return np.empty((0, 3), dtype=np.float32)
    return np.column_stack((np.interp(target, forward, samples[:, 0]),
                            -target,
                            np.interp(target, forward, samples[:, 2]))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="Original ROS 2 bag directory")
    parser.add_argument("merged_map", type=Path, help="Exported merged_rails.npz")
    parser.add_argument("--topic", default="/lidar_points")
    parser.add_argument("--max-cloud-points", type=int, default=500_000,
                        help="Approximate total background cloud point cap")
    parser.add_argument("--max-rail-points-per-frame", type=int, default=300,
                        help="Cap for each side in one scan")
    parser.add_argument("--output", type=Path, help="Save .rrd; otherwise open Rerun")
    args = parser.parse_args()
    if args.max_cloud_points < 1 or args.max_rail_points_per_frame < 1:
        parser.error("Point caps must be positive")
    if not (args.bag / "metadata.yaml").is_file():
        parser.error(f"Not a ROS 2 bag: {args.bag}")

    with np.load(args.merged_map) as data:
        center = data["center"]
        lengths = data["lengths"]
        poses = data["poses"]
        timestamps = data["timestamps_ns"]
        initial_speed = float(data["initial_speed_m_s"]) if "initial_speed_m_s" in data else 0.0
    if len(poses) != len(timestamps) or len(center) != len(lengths) or len(center) < 2:
        raise ValueError("Invalid merged map: poses/timestamps/centerline mismatch")

    per_frame = max(1, args.max_cloud_points // len(poses))
    clouds: list[np.ndarray] = []
    rails: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == args.topic]
        if len(connections) != 1 or connections[0].msgtype != "sensor_msgs/msg/PointCloud2":
            raise ValueError(f"Expected one PointCloud2 connection on {args.topic}")
        frames = 0
        for conn, bag_ns, raw in reader.messages(connections=connections):
            if bag_ns < timestamps[0]:
                continue
            if bag_ns > timestamps[-1]:
                break
            if frames >= len(poses) or bag_ns != timestamps[frames]:
                raise ValueError("Map timestamps do not match this bag/topic; use its source bag")
            xyz = valid_xyz(reader.deserialize(raw, conn.msgtype))
            pose = poses[frames]
            rotation = pose[:3, :3]
            translation = pose[:3, 3]
            stride = max(1, int(np.ceil(len(xyz) / per_frame)))
            clouds.append((xyz[::stride] @ rotation.T + translation).astype(np.float32))
            detected = detect_rail_points(xyz)
            for side in ("left", "right"):
                points = detected.get(side)
                if points is not None and len(points):
                    stride = max(1, int(np.ceil(len(points) / args.max_rail_points_per_frame)))
                    rails[side].append((points[::stride] @ rotation.T + translation).astype(np.float32))
            frames += 1
            if frames % 50 == 0:
                print(f"Processed {frames}/{len(poses)} scans", flush=True)
    if frames != len(poses):
        raise ValueError(f"Map has {len(poses)} poses but bag has {frames} scans")

    rec = rr.RecordingStream(f"rail_guard_merged_{args.bag.name}_{initial_speed:g}mps")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rec.save(args.output)
    else:
        rec.spawn()
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    scene = np.concatenate(clouds)
    rec.log("world/merged_cloud", rr.Points3D(scene, colors=[125, 133, 145], radii=0.025), static=True)
    colors = {"left": [255, 86, 79], "right": [72, 166, 255]}
    for side in ("left", "right"):
        if rails[side]:
            points = np.concatenate(rails[side])
            rec.log(f"world/rails/{side}_points",
                    rr.Points3D(points, colors=colors[side], radii=0.06), static=True)
            route = side_route(points, center)
            if len(route) >= 2:
                rec.log(f"world/rails/{side}_route",
                        rr.LineStrips3D([route], colors=colors[side], radii=0.08), static=True)
    rec.log("world/rails/centerline",
            rr.LineStrips3D([center], colors=[255, 217, 51], radii=0.08), static=True)
    rec.log("world/lidar_trajectory",
            rr.LineStrips3D([poses[:, :3, 3]], colors=[95, 255, 147], radii=0.05), static=True)
    rec.flush()
    print(f"Visualized {frames} scans, {len(scene):,} cloud points, "
          f"{len(center)} centerline samples ({lengths[-1]:.1f} m), "
          f"initial speed {initial_speed:g} m/s.", flush=True)
    if args.output:
        print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
