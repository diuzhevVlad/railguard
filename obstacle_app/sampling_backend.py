"""Offline ray-casting simulation backend with stationary world obstacles."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import math
import time

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer, CompressionMode, CompressionFormat
from rosbags.typesys import Stores, get_typestore
from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP

from .bag_storage import estimate_export_bytes
from .cloud import append_xyz, filter_points, valid_xyz, xyz_view
from .core import Progress, Scenario
from .mesh import load_collada_mesh
from .merged_rails import MergedRailMap, merge_rails, world_rail_observation
from .rails import detect_rail_points
from .ray_rendering import MeshRayRenderer, RenderedScan, SensorRayModel
from .scan_mesh import ScanSurface
from .placement_warnings import placement_warnings
from .world_map import validate_map, world_mesh


def place_points(points: np.ndarray, position: np.ndarray,
                 direction: np.ndarray, ground_z: float) -> np.ndarray:
    fx, fy = direction
    # The source model faces +X. Rotate it to face the local rail tangent.
    result = np.empty_like(points)
    result[:, 0] = points[:, 0] * fx - points[:, 1] * fy + position[0]
    result[:, 1] = points[:, 0] * fy + points[:, 1] * fx + position[1]
    result[:, 2] = points[:, 2] - float(points[:, 2].min()) + ground_z
    return result


class RayCastingBackend:
    def run(self, scenario: Scenario, output_directory: Path, progress: Progress) -> Path:
        if scenario.start_seconds < 0 or scenario.duration_seconds <= 0:
            raise ValueError("Start must be nonnegative and duration positive")
        if scenario.every_n_frames < 1:
            raise ValueError("Frame interval must be positive")
        if scenario.rail_end_policy not in ("skip", "extrapolate"):
            raise ValueError("Unknown rail-end policy")
        if output_directory.exists():
            raise FileExistsError(output_directory)
        job_started = time.perf_counter()
        timings = {name: 0.0 for name in ("read_s", "decode_s", "render_s", "pack_s",
                   "serialize_s", "write_s", "writer_finalize_s", "odometry_s", "rail_detection_s")}
        progress(0.01, "Loading obstacle mesh")
        if scenario.world_obstacle is not None and scenario.world_obstacles:
            raise ValueError("Use either a single world obstacle or a list")
        world_items = scenario.world_obstacles or ((scenario.world_obstacle,) if scenario.world_obstacle is not None else ())
        if bool(world_items) and (scenario.obstacles or scenario.model_path is not None):
            raise ValueError("Choose one obstacle placement mode")
        if not scenario.obstacles and scenario.model_path is None and not world_items:
            raise ValueError("Add at least one obstacle")
        meshes = {}
        for item in scenario.obstacles:
            if item.model_path not in meshes:
                meshes[item.model_path] = load_collada_mesh(item.model_path)
        if not scenario.obstacles and not world_items:
            obstacle, faces = load_collada_mesh(scenario.model_path)
        source_metadata = scenario.source_bag / "obstacle_scenario.json"
        inherited = json.loads(source_metadata.read_text()) if source_metadata.is_file() else {}
        inherited_counts = inherited.get("frame_counts", {})
        inherited_modified = set(inherited.get("modified_timestamps_ns", []))
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        with AnyReader([scenario.source_bag], default_typestore=typestore) as reader:
            matches = [c for c in reader.connections if c.topic == scenario.topic]
            if len(matches) != 1 or matches[0].msgtype != "sensor_msgs/msg/PointCloud2":
                raise ValueError("Select one PointCloud2 topic in the source bag")
            start_ns = reader.start_time + round(scenario.start_seconds * 1e9)
            stop_ns = start_ns + round(scenario.duration_seconds * 1e9)
            if start_ns >= reader.end_time:
                raise ValueError("Clip starts after the end of the bag")
            clip_ns = max(1, min(stop_ns, reader.end_time) - start_ns)
            expected = estimate_export_bytes(reader, scenario.source_bag, start_ns, stop_ns, scenario.compression)
            if shutil.disk_usage(output_directory.parent).free < expected:
                raise OSError(f"Not enough free space for this clip (estimate {expected / 1e9:.1f} GB)")
            progress(0.03, "Loading saved map poses" if bool(world_items) else "Estimating poses and merging rails")
            pose_phase_started = time.perf_counter()
            if bool(world_items):
                if scenario.world_map_directory is None:
                    raise ValueError("World placement requires its saved map")
                validate_map(scenario.world_map_directory, scenario.source_bag, scenario.topic)
                with np.load(scenario.world_map_directory / "poses.npz", allow_pickle=False) as saved:
                    selected = (saved["timestamps_ns"] >= start_ns) & (saved["timestamps_ns"] < stop_ns)
                    poses = saved["poses"][selected]
                    frame_timestamps = saved["timestamps_ns"][selected].tolist()
                    merged = MergedRailMap(saved["center"], saved["lengths"])
                if not len(poses):
                    raise ValueError("No saved poses in this recording interval")
                first = next(reader.messages(connections=matches, start=start_ns, stop=stop_ns), None)
                if first is None or first[1] != frame_timestamps[0]:
                    raise ValueError("Saved poses do not match this bag")
                sensor = SensorRayModel.from_cloud(reader.deserialize(first[2], matches[0].msgtype))
                rail_map_method = "cached_editor_map"
                pose_phase_seconds = time.perf_counter() - pose_phase_started
            else:
                config = load_config(None)
                config.data.deskew = False
                odometry = KissICP(config)
                poses = []
                rail_observations = []
                frame_timestamps = []
                track_bases = []
                for _, bag_ns, raw in reader.messages(connections=matches,
                                                       start=start_ns, stop=stop_ns):
                    msg = reader.deserialize(raw, matches[0].msgtype)
                    if not poses:
                        sensor = SensorRayModel.from_cloud(msg)
                    xyz = valid_xyz(msg)
                    if len(poses) == 1:
                        odometry.last_delta[1, 3] = -scenario.initial_speed_m_s * 0.1
                    operation_started = time.perf_counter()
                    odometry.register_frame(xyz[::8].astype(np.float64),
                                            np.empty(0, dtype=np.float64))
                    timings["odometry_s"] += time.perf_counter() - operation_started
                    pose = odometry.last_pose.copy()
                    poses.append(pose)
                    frame_timestamps.append(bag_ns)
                    operation_started = time.perf_counter()
                    rails = detect_rail_points(xyz)
                    base = None
                    if len(rails) == 2:
                        local_sides = []
                        for side in ("left", "right"):
                            points = rails[side]
                            nearby = points[(-points[:, 1] >= 1) & (-points[:, 1] <= 10)]
                            if len(nearby) < 5:
                                break
                            # Estimate rail level directly under the LiDAR from nearby rails.
                            local_sides.append(np.array([np.polyfit(-nearby[:, 1], nearby[:, 0], 1)[1],
                                0, np.polyfit(-nearby[:, 1], nearby[:, 2], 1)[1]]))
                        if len(local_sides) == 2:
                            base = np.mean(local_sides, axis=0) @ pose[:3, :3].T + pose[:3, 3]
                    track_bases.append(base)
                    if len(rails) == 2:
                        rail_observations.append(world_rail_observation(rails, pose))
                    timings["rail_detection_s"] += time.perf_counter() - operation_started
                    if len(poses) % 10 == 0:
                        progress(0.03 + 0.43 * min(1, (bag_ns - start_ns) / clip_ns),
                                 f"Merged rails from {len(poses)} LiDAR frames")
                pose_phase_seconds = time.perf_counter() - pose_phase_started
                if not poses:
                    raise ValueError("No LiDAR frames in the selected bag")
                rail_map_method = "global_rail_merge"
                try:
                    merged = merge_rails(rail_observations)
                except ValueError:
                    if not scenario.obstacles and not world_items:
                        raise
                    # Timed placement needs rails around the encounter, not a
                    # globally continuous map through every turn in the recording.
                    center = []
                    for base in track_bases:
                        if base is not None and (not center or np.linalg.norm(base - center[-1]) >= .3):
                            center.append(base)
                    if not center:
                        raise ValueError("No nearby paired rails available for timed placement")
                    if len(center) == 1:
                        direction = poses[0][:3, :3] @ np.array([0., -1., 0.])
                        center.append(center[0] + direction)
                    center = np.asarray(center, dtype=np.float32)
                    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
                    merged = MergedRailMap(center, lengths)
                    rail_map_method = "pose_anchored_nearby_rails"
            anchors = []
            warnings = []
            if world_items:
                world_meshes = []
                for index, item in enumerate(world_items):
                    world_meshes.append(world_mesh(item))
                    anchors.append({"id": f"map-obstacle-{index+1}", "model": item.model_id,
                        "anchor_position_world_m": list(item.position), "yaw_deg": item.yaw_deg,
                        "scale": item.scale, "map_id": scenario.world_map_directory.name,
                        "anchor": "top" if item.model_id == "wire_hanging" else "bottom_center"})
                renderer = MeshRayRenderer.from_meshes(world_meshes, sensor)
                progress(.47, f"Placed {len(world_items)} map obstacles; writing bag with saved poses")
            elif scenario.obstacles:
                from .timed_placement import timed_anchor
                world_meshes = []
                for item in scenario.obstacles:
                    if not all(math.isfinite(value) for value in
                               (item.encounter_seconds, item.lateral_offset_m, item.height_above_track_m)):
                        raise ValueError("Obstacle placement must be finite")
                    if item.encounter_seconds < 0 or item.height_above_track_m < 0:
                        raise ValueError("Encounter time and height must be nonnegative")
                    position, direction, timestamp = timed_anchor(item, reader.start_time,
                        frame_timestamps, poses, track_bases, merged)
                    vertices, triangles = meshes[item.model_path]
                    placed = place_points(vertices, position, direction[:2], float(position[2]))
                    # Reject a starting placement that encloses or nearly touches
                    # the sensor. Otherwise near-zero first hits erase most of a scan.
                    origin = poses[0][:3, 3]
                    separation = np.maximum(np.maximum(placed.min(axis=0) - origin,
                                                       origin - placed.max(axis=0)), 0)
                    if np.linalg.norm(separation) < sensor.min_range_m:
                        movement = float(np.max(np.linalg.norm(
                            np.asarray(poses)[:, :3, 3] - origin, axis=1)))
                        reason = (f"The estimated train moves only {movement:.2f} m in this recording. "
                                  "Encounter times need a moving recording."
                                  if movement < .5 else
                                  "Choose a later encounter time.")
                        raise ValueError(f"Obstacle {item.id} at {item.encounter_seconds:g} s "
                                         f"overlaps the LiDAR's starting clearance. {reason}")
                    world_meshes.append((placed, triangles))
                    anchors.append({"id": item.id, "model": item.model_id,
                        "encounter_seconds": item.encounter_seconds,
                        "anchor_timestamp_ns": int(timestamp),
                        "anchor_position_world_m": position.tolist(),
                        "anchor_direction_world": direction.tolist(),
                        "lateral_offset_m": item.lateral_offset_m,
                        "height_above_track_m": item.height_above_track_m})
                warnings = placement_warnings(anchors, poses, frame_timestamps)
                renderer = MeshRayRenderer.from_meshes(world_meshes, sensor)
                progress(0.47, f"Placed {len(anchors)} stationary obstacles; writing bag")
            else:
                # Anchor once, relative to the first exported LiDAR frame. Every later
                # scan observes the same world-space geometry, even past the rail-map
                # end or after the train has passed the obstacle.
                anchor_pose = poses[0]
                try:
                    anchor = merged.locate(
                        anchor_pose, scenario.distance_m,
                        extrapolate=scenario.rail_end_policy == "extrapolate")
                except ValueError as exc:
                    raise ValueError(f"Cannot anchor obstacle at the start of the bag: {exc}. "
                                     "Reduce the initial distance or extend the last measured direction.") from exc
                z = scenario.ground_z_m
                if z is None:
                    z = float(anchor.position[2])
                initial_points = place_points(obstacle, anchor.position, anchor.direction,
                                              z + scenario.ground_offset_m)
                world_points = (initial_points.astype(np.float64) @ anchor_pose[:3, :3].T
                                + anchor_pose[:3, 3])
                renderer = MeshRayRenderer(world_points, faces, sensor)
                anchor_base = np.array([anchor.position[0], anchor.position[1],
                                        z + scenario.ground_offset_m])
                world_base = anchor_pose[:3, :3] @ anchor_base + anchor_pose[:3, 3]
                progress(0.47, f"Anchored stationary obstacle {scenario.distance_m:g} m ahead at bag start; writing bag")
            count = 0
            messages = 0
            modified = 0
            modified_timestamps = []
            frame_counts = {}
            export_started = time.perf_counter()
            writer = Writer(output_directory, version=8)
            if scenario.compression == "zstd":
                writer.set_compression(CompressionMode.MESSAGE, CompressionFormat.ZSTD)
            with writer:
                connections = {c.id: writer.add_connection(
                    c.topic, c.msgtype, typestore=typestore,
                    serialization_format=c.ext.serialization_format,
                    offered_qos_profiles=c.ext.offered_qos_profiles,
                ) for c in reader.connections}
                read_started = time.perf_counter()
                for conn, bag_ns, raw in reader.messages(start=start_ns, stop=stop_ns):
                    timings["read_s"] += time.perf_counter() - read_started
                    if conn.topic == scenario.topic:
                        count += 1
                        if bool(world_items) and (count > len(frame_timestamps) or bag_ns != frame_timestamps[count-1]):
                            raise ValueError("Source scan does not match saved map pose")
                        source_counts = inherited_counts.get(str(bag_ns), {})
                        source_tail = source_counts.get("synthetic_points", source_counts.get("added",
                            inherited.get("sampled_points", 0) if bag_ns in inherited_modified else 0))
                        if count % scenario.every_n_frames == 0 or source_tail:
                            operation_started = time.perf_counter()
                            msg = reader.deserialize(raw, conn.msgtype)
                            timings["decode_s"] += time.perf_counter() - operation_started
                            pose = poses[count - 1]
                            total = msg.width * msg.height
                            if not 0 <= source_tail <= total:
                                raise ValueError("Invalid inherited obstacle point count")
                            operation_started = time.perf_counter()
                            if count % scenario.every_n_frames == 0:
                                rendered = renderer.render(msg, pose, background_count=total-source_tail)
                            else:
                                rendered = RenderedScan(np.empty((0, 3), dtype=np.float32),
                                    np.empty(0, dtype=np.int32), np.ones(total, dtype=bool))
                            timings["render_s"] += time.perf_counter() - operation_started
                            operation_started = time.perf_counter()
                            keep = rendered.keep.copy()
                            removed_rear = 0
                            retained_synthetic = 0
                            if source_tail:
                                # Previously generated points remain a contiguous
                                # tail after filtering; carry their labels forward.
                                behind = xyz_view(msg)["y"].ravel()[-source_tail:] > 0
                                removed_rear = int((keep[-source_tail:] & behind).sum())
                                keep[-source_tail:] &= ~behind
                                retained_synthetic = int(keep[-source_tail:].sum())
                            frame_counts[str(bag_ns)] = {"added": len(rendered.points),
                                "removed": int((~keep).sum()),
                                "synthetic_points": retained_synthetic + len(rendered.points),
                                "removed_rear_obstacle_points": removed_rear,
                                "tunnel_hidden_points": rendered.tunnel_hidden_count,
                                "tunnel_mesh_triangles": rendered.tunnel_triangle_count,
                                "near_contact_obstacles": rendered.near_contact_obstacles}
                            if not keep.all() or len(rendered.points):
                                msg = filter_points(msg, keep)
                                # Flatten organized clouds before appending new records.
                                if msg.height != 1 or msg.row_step != msg.width * msg.point_step:
                                    records = np.ndarray((msg.height, msg.width, msg.point_step),
                                        dtype=np.uint8, buffer=msg.data,
                                        strides=(msg.row_step, msg.point_step, 1))
                                    msg.data = np.frombuffer(records.tobytes(), dtype=np.uint8)
                                    msg.width *= msg.height
                                    msg.height = 1
                                    msg.row_step = msg.width * msg.point_step
                                msg = append_xyz(msg, rendered.points, rendered.rings)
                                timings["pack_s"] += time.perf_counter() - operation_started
                                operation_started = time.perf_counter()
                                raw = typestore.serialize_cdr(msg, conn.msgtype)
                                timings["serialize_s"] += time.perf_counter() - operation_started
                                modified_timestamps.append(bag_ns)
                                modified += 1
                    operation_started = time.perf_counter()
                    writer.write(connections[conn.id], bag_ns, raw)
                    timings["write_s"] += time.perf_counter() - operation_started
                    messages += 1
                    if messages % 10 == 0:
                        if shutil.disk_usage(output_directory).free < 512*1024**2:
                            raise OSError("Less than 512 MiB free; export stopped before filling the disk")
                        progress(0.48 + 0.51 * min(1, (bag_ns - start_ns) / clip_ns),
                                 f"Wrote {messages} messages; modified {modified} frames")
                    read_started = time.perf_counter()
                finalize_started = time.perf_counter()
            timings["writer_finalize_s"] = time.perf_counter() - finalize_started
            export_seconds = time.perf_counter() - export_started
        if bool(world_items) and count != len(frame_timestamps):
            raise ValueError("Incomplete cached pose sequence during export")
        if not messages or not count:
            shutil.rmtree(output_directory)
            raise ValueError("No LiDAR frames in the selected clip")
        contact_frames = sum(item.get("near_contact_obstacles", 0) > 0 for item in frame_counts.values())
        if contact_frames:
            warnings.append(f"Obstacles were temporarily omitted in {contact_frames} scans because "
                "the LiDAR was inside or within 0.3 m of their surfaces. Recorded points were preserved "
                "from those objects' shadows; other obstacles remained active.")
        (output_directory / "obstacle_scenario.json").write_text(json.dumps({
            "backend": "ray_casting",
            "compression": scenario.compression,
            "warnings": warnings,
            "near_contact_policy": {"clearance_m": sensor.min_range_m,
                "reference": "object_mesh_signed_distance", "action": "omit_only_contact_objects"},
            "performance": {"stage_totals_seconds": timings,
                "pose_phase_seconds": pose_phase_seconds, "export_phase_seconds": export_seconds,
                "total_seconds": time.perf_counter() - job_started,
                "source_scan_count": count, "export_messages": messages,
                "export_messages_per_second": messages / max(export_seconds, 1e-6)},
            **({"model": scenario.model_id} if not scenario.obstacles and not world_items else {}),
            "sensor_model": sensor.metadata(),
            "tunnel_visibility": ScanSurface.metadata(),
            "placement": "stationary_world_anchor", "rail_detector": "RailGuard_rail_det",
            "time_reference": "seconds_from_source_bag_start",
            "obstacles": anchors,
            "odometry": "KISS-ICP", "initial_speed_m_s": scenario.initial_speed_m_s,
            "observed_rail_length_m": merged.length_m, "rail_map_method": rail_map_method,
            **({"anchor_timestamp_ns": frame_timestamps[0],
                "anchor_position_world_m": world_base.tolist(),
                "anchor_direction_world": (anchor_pose[:3, :3] @ np.r_[anchor.direction, 0]).tolist(),
                "distance_m": scenario.distance_m, "rail_end_policy": scenario.rail_end_policy,
                "extrapolated_m_max": anchor.extrapolated_m} if not scenario.obstacles and not world_items else {}),
            "skipped_frames": count - modified if scenario.every_n_frames == 1 else None,
            "every_n_frames": scenario.every_n_frames,
            "frame_counts": frame_counts,
            "total_added_points": sum(item["added"] for item in frame_counts.values()),
            "total_removed_points": sum(item["removed"] for item in frame_counts.values()),
            "modified_timestamps_ns": modified_timestamps,
            "source_bag": scenario.source_bag.name,
            "source_topic": scenario.topic,
        }, indent=2))
        np.savez_compressed(output_directory / "merged_rails.npz",
                            center=merged.center, lengths=merged.lengths,
                            poses=np.asarray(poses), timestamps_ns=np.asarray(frame_timestamps))
        progress(1.0, f"Done: {count} LiDAR frames, {modified} modified")
        return output_directory


# Compatibility for callers of the previous backend; jobs now use ray rendering.
MeshSamplingBackend = RayCastingBackend
