"""Small bag and fake-backend checks for the obstacle app."""

from __future__ import annotations

from pathlib import Path
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from obstacle_app.cloud import append_xyz, valid_xyz
from obstacle_app.core import ObstacleSpec, Scenario
from obstacle_app.detection import (MergedRailPredictor, PrecomputedMergedRailProvider, RailPrediction,
                                    detect_obstacle_points, merged_trail_from_anchors)
from obstacle_app.mesh import load_collada_mesh, sample_collada_surface
from obstacle_app.merged_rails import MergedRailMap, merge_rails, world_rail_observation
from obstacle_app.rails import detect_rail_points
from obstacle_app.sampling_backend import MeshSamplingBackend, place_points
from obstacle_app.web import create_app


TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
def fake_scan_detector(bag_id, topic):
    from obstacle_app.scan_detection import ScanDetector
    class Provider:
        def predict(self, timestamp_ns, point_cloud):
            assert point_cloud is not None
            center=np.column_stack((np.zeros(81),-np.arange(81)*.5,np.full(81,-1.1))).astype(np.float32)
            return RailPrediction(center,np.arange(81)*.5)
    return ScanDetector(Provider())


MODEL = Path(__file__).resolve().parent.parent / "models/person_standing/meshes/person.dae"
MODELS_ROOT = MODEL.parent.parent.parent


def make_bag(path: Path, rails: bool = False, obstacle: bool = False,
             obstacle_y: float = -10, obstacle_z: float = -0.5,
             frame_interval_ns: int = 100_000_000) -> list[bytes]:
    Header = TYPESTORE.types["std_msgs/msg/Header"]
    Time = TYPESTORE.types["builtin_interfaces/msg/Time"]
    Field = TYPESTORE.types["sensor_msgs/msg/PointField"]
    Cloud = TYPESTORE.types["sensor_msgs/msg/PointCloud2"]
    fields = [Field(name=name, offset=offset, datatype=datatype, count=1)
              for name, offset, datatype in (("x", 0, 7), ("y", 4, 7), ("z", 8, 7),
                                              ("intensity", 12, 7), ("ring", 16, 4),
                                              ("timestamp", 18, 8))]
    raws = []
    with Writer(path, version=8) as writer:
        conn = writer.add_connection("/lidar_points", Cloud.__msgtype__, typestore=TYPESTORE)
        for frame in range(1, 10):
            along = np.arange(2, 42, 0.05, dtype=np.float32) if rails else None
            rail_count = len(along) * 2 if rails else 32
            count = rail_count + (31 if obstacle else 0)
            points = np.zeros(count, dtype=np.dtype({
                "names": ["x", "y", "z", "intensity", "ring", "timestamp"],
                "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f8"],
                "offsets": [0, 4, 8, 12, 16, 18], "itemsize": 26}))
            if rails:
                points["x"][:rail_count] = np.concatenate((0.005 * along**2 - 0.75,
                                                            0.005 * along**2 + 0.75))
                points["y"][:rail_count] = np.tile(-along, 2)
                points["z"][:rail_count] = -1.1
            else:
                points["x"][:rail_count] = np.linspace(-0.5, 0.5, 32)
                points["y"][:rail_count] = -10
                points["z"][:rail_count] = -1.5
            if obstacle:
                points["x"][rail_count:] = 0
                points["y"][rail_count:] = obstacle_y
                points["z"][rail_count:] = obstacle_z
            points["intensity"] = 5
            points["timestamp"] = frame
            raw_data = np.frombuffer(points.tobytes(), dtype=np.uint8)
            msg = Cloud(header=Header(stamp=Time(sec=frame, nanosec=0), frame_id="hesai_lidar"),
                        height=1, width=count, fields=fields, is_bigendian=False,
                        point_step=26, row_step=count * 26, data=raw_data, is_dense=False)
            raw = bytes(TYPESTORE.serialize_cdr(msg, Cloud.__msgtype__))
            raws.append(raw)
            writer.write(conn, frame * frame_interval_ns, raw)
    return raws


class ObstacleAppTests(unittest.TestCase):
    def test_placement_warning_for_duplicate_positions_during_stop(self):
        from obstacle_app.placement_warnings import placement_warnings
        poses=np.repeat(np.eye(4)[None],5,axis=0)
        times=np.arange(5)*1_000_000_000
        anchors=[dict(encounter_seconds=seconds,anchor_timestamp_ns=seconds*1_000_000_000,
                      anchor_position_world_m=[0,0,-1]) for seconds in (0,2,4)]
        warnings=placement_warnings(anchors,poses,times)
        self.assertEqual(len(warnings),1)
        self.assertIn('0, 2, 4 s',warnings[0])
        poses[1,1,3]=-10
        self.assertEqual(placement_warnings(anchors[:2],poses,times),[])

    def test_mesh_sampling_is_deterministic(self):
        a = sample_collada_surface(MODEL, 1000)
        b = sample_collada_surface(MODEL, 1000)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (1000, 3))
        self.assertGreater(float(a[:, 2].max() - a[:, 2].min()), 1.5)

    def test_all_obstacle_meshes_sample_complete_height(self):
        for relative in ("person_standing/meshes/person.dae", "person_photo/meshes/person.dae",
                         "alien_reference/meshes/alien.dae", "trump_stylized/meshes/trump.dae"):
            with self.subTest(model=relative):
                points = sample_collada_surface(MODELS_ROOT / relative, 1000)
                self.assertEqual(points.shape, (1000, 3))
                self.assertGreater(float(np.ptp(points[:, 2])), 1.4)

    def test_merged_rails_use_later_scans_for_100_metres(self):
        along = np.arange(2, 42, 0.05, dtype=np.float32)
        xyz = np.concatenate([np.column_stack((np.full_like(along, side * 0.75),
                                               -along, np.full_like(along, -1.1)))
                              for side in (-1, 1)]).astype(np.float32)
        rails = detect_rail_points(xyz)
        lower_rails = xyz.copy()
        lower_rails[:, 2] -= 0.6
        recovered = detect_rail_points(lower_rails)
        self.assertEqual(set(recovered), {"left", "right"})
        self.assertAlmostEqual(float(np.median(recovered["left"][:, 2])), -1.7, places=2)
        poses = []
        observations = []
        for distance in (0, 25, 50, 75):
            pose = np.eye(4)
            pose[1, 3] = -distance
            poses.append(pose)
            observations.append(world_rail_observation(rails, pose))
        merged = merge_rails(observations)
        self.assertGreater(merged.length_m, 100)
        first = merged.locate(poses[0], 100)
        self.assertAlmostEqual(float(first.position[1]), -100, places=1)
        with self.assertRaisesRegex(ValueError, "extend only"):
            merged.locate(poses[-1], 100)
        extended = merged.locate(poses[-1], 100, extrapolate=True)
        self.assertGreater(extended.extrapolated_m, 50)

    def test_merged_distance_is_arc_length_on_a_curve(self):
        along = np.arange(0, 111, 0.5, dtype=np.float32)
        observations = [{side: np.column_stack((0.005 * along**2 + offset,
                                                -along, np.full_like(along, -1.1)))
                         for side, offset in (("left", -0.75), ("right", 0.75))}]
        merged = merge_rails(observations)
        placement = merged.locate(np.eye(4), 100)
        self.assertGreater(float(placement.position[1]), -100)
        self.assertGreater(float(placement.position[0]), 20)
        self.assertLess(float(np.linalg.norm(placement.position[:2])), 100)

    def test_train_envelope_filters_isolated_points_above_rails_inside_width(self):
        along = np.arange(0, 21, 0.5, dtype=np.float32)
        center = np.column_stack((np.zeros_like(along), -along,
                                  np.full_like(along, -1.1)))
        prediction = RailPrediction(center, along)
        obstacle = np.tile(np.array([[0, -10, -0.5]], dtype=np.float32), (31, 1))
        rails = np.tile(np.array([[0.75, -10, -1.1]], dtype=np.float32), (100, 1))
        platform = np.tile(np.array([[1.2, -10, -0.5]], dtype=np.float32), (100, 1))
        overhead = np.tile(np.array([[0, -10, 2.1]], dtype=np.float32), (100, 1))
        just_above = np.array([[0, -10, -1.09]], dtype=np.float32)
        xyz = np.concatenate((obstacle, rails, platform, overhead, just_above))
        found = detect_obstacle_points(xyz, prediction)
        self.assertTrue(found.detected)
        self.assertEqual(found.point_count, 31)
        self.assertAlmostEqual(found.distance_m, 10)
        self.assertEqual(len(found.points), 31)
        np.testing.assert_array_equal(found.rejected_points, just_above)
        self.assertFalse(detect_obstacle_points(rails, prediction).detected)

    def test_merged_predictor_returns_path_ahead_in_lidar_coordinates(self):
        center = np.column_stack((np.zeros(101), -np.arange(101),
                                  np.full(101, -1.1))).astype(np.float32)
        merged = type("Map", (), {"center": center, "lengths": np.arange(101)})()
        pose = np.eye(4)
        pose[1, 3] = -20
        predicted = MergedRailPredictor(merged).predict(pose, 10)
        self.assertAlmostEqual(float(predicted.center[0, 1]), 0)
        self.assertAlmostEqual(float(predicted.center[-1, 1]), -10)
        self.assertAlmostEqual(float(predicted.distance_along_m[-1]), 10)

    def test_pose_anchored_trail_covers_long_run_after_global_merge_cutoff(self):
        anchors = [np.array([0.1 * np.sin(y / 20), -y, -1.1])
                   for y in np.arange(0, 200, 0.5)]
        tail = np.column_stack((np.zeros(20), -np.arange(201, 221),
                                np.full(20, -1.1)))
        merged = merged_trail_from_anchors(anchors, tail)
        self.assertGreater(merged.length_m, 210)
        pose = np.eye(4)
        pose[1, 3] = -150
        predicted = MergedRailPredictor(merged).predict(pose, 30)
        self.assertGreater(len(predicted.center), 40)

    def test_backend_adds_points_to_every_frame_and_writes_replayable_bag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            raws = make_bag(source, rails=True)
            scenario = Scenario(source, "/lidar_points", 0, 2, MODEL,
                                distance_m=15)
            result = MeshSamplingBackend().run(scenario, output, lambda *_: None)
            self.assertEqual(result, output)
            self.assertTrue((output / "metadata.yaml").is_file())
            provenance = json.loads((output / "obstacle_scenario.json").read_text())
            self.assertEqual(provenance["modified_timestamps_ns"], [i * 100_000_000 for i in range(1, 10)])
            self.assertEqual(provenance["distance_m"], 15)
            self.assertEqual(provenance["placement"], "stationary_world_anchor")
            self.assertEqual(provenance["anchor_timestamp_ns"], 100_000_000)
            self.assertEqual(provenance["odometry"], "KISS-ICP")
            self.assertGreater(provenance["observed_rail_length_m"], 30)
            self.assertEqual(provenance["skipped_frames"], 0)
            self.assertTrue((output / "merged_rails.npz").is_file())
            self.assertEqual(provenance["every_n_frames"], 1)
            self.assertEqual(provenance["backend"], "ray_casting")
            self.assertTrue(provenance["sensor_model"]["approximate"])
            with AnyReader([output], default_typestore=TYPESTORE) as reader:
                messages = list(reader.messages())
                self.assertEqual(len(messages), 9)
                self.assertEqual([ts for _, ts, _ in messages], [i * 100_000_000 for i in range(1, 10)])
                for index, (conn, _, raw) in enumerate(messages, 1):
                    self.assertEqual(conn.topic, "/lidar_points")
                    msg = reader.deserialize(raw, conn.msgtype)
                    original = TYPESTORE.deserialize_cdr(raws[index - 1], conn.msgtype)
                    statistics = provenance["frame_counts"][str(index * 100_000_000)]
                    added_count = statistics["added"]
                    self.assertGreater(added_count, 0)
                    self.assertEqual(msg.width, original.width - statistics["removed"] + added_count)
                    self.assertEqual(msg.header.frame_id, "hesai_lidar")
                    if not statistics["removed"]:
                        np.testing.assert_array_equal(msg.data[:len(original.data)], original.data)
                    xyz = valid_xyz(msg)
                    self.assertEqual(len(xyz), original.width - statistics["removed"] + added_count)
                    added = xyz[-added_count:]
                    self.assertLess(abs(float(np.median(added[:, 1])) + 15), 0.5)
                    self.assertGreater(float(added[:, 2].max() - added[:, 2].min()), 1.5)
            # Seeking into the exported bag must identify the synthetic points.
            from fastapi.testclient import TestClient
            client = TestClient(create_app(data_root=root, output_root=root, model_path=MODEL))
            with client.websocket_connect(
                "/ws/playback?bag_id=output%2Foutput&topic=%2Flidar_points&start=0.2&paused=1"
            ) as ws:
                ws.send_json({"action": "step"})
                first = ws.receive_json()
                self.assertEqual(first["obstacle_points"], provenance["frame_counts"][first["timestamp_ns"]]["added"])
                self.assertEqual(len(ws.receive_bytes()), first["points"] * 12)

    def test_timed_multiple_obstacles_export_through_job_interface(self):
        poses = [np.eye(4) for _ in range(9)]
        for index, pose in enumerate(poses):
            pose[1, 3] = -index * 5
        class MovingOdometry:
            def __init__(self, config):
                self.last_delta = np.eye(4)
                self.index = 0
            def register_frame(self, *_):
                self.last_pose = poses[self.index]
                self.index += 1
        rail_map = MergedRailMap(np.array([[0, 0, -1.1], [0, -50, -1.1]],
                                         dtype=np.float32), np.array([0., 50.]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            make_bag(source, rails=True)
            scenario = Scenario(source, "/lidar_points", 0, 2, obstacles=(
                ObstacleSpec("first", MODEL, "person_standing", .3),
                ObstacleSpec("second", MODEL, "person_standing", .5, 2, .4)))
            with patch("obstacle_app.sampling_backend.merge_rails", return_value=rail_map), \
                 patch("obstacle_app.sampling_backend.KissICP", MovingOdometry):
                MeshSamplingBackend().run(scenario, output, lambda *_: None)
            metadata = json.loads((output / "obstacle_scenario.json").read_text())
            self.assertEqual(len(metadata["obstacles"]), 2)
            np.testing.assert_allclose(metadata["obstacles"][0]["anchor_position_world_m"], [0, -15, -1.1])
            np.testing.assert_allclose(metadata["obstacles"][1]["anchor_position_world_m"], [-2, -25, -.7])
            self.assertGreater(metadata["total_added_points"], 0)
            self.assertGreater(metadata["performance"]["export_phase_seconds"],0)
            self.assertIn("render_s",metadata["performance"]["stage_totals_seconds"])
            with patch("obstacle_app.sampling_backend.merge_rails", side_effect=ValueError("global gap")), \
                 patch("obstacle_app.sampling_backend.KissICP", MovingOdometry):
                fallback_output = root / "fallback"
                MeshSamplingBackend().run(scenario, fallback_output, lambda *_: None)
            fallback = json.loads((fallback_output / "obstacle_scenario.json").read_text())
            self.assertEqual(fallback["rail_map_method"], "pose_anchored_nearby_rails")
            self.assertGreater(fallback["total_added_points"], 0)
            with AnyReader([output], default_typestore=TYPESTORE) as reader:
                frames = list(reader.messages())
                self.assertEqual(len(frames), 9)
                for conn, timestamp, raw in frames:
                    count = metadata["frame_counts"][str(timestamp)]["added"]
                    if timestamp >= 700_000_000:
                        self.assertEqual(count, 0)
                    if count:
                        added = valid_xyz(reader.deserialize(raw, conn.msgtype))[-count:]
                        self.assertTrue(np.all(added[:, 1] <= 0))

    def test_timed_stationary_recording_rejects_sensor_overlap_before_writing(self):
        class StationaryOdometry:
            def __init__(self, config):
                self.last_delta = np.eye(4)
                self.last_pose = np.eye(4)
            def register_frame(self, *_):
                pass
        rail_map = MergedRailMap(np.array([[0., 0., -1.1], [0., -30., -1.1]]),
                                 np.array([0., 30.]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            make_bag(source, rails=True)
            scenario = Scenario(source, "/lidar_points", 0, 2, obstacles=(
                ObstacleSpec("person", MODEL, "person_standing", .3),))
            with patch("obstacle_app.sampling_backend.merge_rails", return_value=rail_map), \
                 patch("obstacle_app.sampling_backend.KissICP", StationaryOdometry):
                with self.assertRaisesRegex(ValueError, "moves only 0.00 m"):
                    MeshSamplingBackend().run(scenario, output, lambda *_: None)
            self.assertFalse(output.exists())

    def test_timed_anchor_interpolates_and_does_not_snap_to_truncated_route(self):
        from obstacle_app.timed_placement import timed_anchor
        poses = [np.eye(4), np.eye(4)]
        poses[1][1, 3] = -20
        rails = MergedRailMap(np.array([[0, 0, -1.1], [0, -5, -1.1]]), np.array([0., 5.]))
        spec = ObstacleSpec("person", MODEL, "person_standing", .5, 2, .4)
        bases = [np.array([0., 0., -1.1]), np.array([0., -20., -1.1])]
        position, direction, timestamp = timed_anchor(spec, 0, [0, 1_000_000_000], poses, bases, rails)
        np.testing.assert_allclose(position, [-2, -10, -.7])
        np.testing.assert_allclose(direction, [0, -1, 0])
        self.assertEqual(timestamp, 500_000_000)
        with self.assertRaisesRegex(ValueError, "no reliable rails"):
            timed_anchor(spec, 0, [0, 1_000_000_000], poses, [None, None], rails)

    def test_stationary_obstacle_approaches_and_is_omitted_after_passage(self):
        poses = []
        for frame in range(9):
            yaw, pitch = frame * 0.04, frame * 0.01
            rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                           [-np.sin(pitch), 0, np.cos(pitch)]])
            pose = np.eye(4)
            pose[:3, :3] = rz @ ry
            pose[:3, 3] = [frame * 0.1, -frame * 5, frame * 0.02]
            poses.append(pose)

        class MovingOdometry:
            def __init__(self, config):
                self.last_delta = np.eye(4)
                self.index = 0

            def register_frame(self, *_):
                self.last_pose = poses[self.index]
                self.index += 1

        center = np.array([[0, 0, -1.1], [0, -20, -1.1]], dtype=np.float32)
        rail_map = MergedRailMap(center, np.array([0.0, 20.0]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            raws = make_bag(source, rails=True)
            scenario = Scenario(source, "/lidar_points", 0, 2, MODEL, distance_m=15,
                                ground_z_m=-1.3, ground_offset_m=0.2,
                                every_n_frames=3)
            with patch("obstacle_app.sampling_backend.merge_rails", return_value=rail_map), \
                 patch("obstacle_app.sampling_backend.KissICP", MovingOdometry), \
                 patch.object(MergedRailMap, "locate", autospec=True,
                              side_effect=MergedRailMap.locate) as locate:
                MeshSamplingBackend().run(scenario, output, lambda *_: None)
            self.assertEqual(locate.call_count, 1)
            provenance = json.loads((output / "obstacle_scenario.json").read_text())
            np.testing.assert_allclose(provenance["anchor_position_world_m"], [0, -15, -1.1])
            self.assertEqual(provenance["modified_timestamps_ns"], [300_000_000])
            self.assertEqual(provenance["sensor_model"]["obstacle_forward_axis"], "-Y")
            vertices, _ = load_collada_mesh(MODEL)
            world_mesh = place_points(vertices, np.array([0, -15, -1.1]),
                                      np.array([0, -1]), -1.1)
            distances = []
            with AnyReader([output], default_typestore=TYPESTORE) as reader:
                for index, (conn, _, raw) in enumerate(reader.messages()):
                    if (index + 1) % 3 or index >= 3:
                        self.assertEqual(raw, raws[index])
                        if not (index + 1) % 3:
                            self.assertEqual(provenance["frame_counts"][str((index + 1) * 100_000_000)]["added"], 0)
                            self.assertEqual(provenance["frame_counts"][str((index + 1) * 100_000_000)]["removed"], 0)
                        continue
                    msg = reader.deserialize(raw, conn.msgtype)
                    original = TYPESTORE.deserialize_cdr(raws[index], conn.msgtype)
                    statistics = provenance["frame_counts"][str((index + 1) * 100_000_000)]
                    self.assertGreater(statistics["added"], 0)
                    if not statistics["removed"]:
                        np.testing.assert_array_equal(msg.data[:len(original.data)], original.data)
                    added = valid_xyz(msg)[-statistics["added"]:]
                    self.assertTrue(np.all(added[:, 1] <= 0))
                    world = added @ poses[index][:3, :3].T + poses[index][:3, 3]
                    self.assertTrue(np.all(world >= world_mesh.min(axis=0) - 3e-5))
                    self.assertTrue(np.all(world <= world_mesh.max(axis=0) + 3e-5))
                    self.assertLess(abs(float(np.median(world[:, 1])) + 15), .4)
                    distances.append(float(np.linalg.norm(np.mean(added, axis=0))))
            self.assertLess(distances[0], 6)  # Train approaches the initial 15 m position.

    def test_stationary_anchor_requires_initial_rail_coverage_or_explicit_extension(self):
        rail_map = MergedRailMap(np.array([[0, 0, -1.1], [0, -20, -1.1]],
                                         dtype=np.float32), np.array([0.0, 20.0]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            make_bag(source, rails=True)
            scenario = Scenario(source, "/lidar_points", 0, 2, MODEL, distance_m=30)
            with patch("obstacle_app.sampling_backend.merge_rails", return_value=rail_map):
                with self.assertRaisesRegex(ValueError, "Cannot anchor obstacle at the start"):
                    MeshSamplingBackend().run(scenario, output, lambda *_: None)
            self.assertFalse(output.exists())
            from dataclasses import replace
            with patch("obstacle_app.sampling_backend.merge_rails", return_value=rail_map):
                MeshSamplingBackend().run(replace(scenario, rail_end_policy="extrapolate"),
                                          output, lambda *_: None)
            provenance = json.loads((output / "obstacle_scenario.json").read_text())
            self.assertAlmostEqual(provenance["extrapolated_m_max"], 10)
            self.assertEqual(len(provenance["modified_timestamps_ns"]), 9)

    def test_backend_removes_shadow_records_and_preserves_other_topics(self):
        rail_map = MergedRailMap(np.array([[0, 0, -1.1], [0, -20, -1.1]],
                                         dtype=np.float32), np.array([0.0, 20.0]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            originals = make_bag(source, rails=True, obstacle=True, obstacle_z=0)
            with AnyReader([source], default_typestore=TYPESTORE) as reader:
                original_clouds = list(reader.messages())
            combined = root / "combined"
            with Writer(combined, version=8) as writer:
                clouds = writer.add_connection("/lidar_points", "sensor_msgs/msg/PointCloud2",
                                               typestore=TYPESTORE)
                status = writer.add_connection("/status", "std_msgs/msg/String", typestore=TYPESTORE)
                status_raw = bytes(TYPESTORE.serialize_cdr(
                    TYPESTORE.types["std_msgs/msg/String"](data="unchanged"), "std_msgs/msg/String"))
                writer.write(status, 150_000_000, status_raw)
                for _, timestamp, raw in original_clouds:
                    writer.write(clouds, timestamp, raw)
            scenario = Scenario(combined, "/lidar_points", 0, 2, MODEL,
                                distance_m=5, every_n_frames=3)
            with patch("obstacle_app.sampling_backend.merge_rails", return_value=rail_map):
                MeshSamplingBackend().run(scenario, output, lambda *_: None)
            provenance = json.loads((output / "obstacle_scenario.json").read_text())
            self.assertGreaterEqual(provenance["total_removed_points"], 31 * 3)
            with AnyReader([output], default_typestore=TYPESTORE) as reader:
                index = 0
                for conn, timestamp, raw in reader.messages():
                    if conn.topic == "/status":
                        self.assertEqual(timestamp, 150_000_000)
                        self.assertEqual(raw, status_raw)
                        continue
                    index += 1
                    if index % 3:
                        self.assertEqual(raw, originals[index - 1])
                        continue
                    msg = reader.deserialize(raw, conn.msgtype)
                    xyz = valid_xyz(msg)
                    self.assertFalse(np.any((xyz[:, 0] == 0) & (xyz[:, 1] == -10) & (xyz[:, 2] == 0)))
                    counts = provenance["frame_counts"][str(timestamp)]
                    original = TYPESTORE.deserialize_cdr(originals[index - 1], conn.msgtype)
                    kept_size = (msg.width - counts["added"]) * msg.point_step
                    records = {original.data[i:i + original.point_step].tobytes()
                               for i in range(0, len(original.data), original.point_step)}
                    for i in range(0, kept_size, msg.point_step):
                        self.assertIn(msg.data[i:i + msg.point_step].tobytes(), records)

    def test_export_removes_inherited_rear_obstacles_and_carries_tail_labels(self):
        rail_map = MergedRailMap(np.array([[0, 0, -1.1], [0, -30, -1.1]],
                                         dtype=np.float32), np.array([0.0, 30.0]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean, source, output = root / "clean", root / "parent", root / "child"
            originals = make_bag(clean, rails=True)
            source_counts = {}
            with Writer(source, version=8) as writer:
                conn = writer.add_connection("/lidar_points", "sensor_msgs/msg/PointCloud2", typestore=TYPESTORE)
                for frame, raw in enumerate(originals, 1):
                    msg = TYPESTORE.deserialize_cdr(raw, conn.msgtype)
                    append_xyz(msg, np.array([[0, 10, 0], [0, -8, 0]], dtype=np.float32))
                    timestamp = frame * 100_000_000
                    writer.write(conn, timestamp, TYPESTORE.serialize_cdr(msg, conn.msgtype))
                    source_counts[str(timestamp)] = {"added": 2}
            (source / "obstacle_scenario.json").write_text(json.dumps({"frame_counts": source_counts}))
            # Rear cleanup applies even when this scan is not adding a new model.
            scenario = Scenario(source, "/lidar_points", 0, 2, MODEL, distance_m=15, every_n_frames=9)
            with patch("obstacle_app.sampling_backend.merge_rails", return_value=rail_map):
                MeshSamplingBackend().run(scenario, output, lambda *_: None)
            provenance = json.loads((output / "obstacle_scenario.json").read_text())
            with AnyReader([output], default_typestore=TYPESTORE) as reader:
                for conn, timestamp, raw in reader.messages():
                    msg = reader.deserialize(raw, conn.msgtype)
                    xyz = valid_xyz(msg)
                    self.assertFalse(np.any(xyz[:, 1] > 0))
                    self.assertTrue(np.any((xyz[:, 1] == -8) & (xyz[:, 0] == 0)))
                    counts = provenance["frame_counts"][str(timestamp)]
                    self.assertEqual(counts["removed_rear_obstacle_points"], 1)
                    self.assertEqual(counts["synthetic_points"], counts["added"] + 1)
            from fastapi.testclient import TestClient
            client = TestClient(create_app(data_root=root, output_root=root, model_path=MODEL))
            with client.websocket_connect("/ws/playback?bag_id=output%2Fchild&topic=%2Flidar_points&paused=1") as ws:
                ws.send_json({"action": "step"})
                frame = ws.receive_json()
                self.assertEqual(frame["obstacle_points"], 1)
                self.assertEqual(len(ws.receive_bytes()), frame["points"] * 12)

    def test_web_job_uses_injected_backend(self):
        from fastapi.testclient import TestClient

        class FakeBackend:
            calls = []
            def run(self, scenario, output_directory, progress):
                self.calls.append(scenario)
                progress(0.5, "Fake progress")
                with Writer(output_directory, version=8):
                    pass
                (output_directory / "obstacle_scenario.json").write_text(json.dumps({"warnings":["Example placement warning"]}))
                return output_directory

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root / "data", root / "output"
            data.mkdir()
            output.mkdir()
            make_bag(data / "sample", frame_interval_ns=1_000_000_000)
            fake = FakeBackend()
            client = TestClient(create_app(fake, data, output, MODEL))
            self.assertEqual({m["id"] for m in client.get("/api/models").json()},
                             {"person_standing", "person_photo", "alien_reference", "trump_stylized",
                              "box_square", "box_long_thin", "wire_hanging", "niva_car"})
            response = client.post("/api/jobs", json={"bag_id": "source/sample",
                "topic": "/lidar_points", "obstacles": [
                    {"id": "alien", "model_id": "alien_reference", "encounter_seconds": 2,
                     "lateral_offset_m": -.8, "height_above_track_m": .3},
                    {"id": "person", "model_id": "person_standing", "encounter_seconds": 4}]})
            self.assertEqual(response.status_code, 202)
            job_id = response.json()["id"]
            for _ in range(100):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["state"] in ("done", "failed"):
                    break
                time.sleep(0.02)
            self.assertEqual(job["state"], "done", job)
            self.assertEqual(job["warnings"],["Example placement warning"])
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(len(fake.calls[0].obstacles), 2)
            first = fake.calls[0].obstacles[0]
            self.assertEqual(first.encounter_seconds, 2)
            self.assertEqual(first.lateral_offset_m, -.8)
            self.assertEqual(first.height_above_track_m, .3)
            self.assertEqual(first.model_id, "alien_reference")
            self.assertEqual(first.model_path, MODELS_ROOT / "alien_reference/meshes/alien.dae")
            self.assertEqual(fake.calls[0].every_n_frames, 1)
            self.assertEqual(fake.calls[0].start_seconds, 0)
            with AnyReader([data / "sample"], default_typestore=TYPESTORE) as reader:
                self.assertAlmostEqual(fake.calls[0].duration_seconds,
                                       reader.duration / 1e9)
            self.assertTrue(any(b["id"] == job["output_bag_id"] for b in client.get("/api/bags").json()))
            invalid = client.post("/api/jobs", json={"bag_id": "source/sample",
                "topic": "/lidar_points", "obstacles": [{"id": "bad", "model_id": "../../other", "encounter_seconds": .2}]})
            self.assertEqual(invalid.status_code, 400)
            base_request = {"bag_id": "source/sample", "topic": "/lidar_points"}
            for objects, status in [([], 422),
                ([{"id":"side", "model_id":"person_standing", "encounter_seconds":2, "lateral_offset_m":1.1}], 422),
                ([{"id":"side", "model_id":"person_standing", "encounter_seconds":2, "lateral_offset_m":-1.1}], 422),
                ([{"id": "early", "model_id": "person_standing", "encounter_seconds": 1.9}], 400),
                ([{"id": "near_end", "model_id": "person_standing", "encounter_seconds": 6.1}], 400),
                ([{"id": "late", "model_id": "person_standing", "encounter_seconds": 7}], 400),
                ([{"id": "same", "model_id": "person_standing", "encounter_seconds": .2}] * 2, 400),
                ([{"id": "low", "model_id": "person_standing", "encounter_seconds": .2,
                   "height_above_track_m": -1}], 422)]:
                rejected = client.post("/api/jobs", json={**base_request, "obstacles": objects})
                self.assertEqual(rejected.status_code, status, rejected.json())

            make_bag(data / "short")
            rejected = client.post("/api/jobs", json={"bag_id":"source/short", "topic":"/lidar_points",
                "obstacles":[{"id":"short", "model_id":"person_standing", "encounter_seconds":2}]})
            self.assertEqual(rejected.status_code,400)
            self.assertIn("longer than 4",rejected.json()["detail"])

    def test_web_playback_can_step_frames(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root / "data", root / "output"
            data.mkdir()
            output.mkdir()
            make_bag(data / "sample")
            client = TestClient(create_app(data_root=data, output_root=output, model_path=MODEL))
            with client.websocket_connect(
                "/ws/playback?bag_id=source%2Fsample&topic=%2Flidar_points&start=0&paused=1"
            ) as ws:
                ws.send_json({"action": "step"})
                first = ws.receive_json()
                points = ws.receive_bytes()
                self.assertEqual(first["frame"], 1)
                self.assertEqual(first["points"], 32)
                self.assertEqual(len(points), 32 * 3 * 4)
                ws.send_json({"action": "step"})
                second = ws.receive_json()
                self.assertEqual(second["frame"], 2)
                ws.receive_bytes()
                ws.send_json({"action": "resume"})
                for _ in range(7):
                    self.assertEqual(ws.receive_json()["type"], "frame")
                    ws.receive_bytes()
                self.assertEqual(ws.receive_json(), {"type": "end", "frames": 9})

    def test_detection_job_supplies_predicted_rails_to_playback(self):
        from fastapi.testclient import TestClient

        def fake_map_builder(bag, topic, speed, path, progress):
            self.assertEqual(topic, "/lidar_points")
            self.assertEqual(speed, 0)
            with AnyReader([bag], default_typestore=TYPESTORE) as reader:
                times = [timestamp for _, timestamp, _ in reader.messages()]
            center = np.column_stack((np.zeros(81), -np.arange(81) * 0.5,
                                      np.full(81, -1.1))).astype(np.float32)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, center=center, lengths=np.arange(81) * 0.5,
                                poses=np.repeat(np.eye(4)[None], len(times), axis=0),
                                timestamps_ns=times)
            progress(1, "Ready")
            return path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root / "data", root / "output"
            data.mkdir()
            output.mkdir()
            make_bag(data / "sample", rails=True)
            client = TestClient(create_app(data_root=data, output_root=output,
                                           model_path=MODEL,
                                           detection_map_builder=fake_map_builder,
                                           scan_detector_factory=fake_scan_detector))
            response = client.post("/api/detections", json={"bag_id": "source/sample",
                                                           "topic": "/lidar_points"})
            self.assertEqual(response.status_code, 202)
            job_id = response.json()["id"]
            for _ in range(100):
                job = client.get(f"/api/detections/{job_id}").json()
                if job["state"] in ("done", "failed"):
                    break
                time.sleep(0.02)
            self.assertEqual(job["state"], "done", job)
            self.assertFalse(list((output / ".detection_maps").glob("*.sqlite")))
            self.assertFalse((output / ".detection_maps").exists())
            with client.websocket_connect(
                f"/ws/playback?bag_id=source%2Fsample&topic=%2Flidar_points&paused=1"
                f"&detection_id={job_id}"
            ) as ws:
                ws.send_json({"action": "step"})
                frame = ws.receive_json()
                self.assertGreater(len(frame["rail_center"]), 10)
                self.assertFalse(frame["detected"])
                self.assertEqual(frame["candidate_points"], 0)
                self.assertEqual(len(ws.receive_bytes()), frame["points"] * 12)

    def test_app_start_does_not_prepare_a_map_or_load_inference(self):
        from fastapi.testclient import TestClient

        called = []

        def fake_map_builder(bag, topic, speed, path, progress):
            called.append((bag.name, topic, speed))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, center=np.array([[0, 0, -1], [0, -1, -1]]),
                                lengths=np.array([0, 1]),
                                poses=np.repeat(np.eye(4)[None], 9, axis=0),
                                timestamps_ns=np.arange(1, 10) * 100_000_000)
            progress(1, "Ready")
            return path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root / "data", root / "output"
            data.mkdir()
            output.mkdir()
            make_bag(data / "cloud_with_fake_obj")
            with TestClient(create_app(data_root=data, output_root=output,
                                       detection_map_builder=fake_map_builder,
                                           scan_detector_factory=fake_scan_detector)):
                self.assertFalse((output / ".detection_maps").exists())
            self.assertEqual(called, [])
            self.assertEqual(len(list((output / ".detection_maps").glob("*.npz"))), 0)

    def test_detection_overlay_updates_paused_frame_without_replaying_bag(self):
        from fastapi.testclient import TestClient

        def fake_map_builder(bag, topic, speed, path, progress):
            with AnyReader([bag], default_typestore=TYPESTORE) as reader:
                times = [timestamp for _, timestamp, _ in reader.messages()]
            center = np.column_stack((np.zeros(81), -np.arange(81) * 0.5,
                                      np.full(81, -1.1))).astype(np.float32)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, center=center, lengths=np.arange(81) * 0.5,
                                poses=np.repeat(np.eye(4)[None], len(times), axis=0),
                                timestamps_ns=times)
            progress(1, "Ready")
            return path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root / "data", root / "output"
            data.mkdir()
            output.mkdir()
            make_bag(data / "sample", rails=True, obstacle=True)
            client = TestClient(create_app(data_root=data, output_root=output,
                                           model_path=MODEL,
                                           detection_map_builder=fake_map_builder,
                                           scan_detector_factory=fake_scan_detector))
            with client.websocket_connect(
                "/ws/playback?bag_id=source%2Fsample&topic=%2Flidar_points&paused=1"
            ) as ws:
                ws.send_json({"action": "step"})
                first = ws.receive_json()
                ws.receive_bytes()
                response = client.post("/api/detections", json={"bag_id": "source/sample",
                                                               "topic": "/lidar_points"})
                job_id = response.json()["id"]
                for _ in range(100):
                    job = client.get(f"/api/detections/{job_id}").json()
                    if job["state"] in ("done", "failed"):
                        break
                    time.sleep(0.02)
                self.assertEqual(job["state"], "done", job)
                self.assertEqual([bag["id"] for bag in client.get("/api/bags").json()],
                                 ["source/sample"])
                result = client.get(f"/api/detections/{job_id}/frame", params={
                    "timestamp_ns": first["timestamp_ns"]}).json()
                self.assertTrue(result["detected"])
                self.assertEqual(result["candidate_points"], 31)
                ws.send_json({"action": "show_detection", "detection_id": job_id})
                shown = ws.receive_json()
                self.assertEqual(shown["frame"], first["frame"])
                self.assertEqual(shown["timestamp_ns"], first["timestamp_ns"])
                self.assertTrue(shown["detected"])
                self.assertEqual(shown["detected_points"], 31)
                ws.receive_bytes()
                ws.send_json({"action": "hide_detection"})
                hidden = ws.receive_json()
                self.assertEqual(hidden["frame"], first["frame"])
                self.assertIsNone(hidden["rail_center"])
                ws.receive_bytes()
                ws.send_json({"action": "step"})
                next_frame = ws.receive_json()
                self.assertEqual(next_frame["frame"], first["frame"] + 1)
                ws.receive_bytes()

    def test_running_uvicorn_accepts_websocket_playback(self):
        """Exercise the ASGI server, which TestClient alone does not cover."""
        from websockets.sync.client import connect

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root / "data", root / "output"
            data.mkdir()
            output.mkdir()
            make_bag(data / "sample")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = os.environ | {"OBSTACLE_DATA_DIR": str(data), "OBSTACLE_OUTPUT_DIR": str(output)}
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "obstacle_app.web:app",
                                        "--host", "127.0.0.1", "--port", str(port)],
                                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for _ in range(50):
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/api/bags", timeout=0.2):
                            break
                    except OSError:
                        time.sleep(0.1)
                else:
                    self.fail("Uvicorn did not start")
                with connect(f"ws://127.0.0.1:{port}/ws/playback?bag_id=source%2Fsample"
                             "&topic=%2Flidar_points&start=0&paused=1", open_timeout=3) as ws:
                    ws.send(json.dumps({"action": "step"}))
                    self.assertEqual(json.loads(ws.recv())["frame"], 1)
                    self.assertEqual(len(ws.recv()), 32 * 3 * 4)
            finally:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
