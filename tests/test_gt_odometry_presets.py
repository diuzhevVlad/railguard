import json
from types import SimpleNamespace

import numpy as np
import pytest

from obstacle_app.gt_odometry import (
    GT_ODOMETRY_PRESETS, detection_initial_speed, gt_odometry_preset, odometry_input,
)
from obstacle_app.tracked_bag_detection import pose_identity
from railpath.tracking_odometry import TrackingOdometry


def test_fixed_speeds_including_zero_and_export_provenance(tmp_path):
    expected = {
        "roundT_doubleT": 10, "roundT_pressureGate_roundT": 10,
        "roundT_squareT_pressureGate_squareT": 10, "squareT_platform_squareT_switch": 15,
        "doubleT_platform": 0, "new_data": 0, "doubleT_obstacle": 0,
        "cloud_with_fake_obj": 0, "synthetic_switch_cloud_only": 0,
        "synthetic_tunnels_cloud_only": 10, "synthetic_platform_cloud_only": 16,
    }
    for name, speed in expected.items():
        assert detection_initial_speed(tmp_path/name, 99) == speed
    assert detection_initial_speed(tmp_path/"unknown", 7) == 7
    assert detection_initial_speed(tmp_path/"unknown") is None
    assert detection_initial_speed(tmp_path/"unknown", 0) == 0
    assert gt_odometry_preset(tmp_path/"unknown") is None
    exported = tmp_path/"export"
    exported.mkdir()
    (exported/"obstacle_scenario.json").write_text(json.dumps({
        "source_bag": "/data/cloud_with_fake_obj",
    }))
    assert detection_initial_speed(exported, 3) == 0


@pytest.mark.parametrize("big_endian", [False, True])
def test_full_input_and_aligned_timestamps_with_row_padding(big_endian):
    endian = ">" if big_endian else "<"
    data = bytearray(144)
    xyz = np.array([[.1, 0, 0], [0, 0, 0], [1, 2, 3],
                    [4, 5, 6], [np.nan, 1, 1], [7, 8, 9]])
    times = [0, .1, np.nan, .3, .4, .5]
    fields = [SimpleNamespace(name=n, offset=i*4, datatype=7, count=1)
              for i, n in enumerate(("x", "y", "z"))]
    fields.append(SimpleNamespace(name="timestamp", offset=12, datatype=8, count=1))
    for i, name in enumerate(("x", "y", "z")):
        np.ndarray((2, 3), dtype=endian+"f4", buffer=data, offset=i*4, strides=(72,20))[:] = xyz[:,i].reshape(2,3)
    np.ndarray((2,3), dtype=endian+"f8", buffer=data, offset=12, strides=(72,20))[:] = np.array(times).reshape(2,3)
    msg = SimpleNamespace(fields=fields, height=2, width=3, point_step=20,
                          row_step=72, is_bigendian=big_endian, data=data)
    original = bytes(data)
    points, timestamps = odometry_input(msg)
    np.testing.assert_allclose(points, xyz[[0,3,5]])
    np.testing.assert_array_equal(timestamps, [0, .3, .5])
    assert bytes(data) == original
    msg.fields = fields[:3]
    points, timestamps = odometry_input(msg)
    assert len(points) == 4 and not len(timestamps)


def test_gt_processing_uses_full_input_and_fixed_seed_interval():
    class Engine:
        def __init__(self):
            self.last_pose = np.eye(4)
            self.last_delta = np.eye(4)
        def register_frame(self, points, timestamps):
            self.points, self.timestamps = points.copy(), timestamps.copy()
            self.last_pose = self.last_pose @ self.last_delta
    engine = Engine()
    odom = TrackingOdometry(10, factory=lambda:engine, point_stride=1,
                            deskew=True, min_distance_m=0, initial_scan_dt_s=.0805)
    points = np.tile([.1, 0., 0.], (80,1))
    times = np.arange(80)*.001
    odom.update(points, 1_000_000_000, times)
    pose, reset = odom.update(points, 1_200_000_000, times)
    assert not reset
    assert pose[1,3] == pytest.approx(-.805)  # Not receipt-time dt=0.2 s.
    np.testing.assert_array_equal(engine.points, points)
    np.testing.assert_array_equal(engine.timestamps, times)


def test_cache_identity_uses_effective_preset_and_new_processing(tmp_path):
    bag = tmp_path/"cloud_with_fake_obj"
    bag.mkdir()
    a, b = pose_identity(bag, "/lidar_points", 3), pose_identity(bag, "/lidar_points", 99)
    assert a == b
    assert a["initial_speed"] == 0
    assert a["seed_scan_dt_s"] == .1
    assert a["cache_version"] == 3
    assert a["point_stride"] == 1 and a["deskew"]


def test_manager_cannot_override_known_gt_speed(tmp_path, monkeypatch):
    from obstacle_app.learned_detection import LearnedDetectionManager
    manager = LearnedDetectionManager(detector_factory=lambda *_: None)
    monkeypatch.setattr(manager, "_load", lambda job: None)
    bag = tmp_path/"cloud_with_fake_obj"
    bag.mkdir()
    first = manager.start("source/cloud_with_fake_obj", bag, "/lidar_points", initial_speed_m_s=3)
    second = manager.start("source/cloud_with_fake_obj", bag, "/lidar_points", initial_speed_m_s=15)
    assert first is second
    assert first.initial_speed_m_s == 0
    assert manager.start("source/cloud_with_fake_obj", bag, "/lidar_points") is first
    unknown = tmp_path/"unknown"
    unknown.mkdir()
    with pytest.raises(ValueError, match="Set the detection initial speed"):
        manager.start("source/unknown", unknown, "/lidar_points")
    explicit_zero = manager.start("source/unknown", unknown, "/lidar_points", initial_speed_m_s=0)
    assert explicit_zero.initial_speed_m_s == 0
    manager.close()
