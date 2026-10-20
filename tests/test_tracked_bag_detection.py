from dataclasses import replace
from pathlib import Path
import time

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from obstacle_app.cloud import valid_xyz
from obstacle_app.scan_detection import ScanDetector
from obstacle_app.tracked_bag_detection import TrackedBagDetector
from railpath.obstacle_filter import RailPrediction
from railpath.tracked_detection import CANDIDATE_CONFIG, detect_candidates

STORE = get_typestore(Stores.ROS2_HUMBLE)


def make_bag(path):
    types = STORE.types
    frames = []
    with Writer(path, version=9) as writer:
        connection = writer.add_connection('/lidar_points', 'sensor_msgs/msg/PointCloud2', typestore=STORE)
        for i in range(8):
            stamp = 1_000_000_000 + i * 100_000_000
            xyz = np.array([[x, -20.+i, 1.] for x in (0., .01, .02, .03)], dtype='<f4')
            msg = types['sensor_msgs/msg/PointCloud2'](
                header=types['std_msgs/msg/Header'](types['builtin_interfaces/msg/Time'](1, i*100_000_000), 'lidar'),
                height=1, width=len(xyz),
                fields=[types['sensor_msgs/msg/PointField'](name, j*4, 7, 1) for j, name in enumerate(('x','y','z'))],
                is_bigendian=False, point_step=12, row_step=len(xyz)*12,
                data=np.frombuffer(xyz.tobytes(), dtype=np.uint8), is_dense=True)
            writer.write(connection, stamp, STORE.serialize_cdr(msg, connection.msgtype))
            frames.append(msg)
    return frames


def prediction():
    depths = np.arange(151., dtype=np.float32)
    return RailPrediction(np.column_stack([depths*0, -depths, depths*0]), depths,
                          rail_points=np.empty((0,3)), max_forward_m=150.)


def poses():
    stamps = 1_000_000_000 + np.arange(8)*100_000_000
    matrices = np.tile(np.eye(4), (8,1,1))
    matrices[:,1,3] = -np.arange(8)
    return stamps, matrices, np.array([True]+[False]*7)


class Odom:
    def __init__(self, speed=0.):
        self.calls = []
        self.previous = None
    def reset(self):
        self.previous = None
    def update(self, xyz, stamp):
        reset = self.previous is None or not 0 < stamp-self.previous <= 2e9
        self.previous = stamp
        self.calls.append(stamp)
        pose = np.eye(4)
        pose[1,3] = -(stamp-1_000_000_000)/100_000_000
        return pose, reset


def test_online_history_pause_seek_and_session_isolation(tmp_path):
    frames = make_bag(tmp_path/'bag')
    calls = []
    class Provider:
        def predict(self, stamp, msg):
            calls.append(stamp)
            return prediction()
    raw = ScanDetector(Provider(), filter_config=CANDIDATE_CONFIG, candidate_detector=detect_candidates)
    stamps = poses()[0]
    detector = TrackedBagDetector(raw, odometry_factory=Odom)
    # Starting at the end does not read any hidden history.
    assert not detector.detect(int(stamps[7]), frames[7])[1].detected
    assert calls == [int(stamps[7])]
    for i in range(8):
        result = detector.detect(int(stamps[i]), frames[i])
        found = result[1]
        assert found.detected == (i >= 3)
        np.testing.assert_array_equal(found.candidate_points, valid_xyz(frames[i]))
        assert found.tracks[0]['hits'] == min(i+1, 5)
        assert detector.detect(int(stamps[i]), frames[i]) is result
    assert len(detector.odometry.calls) == 9  # paused refreshes never run ICP
    assert len(calls) == 8  # raw neural cache may be shared
    other = TrackedBagDetector(raw, odometry_factory=Odom)
    assert not other.detect(int(stamps[7]), frames[7])[1].detected
    # A gap, rewind, or changed sensor frame cannot inherit confirmation.
    assert not detector.detect(int(stamps[7]+3e9), frames[7])[1].detected
    assert not detector.detect(int(stamps[0]), frames[0])[1].detected
    frames[0].header.frame_id = 'another_lidar'
    assert not detector.detect(int(stamps[0]), frames[0])[1].detected


def test_web_job_no_prepass_and_each_websocket_owns_history(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from obstacle_app import learned_detection
    from obstacle_app.web import create_app
    data, output = tmp_path/'data', tmp_path/'output'
    data.mkdir()
    output.mkdir()
    make_bag(data/'bag')
    class Runtime:
        device = 'test'
        range_m = 200
        def __init__(self, *args):
            pass
        def close(self):
            pass
        def load(self):
            pass
        def predict(self, msg):
            return prediction()
    from obstacle_app import model_worker
    monkeypatch.setattr(model_worker, 'ProcessRailRuntime', Runtime)
    from obstacle_app import tracked_bag_detection
    monkeypatch.setattr(tracked_bag_detection, 'TrackingOdometry', Odom)
    client = TestClient(create_app(data_root=data, output_root=output))
    response = client.post('/api/detections', json={'bag_id':'source/bag','topic':'/lidar_points'})
    assert response.status_code == 202
    job = response.json()
    for _ in range(100):
        job = client.get('/api/detections/'+job['id']).json()
        if job['state'] in ('done','failed'):
            break
        time.sleep(.01)
    assert job['state'] == 'done', job
    stamps = poses()[0]
    for i in (7,0,3):
        result = client.get(f"/api/detections/{job['id']}/frame", params={'timestamp_ns':int(stamps[i])}).json()
        assert not result['detected'], result
        assert result['raw_candidate_points'] == 4
        assert len(result['unconfirmed_xyz']) == 4
    with client.websocket_connect(f"/ws/playback?bag_id=source/bag&topic=/lidar_points&detection_id={job['id']}") as ws:
        frame = ws.receive_json()
        cloud = ws.receive_bytes()
        assert frame['type'] == 'frame' and not frame['detected']
        assert len(frame['unconfirmed_xyz']) == 4
        assert len(cloud) == frame['points']*12

        for i in range(1, 8):
            frame = ws.receive_json()
            ws.receive_bytes()
            assert frame['detected'] == (i >= 3)
        with client.websocket_connect(f"/ws/playback?bag_id=source/bag&topic=/lidar_points&detection_id={job['id']}&start=0.7") as other:
            frame = other.receive_json()
            other.receive_bytes()
            assert not frame['detected']
    assert not (output/'.tracking_poses').exists()
