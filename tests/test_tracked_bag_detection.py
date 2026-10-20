from dataclasses import replace
from pathlib import Path
import time

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from obstacle_app.cloud import valid_xyz
from obstacle_app.scan_detection import ScanDetector
from obstacle_app.tracked_bag_detection import TrackedBagDetector, prepare_poses
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


def test_seek_uses_recorded_history_and_pose_not_playback_order(tmp_path):
    bag = tmp_path/'bag'
    frames = make_bag(bag)
    calls = []
    class Provider:
        def predict(self, stamp, msg):
            calls.append(stamp)
            return prediction()
    raw = ScanDetector(Provider(), filter_config=CANDIDATE_CONFIG, candidate_detector=detect_candidates)
    stamps, matrices, resets = poses()
    detector = TrackedBagDetector(raw, bag, '/lidar_points', stamps, matrices, resets)
    for frame in (7, 0, 3, 2, 7, 4):
        _, found = detector.detect(int(stamps[frame]), frames[frame])
        assert found.detected == (frame >= 3)
        np.testing.assert_array_equal(found.candidate_points, valid_xyz(frames[frame]))
        assert max(found.tracks[0]['observation_frames']) == frame
        assert found.tracks[0]['hits'] == min(frame+1, 5)
    assert len(calls) == len(set(calls))
    # Even continuous timestamps cannot bridge an explicitly broken odometry segment.
    resets[6] = True
    other = TrackedBagDetector(raw, bag, '/lidar_points', stamps, matrices, resets)
    assert not other.detect(int(stamps[7]), frames[7])[1].detected


def test_pose_cache_bound_to_source_topic_speed_and_resets(tmp_path):
    bag = tmp_path/'bag'
    make_bag(bag)
    calls = []
    class Odom:
        def __init__(self, speed):
            calls.append(speed)
        def update(self, xyz, stamp):
            return np.eye(4), stamp == 1_000_000_000
    args = (bag, '/lidar_points', 3., tmp_path/'cache', lambda *_: None)
    first = prepare_poses(*args, odometry_factory=Odom)
    second = prepare_poses(*args, odometry_factory=Odom)
    assert calls == [3.]
    for a,b in zip(first, second):
        np.testing.assert_array_equal(a,b)
    prepare_poses(bag, '/lidar_points', 5., tmp_path/'cache', lambda *_: None, odometry_factory=Odom)
    assert calls == [3.,5.]
    assert len(list((tmp_path/'cache').glob('*.npz'))) == 2


def test_web_job_frame_and_websocket_confirm_same_output(tmp_path, monkeypatch):
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
        def load(self):
            pass
        def predict(self, msg):
            return prediction()
    monkeypatch.setattr(learned_detection, 'RailGuardRuntime', Runtime)
    monkeypatch.setattr(learned_detection, 'prepare_poses', lambda *args: poses())
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
        assert result['detected'] == (i>=3), result
        assert result['raw_candidate_points'] == 4
        assert len(result['unconfirmed_xyz']) == (4 if i<3 else 0)
    with client.websocket_connect(f"/ws/playback?bag_id=source/bag&topic=/lidar_points&detection_id={job['id']}") as ws:
        frame = ws.receive_json()
        cloud = ws.receive_bytes()
        assert frame['type'] == 'frame' and not frame['detected']
        assert len(frame['unconfirmed_xyz']) == 4
        assert len(cloud) == frame['points']*12
