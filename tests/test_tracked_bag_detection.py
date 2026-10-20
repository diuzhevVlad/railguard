from dataclasses import replace
from pathlib import Path
import time
import pytest

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from obstacle_app.cloud import valid_xyz
from obstacle_app.scan_detection import ScanDetector
from obstacle_app.tracked_bag_detection import TrackedBagDetector, BagOdometry
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
    def __init__(self, speed=0., **settings):
        self.calls = []
        self.previous = None
    def reset(self):
        self.previous = None
    def update(self, xyz, stamp, point_timestamps=None):
        reset = self.previous is None or not 0 < stamp-self.previous <= 2e9
        self.previous = stamp
        self.calls.append(stamp)
        pose = np.eye(4)
        pose[1,3] = -(stamp-1_000_000_000)/100_000_000
        return pose, reset


def raw_detector():
    class Provider:
        def predict(self, stamp, msg):
            return prediction()
    return ScanDetector(Provider(), filter_config=CANDIDATE_CONFIG, candidate_detector=detect_candidates)


def test_incremental_rewind_complete_cache_and_no_skipped_odometry(tmp_path):
    bag=tmp_path/'bag'
    frames=make_bag(bag)
    stamps=poses()[0]
    odom=BagOdometry(bag,'/lidar_points',3.,tmp_path/'cache',Odom)
    detector=TrackedBagDetector(raw_detector(),odom)
    assert odom.status()['processed_scans']==0
    assert not odom.odometry.calls
    with pytest.raises(ValueError,match='Forward seeking'):
        detector.detect(int(stamps[7]),frames[7])
    with pytest.raises(ValueError,match='Forward seeking'):
        detector.validate_start(.1)
    for i in range(4):
        result=detector.detect(int(stamps[i]),frames[i])
        assert result[1].detected==(i>=3)
    assert len(odom.odometry.calls)==4
    assert not detector.detect(int(stamps[0]),frames[0])[1].detected
    assert detector.detect(int(stamps[3]),frames[3])[1].detected
    # Another job/browser can replay a processed scan without changing ICP.
    other=TrackedBagDetector(raw_detector(),odom)
    assert other.detect(int(stamps[3]),frames[3])[1].detected
    assert len(odom.odometry.calls)==4
    with pytest.raises(ValueError,match='Forward seeking'):
        detector.detect(int(stamps[5]),frames[5])
    for i in range(4,8):
        assert detector.detect(int(stamps[i]),frames[i])[1].detected
    assert odom.status()['complete'] and odom.path.exists()
    assert len(odom.odometry.calls)==8
    odom.close()
    cached=BagOdometry(bag,'/lidar_points',3.,tmp_path/'cache',Odom)
    assert cached.status()['complete'] and not cached.odometry.calls
    cached.validate_start(.7)
    assert TrackedBagDetector(raw_detector(),cached).detect(int(stamps[7]),frames[7])[1].detected
    different=BagOdometry(bag,'/lidar_points',5.,tmp_path/'cache',Odom)
    assert not different.status()['complete']
    assert cached.path!=different.path
    cached.close();different.close()


def test_partial_cache_does_not_resume_without_icp_state(tmp_path):
    frames=make_bag(tmp_path/'bag')
    args=(tmp_path/'bag','/lidar_points',3.,tmp_path/'cache',Odom)
    odom=BagOdometry(*args)
    odom.ensure(int(poses()[0][0]),frames[0])
    assert not odom.path.exists()
    odom.close()
    restarted=BagOdometry(*args)
    assert restarted.status()['processed_scans']==0
    restarted.close()


def test_web_seeking_and_normal_viewer_are_independent(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from obstacle_app import model_worker,tracked_bag_detection
    from obstacle_app.web import create_app
    data,output=tmp_path/'data',tmp_path/'output'
    data.mkdir();output.mkdir();make_bag(data/'bag')
    make_bag(data/'cloud_with_fake_obj')
    class Runtime:
        device='test'
        range_m=200
        def __init__(self,*args):pass
        def load(self):pass
        def close(self):pass
        def predict(self,msg):return prediction()
    monkeypatch.setattr(model_worker,'ProcessRailRuntime',Runtime)
    monkeypatch.setattr(tracked_bag_detection,'TrackingOdometry',Odom)
    with TestClient(create_app(data_root=data,output_root=output)) as client:
        info=client.get('/api/info',params={'bag_id':'source/cloud_with_fake_obj'}).json()
        assert info['detection_initial_speed_m_s']==0 and info['detection_speed_fixed']
        assert info['initial_speed_m_s']==3  # Simulation settings remain separate.
        info=client.get('/api/info',params={'bag_id':'source/bag'}).json()
        assert not info['detection_speed_fixed']
        assert info['detection_initial_speed_m_s'] is None
        request={'bag_id':'source/bag','topic':'/lidar_points'}
        for extra in ({}, {'initial_speed_m_s':None}):
            response=client.post('/api/detections',json={**request,**extra})
            assert response.status_code==400
            assert 'Set the detection initial speed' in response.json()['detail']
        job=client.post('/api/detections',json={**request,'initial_speed_m_s':0}).json()
        for _ in range(100):
            job=client.get('/api/detections/'+job['id']).json()
            if job['state'] in ('done','failed'):break
            time.sleep(.01)
        assert job['state']=='done',job
        assert job['odometry']['processed_scans']==0
        base='/ws/playback?bag_id=source/bag&topic=/lidar_points'
        detected=base+'&detection_id='+job['id']
        with client.websocket_connect(detected+'&start=0.7') as ws:
            assert ws.receive_json()['type']=='error'
        # Detection off always permits arbitrary seeks and does not advance ICP.
        with client.websocket_connect(base+'&start=0.7') as ws:
            frame=ws.receive_json();ws.receive_bytes()
            assert frame['odometry'] is None
        assert client.get('/api/detections/'+job['id']).json()['odometry']['processed_scans']==0
        assert client.get(f"/api/detections/{job['id']}/frame",params={'timestamp_ns':int(poses()[0][7])}).status_code==400
        with client.websocket_connect(detected) as ws:
            for i in range(8):
                frame=ws.receive_json();ws.receive_bytes()
                assert frame['detected']==(i>=3)
                assert frame['odometry']['processed_scans']==i+1
            assert frame['odometry']['complete']
        with client.websocket_connect(detected+'&start=0.7') as ws:
            frame=ws.receive_json();ws.receive_bytes()
            assert frame['detected'] and frame['odometry']['complete']
        preview=client.get(f"/api/detections/{job['id']}/frame",params={'timestamp_ns':int(poses()[0][7])}).json()
        assert preview['detected'] and preview['tracks']==frame['tracks']
