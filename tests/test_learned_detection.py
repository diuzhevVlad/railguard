import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from obstacle_app.learned_detection import (scan_input, adapt_prediction, RailGuardRuntime,
    LearnedDetectionManager, LearnedRailProvider, prediction_payload)
from obstacle_app.scan_detection import ScanDetector
from obstacle_app.obstacle_filter import ObstacleFilterConfig
from obstacle_app.web import create_app
from tests.test_obstacle_app import MODEL, make_bag
from tests.test_ray_rendering import cloud


def output(horizon=50):
    d=np.arange(1,201,dtype=np.float32)
    return dict(centerline=np.column_stack((d*.1,-d,np.full(200,-1.1))),
                horizon=horizon,sigma=np.linspace(.2,2,200))


class LearnedDetectionTests(unittest.TestCase):
    def test_full_curve_distance_and_uncertainty(self):
        result=adapt_prediction(output(),12,'cuda')
        self.assertAlmostEqual(result.distance_along_m[-1],200*np.sqrt(1.01),places=3)
        self.assertEqual(result.center[-1,1],-200)
        self.assertGreater(len(result.center),100)
        self.assertEqual(result.continuation[-1,1],-200)
        self.assertEqual(result.details['inference_ms'],12)
        self.assertEqual(result.details['status'],'ready')
        limited=adapt_prediction(output(180))
        self.assertEqual(limited.center[-1,1],-200)
        self.assertEqual(limited.details['horizon_m'],180)

    def test_invalid_output_does_not_bridge_gaps_or_claim_clear_track(self):
        result=output();result['centerline'][20,0]=np.nan
        prediction=adapt_prediction(result)
        self.assertEqual(prediction.center[-1,1],-20)
        result['centerline'][:]=np.nan
        prediction=adapt_prediction(result)
        self.assertEqual(len(prediction.center),0)
        self.assertEqual(prediction.details['status'],'unavailable')
        self.assertEqual(adapt_prediction(output(0)).center[-1,1],-200)

    def test_detection_includes_150m_but_excludes_points_beyond_it(self):
        class Provider:
            def predict(self, stamp, msg):
                return adapt_prediction(output(50))
        msg = cloud([[8,-80,0], [15,-150,0], [15.001,-150.01,0], [19.9,-199,0],
                     [20,-150,0], [15,-150,-2]])
        prediction, detection = ScanDetector(Provider(), filter_config=ObstacleFilterConfig(min_cluster_points=1)).detect(1, msg)
        self.assertEqual(prediction.details['checked_forward_m'], 150)
        self.assertAlmostEqual(prediction.details['sigma_max_checked_m'], output()['sigma'][149])
        np.testing.assert_allclose(detection.points, [[8,-80,0], [15,-150,0]])

    def test_segmented_rails_are_preserved_in_playback_payload(self):
        result = output()
        result['rails'] = np.array([[1,-8,-1], [-1,-9,-1], [np.nan,0,0]])
        prediction = adapt_prediction(result)
        self.assertEqual(prediction_payload(prediction)['rail_points'],
                         [[1,-8,-1], [-1,-9,-1]])
        self.assertEqual(prediction_payload(adapt_prediction(output()))['rail_points'], [])

    def test_xyz_and_ring_remain_aligned(self):
        msg=cloud([[0,-2,0],[np.nan,-3,0],[0,0,0],[1,-4,.5]],[3,4,5,6])
        before=bytes(msg.data)
        xyz,ring=scan_input(msg)
        np.testing.assert_array_equal(ring,[3,6])
        np.testing.assert_array_equal(xyz,[[0,-2,0],[1,-4,.5]])
        self.assertEqual(bytes(msg.data),before)
        msg.fields=[f for f in msg.fields if f.name!='ring']
        self.assertIsNone(scan_input(msg)[1])

    def test_runtime_does_not_reuse_sparse_ring_state_after_seek(self):
        runtime=RailGuardRuntime(device='cpu')
        class Model:
            sparse='previous bag'
            def predict(self,xyz,ring):
                assert self.sparse is None
                self.sparse='current scan'
                return output()
        runtime.model=Model()
        for _ in range(2):self.assertEqual(runtime.predict(cloud([[0,-10,0]])).details['status'],'ready')

    def test_scan_cache_is_bounded_and_receives_current_points(self):
        class Provider:
            calls=0
            def predict(self,stamp,msg):
                self.calls+=1
                assert msg.width==1
                return adapt_prediction(output())
        provider=Provider();detector=ScanDetector(provider,cache_frames=2)
        msg=cloud([[0,-10,0]])
        detector.detect(1,msg);detector.detect(1,msg)
        self.assertEqual(provider.calls,1)
        detector.detect(2,msg);detector.detect(3,msg)
        self.assertIsNone(detector.cached(1))
        self.assertEqual(len(detector.cache),2)

    def test_sessions_do_not_share_cached_frames_between_bags(self):
        from tests.test_obstacle_app import fake_scan_detector
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['a','b']:make_bag(root/name)
            manager=LearnedDetectionManager(fake_scan_detector)
            first=manager.start('source/a',root/'a','/lidar_points')
            second=manager.start('source/b',root/'b','/lidar_points')
            for _ in range(100):
                if first.state=='done' and second.state=='done':break
                time.sleep(.01)
            self.assertEqual((first.state,second.state),('done','done'))
            detector=manager.detector(first)
            detector.detect(100,cloud([[0,-10,0]]))
            self.assertIsNone(manager.detector(second).cached(100))
            self.assertEqual(manager.start('source/a',root/'a','/lidar_points').id,first.id)
            metadata=root/'a'/'metadata.yaml';metadata.write_text(metadata.read_text()+'\n')
            self.assertNotEqual(manager.start('source/a',root/'a','/lidar_points').id,first.id)

    def test_missing_weights_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Missing RailGuard'):
                RailGuardRuntime(tmp).load()

    def test_inference_error_keeps_playback_alive(self):
        class Provider:
            def predict(self,stamp,msg):raise RuntimeError('test inference failure')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir();make_bag(data/'tiny')
            client=TestClient(create_app(data_root=data,output_root=root/'out',model_path=MODEL,
                scan_detector_factory=lambda *args:ScanDetector(Provider())))
            job=client.post('/api/detections',json=dict(bag_id='source/tiny',topic='/lidar_points')).json()
            for _ in range(100):
                state=client.get('/api/detections/'+job['id']).json()
                if state['state']=='done':break
                time.sleep(.01)
            self.assertEqual(state['state'],'done')
            with client.websocket_connect('/ws/playback?bag_id=source/tiny&topic=/lidar_points&paused=1&detection_id='+job['id']) as ws:
                ws.send_json({'action':'step'})
                self.assertEqual(ws.receive_json()['type'],'overlay_error')
                frame=ws.receive_json();self.assertEqual(frame['type'],'frame')
                self.assertEqual(frame['rail_prediction']['status'],'error')
                ws.receive_bytes()
                ws.send_json({'action':'hide_detection'})
                self.assertIsNone(ws.receive_json()['rail_prediction']);ws.receive_bytes()
            self.assertFalse((root/'out'/'.detection_maps').exists())
            client.close()


if __name__=='__main__':unittest.main()
