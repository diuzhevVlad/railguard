import tempfile
import time
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from fastapi.testclient import TestClient
from obstacle_app.detection import RailPrediction, detect_obstacle_points
from obstacle_app.obstacle_filter import ObstacleFilterConfig
from obstacle_app.scan_detection import ScanDetector
from obstacle_app.web import create_app
from tests.test_obstacle_app import make_bag, MODEL, fake_scan_detector
from tests.test_world_map import build_test_map
from tests.test_ray_rendering import cloud


class AppParametersTests(unittest.TestCase):
    def test_dimensions_and_range_change_actual_classification(self):
        depth=np.arange(1,201,.1,dtype=np.float32)
        prediction=RailPrediction(np.column_stack((depth*0,-depth,depth*0)),depth,
                                  details={'status':'ready'},max_forward_m=150)
        points=np.array([[1.4+i*.01,-20,1] for i in range(5)]+
                        [[i*.01,-30,3.5] for i in range(5)]+
                        [[i*.01,-175,1] for i in range(5)])
        self.assertFalse(detect_obstacle_points(points,prediction).detected)
        cfg=ObstacleFilterConfig(train_width_m=4,train_height_m=4)
        self.assertEqual(detect_obstacle_points(points,prediction,cfg).point_count,10)
        class Provider:
            def predict(self,*args):return prediction
        detector=ScanDetector(Provider(),filter_config=cfg,max_forward_m=200)
        pred,result=detector.detect(1,cloud(points))
        self.assertEqual(result.point_count,15)
        self.assertEqual(pred.details['train_width_m'],4)
        self.assertEqual(pred.details['lookahead_limit_m'],200)

    def test_api_settings_validation_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';out=root/'out';data.mkdir();out.mkdir()
            make_bag(data/'tiny',rails=True)
            client=TestClient(create_app(data_root=data,output_root=out,model_path=MODEL,
                detection_map_builder=build_test_map,scan_detector_factory=fake_scan_detector))
            request=dict(bag_id='source/tiny',topic='/lidar_points')
            def await_job(path,job):
                for _ in range(500):
                    current=client.get(path+job['id']).json()
                    if current['state'] in ('done','failed'):break
                    time.sleep(.01)
                self.assertEqual(current['state'],'done',current)
                return current
            first=client.post('/api/detections',json=request).json()
            await_job('/api/detections/',first)
            changed={**request,'train_width_m':4,'train_height_m':4,'max_forward_m':200}
            second=client.post('/api/detections',json=changed).json()
            self.assertNotEqual(first['id'],second['id'])
            self.assertEqual(second['id'],client.post('/api/detections',json=changed).json()['id'])
            await_job('/api/detections/',second)
            for param,value in [('train_width_m',0),('train_height_m',-1),('max_forward_m',201),('initial_speed_m_s',-1)]:
                self.assertEqual(client.post('/api/detections',json={**request,param:value}).status_code,422)
            a=await_job('/api/maps/jobs/',client.post('/api/maps',json={**request,'initial_speed_m_s':3}).json())
            b=await_job('/api/maps/jobs/',client.post('/api/maps',json={**request,'initial_speed_m_s':5}).json())
            self.assertNotEqual(a['id'],b['id']);self.assertNotEqual(a['map_id'],b['map_id'])
            response=client.post('/api/jobs',json={**request,'map_id':a['map_id'],'initial_speed_m_s':5,
                'world_obstacle':dict(model_id='person_standing',position=[0,-20,-1.1])})
            self.assertEqual(response.status_code,400)
            self.assertIn('speed changed',response.json()['detail'])
            client.close()
