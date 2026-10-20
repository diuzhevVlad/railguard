"""Run inside the web image: python -m unittest discover -s tests -p test_model_range.py."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from pydantic import ValidationError

from obstacle_app.learned_detection import adapt_prediction
from obstacle_app.scan_detection import ScanDetector
from obstacle_app.web import DetectionRequest
from railpath.obstacle_filter import ObstacleFilterConfig


class ModelRangeTests(unittest.TestCase):
    def test_400m_curve_and_sigma_survive_adapter_and_detection(self):
        d=np.arange(1,401,dtype=np.float32)
        out=dict(centerline=np.column_stack([d*0,-d,d*0]),sigma=np.full(400,.3),horizon=80)
        prediction=adapt_prediction(out)
        self.assertEqual(prediction.center[-1,1],-400)
        self.assertEqual(prediction.continuation[-1,1],-400)
        self.assertEqual(len(prediction.details['sigma_m']),400)
        self.assertEqual(prediction.details['model_range_m'],400)
        xyz=np.array([[0,-250,.5],[0,-399,.5],[0,-401,.5]],np.float32)
        fields=[SimpleNamespace(name=k,offset=4*i,datatype=7,count=1) for i,k in enumerate(('x','y','z'))]
        msg=SimpleNamespace(width=3,height=1,point_step=12,row_step=36,is_bigendian=False,fields=fields,data=xyz.tobytes())
        provider=SimpleNamespace(predict=lambda *args:prediction)
        config=ObstacleFilterConfig(min_cluster_points=1,stabilize_envelope=False)
        detector=ScanDetector(provider,filter_config=config,max_forward_m=400)
        pred,result=detector.detect(1,msg)
        np.testing.assert_array_equal(result.points,xyz[:2])
        self.assertEqual(pred.details['checked_forward_m'],400)
        detector=ScanDetector(provider,filter_config=config,max_forward_m=200)
        _,result=detector.detect(1,msg)
        self.assertEqual(result.point_count,0)

    def test_api_and_slider_support_400_and_reject_beyond_it(self):
        request=DetectionRequest(bag_id='source/test',topic='/lidar_points',max_forward_m=400)
        self.assertEqual(request.max_forward_m,400)
        with self.assertRaises(ValidationError):
            DetectionRequest(bag_id='source/test',topic='/lidar_points',max_forward_m=401)
        html=(Path(__file__).resolve().parents[1]/'obstacle_app/static/index.html').read_text()
        self.assertIn('id="detection-distance" type="range" min="5" max="400"',html)


if __name__=='__main__':unittest.main()
