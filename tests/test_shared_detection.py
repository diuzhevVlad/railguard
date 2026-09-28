import unittest
import numpy as np
from railpath import obstacle_filter as core
from obstacle_app import detection, obstacle_filter
from obstacle_app.learned_detection import adapt_prediction


class SharedDetectionTests(unittest.TestCase):
    def test_app_uses_the_same_filter_and_types(self):
        self.assertIs(detection.detect_obstacle_points, core.detect_obstacle_points)
        self.assertIs(detection.RailPrediction, core.RailPrediction)
        self.assertIs(obstacle_filter.ObstacleFilterConfig, core.ObstacleFilterConfig)

    def test_raw_model_and_web_prediction_paths_match(self):
        d = np.arange(1, 201, dtype=np.float32)
        center = np.column_stack((.0005*d*d, -d, d*0))
        xyz = np.array([[.0005*y*y + i*.01, -y, z]
                        for y, z in [(2, .15), (3, .3), (10, .11), (20, .3), (160, .5)]
                        for i in range(5)], np.float32)
        rails = xyz[15:20]
        out = dict(centerline=center, horizon=100, rails=rails, sigma=np.ones(200))
        app = detection.detect_obstacle_points(xyz, adapt_prediction(out))
        ros = core.detect_obstacles(xyz, center, rails)
        np.testing.assert_array_equal(app.points, ros.points)
        np.testing.assert_array_equal(app.rejected_points, ros.rejected_points)
        self.assertEqual(app.distance_m, ros.distance_m)
        self.assertEqual(app.excluded_rail_count, ros.excluded_rail_count)
