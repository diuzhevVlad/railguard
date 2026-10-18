import unittest
from dataclasses import replace
from pathlib import Path

from railpath.detection_config import (ObstacleDetectionConfig,
                                       load_obstacle_detection_config)


CONFIG = Path(__file__).resolve().parents[1] / "railguard/config/railguard.yaml"


class DetectionConfigTests(unittest.TestCase):
    def test_repository_config_loads_spatial_and_bayesian_settings(self):
        config = load_obstacle_detection_config(CONFIG)
        self.assertEqual(config.obstacle.distance_knots_m, (0.0, 30.0, 80.0, 150.0))
        self.assertEqual(config.bayesian.prior_probability, 0.08)
        self.assertEqual(config.bayesian.point_hit_probability, 0.4)
        self.assertEqual(config.bayesian.miss_probability, 0.07)
        self.assertEqual(config.bayesian.detection_probability, 0.90)
        self.assertEqual(config.bayesian.prediction_decay, 0.95)
        self.assertEqual(config.bayesian.prediction_diffusion, 0.05)
        self.assertEqual(config.bayesian.neighbor_hit_weight, 0.25)
        self.assertEqual(config.bayesian.max_scan_hit_probability, 0.85)
        self.assertEqual(config.bayesian.immediate_close_distance_m, 15.0)
        self.assertFalse(config.centerline_filter_enabled)
        self.assertEqual(config.centerline.initial_variance_m2, 0.25)
        self.assertEqual(config.centerline.process_variance_m2, 0.01)
        self.assertEqual(config.centerline.measurement_variance_m2, 0.04)
        self.assertEqual(config.centerline.measurement_variance_growth_m2_per_m, 0.0)
        self.assertEqual(config.centerline.max_measurement_variance_m2, 0.50)
        self.assertTrue(config.centerline.use_model_uncertainty)
        self.assertEqual(config.centerline.max_correspondence_distance_m, 2.0)
        self.assertEqual(config.centerline.association_fade_start_ratio, 0.5)
        self.assertEqual(config.centerline.nurbs_measurement_spacing_m, 2.0)
        self.assertEqual(config.centerline.nurbs_behind_spans, 3)
        self.assertEqual(config.centerline.nurbs_max_distance_m, 20.0)
        self.assertEqual(config.centerline.nurbs_gate, 15.0)
        self.assertTrue(config.centerline.nurbs_path_measurement_enabled)
        self.assertEqual(config.centerline.nurbs_output_strength, 0.2)
        self.assertEqual(config.bayesian.immediate_near_distance_m, 50.0)
        self.assertEqual(config.bayesian.immediate_near_probability, 0.29)
        self.assertEqual(config.bayesian.immediate_far_distance_m, 99.0)

    def test_flat_parameter_round_trip(self):
        expected = ObstacleDetectionConfig(
            centerline_filter_enabled=True,
            max_forward_m=90.0,
            bayesian_filter_enabled=True,
            odometry_enabled=True,
        )
        actual = ObstacleDetectionConfig.from_parameters(expected.to_parameters())
        self.assertEqual(actual, expected)

    def test_maximum_must_cover_minimum_distance(self):
        obstacle = replace(ObstacleDetectionConfig().obstacle, min_forward_m=20.0)
        with self.assertRaises(ValueError):
            ObstacleDetectionConfig(obstacle=obstacle, max_forward_m=10.0)


if __name__ == "__main__":
    unittest.main()
