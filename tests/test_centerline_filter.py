import unittest

import numpy as np

from railpath.centerline_filter import (BayesianCenterlineFilter,
                                        CenterlineFilterConfig)
from railpath.obstacle_filter import ObstacleFilterConfig, RailPrediction
from railpath.occupancy_filter import ObstacleDetectionPipeline


def prediction(x: float = 0.0, y=(-10.0, -20.0)) -> RailPrediction:
    center = np.array([[x, value, 0.0] for value in y], dtype=np.float32)
    distance = np.array([10.0, 20.0], dtype=np.float32)
    return RailPrediction(center, distance, details={"status": "ready"},
                          max_forward_m=30.0)


class CenterlineFilterTests(unittest.TestCase):
    def setUp(self):
        self.config = CenterlineFilterConfig(
            initial_variance_m2=.25, process_variance_m2=.04,
            measurement_variance_m2=.09,
            measurement_variance_growth_m2_per_m=0,
            max_measurement_variance_m2=.09, use_model_uncertainty=False,
            max_correspondence_distance_m=2.0)

    def test_kalman_update_fuses_prior_and_measurement(self):
        tracker = BayesianCenterlineFilter(self.config)
        first = tracker.update(prediction())
        second = tracker.update(prediction(1.0))
        np.testing.assert_allclose(first.prediction.center[:, 0], 0)
        self.assertTrue(np.all(second.prediction.center[:, 0] > 0))
        self.assertTrue(np.all(second.prediction.center[:, 0] < 1))
        expected_gain = (.25 + .04) / (.25 + .04 + .09)
        np.testing.assert_allclose(second.prediction.center[:, 0], expected_gain,
                                   rtol=1e-6)
        self.assertTrue(np.all(second.variances_m2 < self.config.measurement_variance_m2))

    def test_odometry_predicts_previous_line_into_current_lidar_frame(self):
        current = prediction(1.0, (-5.0, -15.0))
        pose = np.eye(4)
        pose[1, 3] = -5.0

        with_odometry = BayesianCenterlineFilter(self.config)
        with_odometry.update(prediction(), np.eye(4))
        fused = with_odometry.update(current, pose, odometry_used=True)

        without_odometry = BayesianCenterlineFilter(self.config)
        without_odometry.update(prediction(), np.eye(4))
        unmatched = without_odometry.update(current, np.eye(4))
        self.assertTrue(np.all(fused.prediction.center[:, 0] < 1))
        np.testing.assert_allclose(unmatched.prediction.center[:, 0], 1)
        self.assertTrue(fused.odometry_used)

    def test_distant_uncertain_samples_are_smoothed_more(self):
        config = CenterlineFilterConfig(
            initial_variance_m2=.25, process_variance_m2=.02,
            measurement_variance_m2=.09,
            measurement_variance_growth_m2_per_m=0,
            max_measurement_variance_m2=1.0,
            use_model_uncertainty=True)
        sigma = [.3] * 10 + [1.0] * 10
        first = prediction()
        second = prediction(1.0)
        first = RailPrediction(
            first.center, first.distance_along_m, details={"sigma_m": sigma})
        second = RailPrediction(
            second.center, second.distance_along_m, details={"sigma_m": sigma})
        tracker = BayesianCenterlineFilter(config)
        tracker.update(first)
        result = tracker.update(second)
        self.assertLess(result.prediction.center[1, 0],
                        result.prediction.center[0, 0])

    def test_association_gate_fades_without_spatial_jump(self):
        config = CenterlineFilterConfig(
            initial_variance_m2=.25, process_variance_m2=.04,
            measurement_variance_m2=.09,
            measurement_variance_growth_m2_per_m=0,
            max_measurement_variance_m2=.09,
            use_model_uncertainty=False,
            max_correspondence_distance_m=2.0)
        distance = np.arange(1., 11., dtype=np.float32)
        y = -distance
        prior = RailPrediction(
            np.column_stack((np.zeros(10), y, np.zeros(10))).astype(np.float32),
            distance)
        offset = np.arange(10, dtype=np.float32) * .25
        measured = RailPrediction(
            np.column_stack((offset, y, np.zeros(10))).astype(np.float32),
            distance)
        tracker = BayesianCenterlineFilter(config)
        tracker.update(prior)
        result = tracker.update(measured).prediction.center[:, 0]
        self.assertLess(np.max(np.diff(result)), .45)
        self.assertAlmostEqual(result[-1], offset[-1], delta=.01)


    def test_pipeline_is_optional_and_exposes_filtered_result(self):
        points = np.empty((0, 3), dtype=np.float32)
        spatial = ObstacleFilterConfig(min_cluster_points=1)
        disabled = ObstacleDetectionPipeline(spatial).update(
            1, points, prediction())
        enabled = ObstacleDetectionPipeline(
            spatial, centerline_config=self.config,
            centerline_filter_enabled=True).update(1, points, prediction())
        self.assertIsNone(disabled.centerline_estimate)
        self.assertFalse(disabled.centerline_filtered)
        self.assertIsNotNone(enabled.centerline_estimate)
        self.assertTrue(enabled.centerline_filtered)

    def test_configuration_validation(self):
        for values in ({"initial_variance_m2": 0},
                       {"process_variance_m2": -1},
                       {"measurement_variance_m2": np.inf},
                       {"measurement_variance_growth_m2_per_m": -1},
                       {"max_measurement_variance_m2": .01},
                       {"use_model_uncertainty": 1},
                       {"max_correspondence_distance_m": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CenterlineFilterConfig(**values)


if __name__ == "__main__":
    unittest.main()
