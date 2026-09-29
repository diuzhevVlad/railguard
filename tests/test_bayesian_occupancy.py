import unittest

import numpy as np

from railpath.obstacle_filter import (ObstacleFilterConfig, RailPrediction,
                                      make_rail_prediction)
from railpath.occupancy_filter import (BayesianFilterConfig,
                                       BayesianObstacleFilter,
                                       ObstacleDetectionPipeline)


def straight_prediction(maximum=40.0):
    distance = np.arange(0, maximum + .25, .25, dtype=np.float32)
    center = np.column_stack((np.zeros_like(distance), -distance,
                              np.zeros_like(distance)))
    return RailPrediction(center, distance, max_forward_m=maximum)


def obstacle(distance, count):
    return np.array([[.01 * index, -distance, .4]
                     for index in range(count)], dtype=np.float32)


class BayesianOccupancyTests(unittest.TestCase):
    def test_more_points_give_more_occupancy_evidence(self):
        config = BayesianFilterConfig(prior_probability=.5,
                                      point_hit_probability=.55,
                                      miss_probability=.4,detection_probability=.65,
                                      max_scan_hit_probability=.62)
        one = BayesianObstacleFilter(config).update(
            straight_prediction(), obstacle(20, 1), min_distance_m=1.5)
        tracker = BayesianObstacleFilter(config)
        five = tracker.update(
            straight_prediction(), obstacle(20, 5), min_distance_m=1.5)
        repeated = tracker.update(
            straight_prediction(), obstacle(20, 5), min_distance_m=1.5)
        index = int(np.flatnonzero(one.distance_along_m == 20)[0])
        self.assertAlmostEqual(float(one.probabilities[index]), .55, places=5)
        self.assertGreater(float(five.probabilities[index]),
                           float(one.probabilities[index]))
        self.assertFalse(one.occupied[index])
        self.assertFalse(five.occupied[index])
        self.assertTrue(repeated.occupied[index])
        self.assertEqual(five.point_counts[index], 5)

    def test_hit_evidence_updates_adjacent_cells_with_lower_weight(self):
        config = BayesianFilterConfig(prior_probability=.4,
                                      point_hit_probability=.6,
                                      miss_probability=.2,
                                      detection_probability=.95,
                                      prediction_diffusion=0,
                                      neighbor_hit_weight=.25,
                                      max_scan_hit_probability=.9)
        result = BayesianObstacleFilter(config).update(
            straight_prediction(), obstacle(20, 1), min_distance_m=1.5)
        index = int(np.flatnonzero(result.distance_along_m == 20)[0])
        neighbor = index + 1
        far = index + 3
        self.assertAlmostEqual(float(result.probabilities[index]), .6, places=5)
        self.assertGreater(result.probabilities[index], result.probabilities[neighbor])
        self.assertGreater(result.probabilities[neighbor], result.probabilities[far])
        self.assertEqual(result.point_counts[neighbor], 0)

    def test_prediction_diffuses_history_to_adjacent_cells(self):
        config = BayesianFilterConfig(prior_probability=.4,
                                      point_hit_probability=.6,
                                      miss_probability=.39,
                                      detection_probability=.95,
                                      prediction_diffusion=.2,
                                      neighbor_hit_weight=0,
                                      max_scan_hit_probability=.9)
        tracker = BayesianObstacleFilter(config)
        tracker.update(straight_prediction(), obstacle(20, 5), min_distance_m=1.5)
        result = tracker.update(
            straight_prediction(), np.empty((0, 3)), min_distance_m=1.5)
        center = int(np.flatnonzero(result.distance_along_m == 20)[0])
        adjacent = center + 1
        far = center + 3
        self.assertGreater(result.probabilities[adjacent], result.probabilities[far])
        self.assertGreater(result.probabilities[center], result.probabilities[adjacent])

    def test_far_obstacle_requires_repeated_scan_evidence(self):
        config = BayesianFilterConfig(prior_probability=.08,
                                      point_hit_probability=.4,
                                      miss_probability=.03,
                                      detection_probability=.95,
                                      prediction_decay=.8,
                                      prediction_diffusion=.1,
                                      neighbor_hit_weight=.25,
                                      max_scan_hit_probability=.9,
                                      max_point_evidence=7)
        tracker = BayesianObstacleFilter(config)
        first = tracker.update(
            straight_prediction(150), obstacle(120, 5), min_distance_m=1.5)
        second = tracker.update(
            straight_prediction(150), obstacle(120, 5), min_distance_m=1.5)
        index = int(np.flatnonzero(first.distance_along_m == 120)[0])
        self.assertFalse(first.occupied[index])
        self.assertTrue(second.occupied[index])
        self.assertAlmostEqual(second.distance_m, 120)

    def test_odometry_predicts_stationary_obstacle_in_current_lidar_frame(self):
        config = BayesianFilterConfig(point_hit_probability=.6,
                                      miss_probability=.35,
                                      detection_probability=.7,
                                      prediction_diffusion=0,
                                      neighbor_hit_weight=0,
                                      max_scan_hit_probability=.9)
        tracker = BayesianObstacleFilter(config)
        first = tracker.update(straight_prediction(), obstacle(20, 10),
                               np.eye(4), min_distance_m=1.5,
                               odometry_used=True)
        self.assertTrue(first.occupied[first.distance_along_m == 20].item())
        current_pose = np.eye(4)
        current_pose[1, 3] = -5
        second = tracker.update(straight_prediction(), np.empty((0, 3)),
                                current_pose, min_distance_m=1.5,
                                odometry_used=True)
        shifted = int(np.flatnonzero(second.distance_along_m == 15)[0])
        old = int(np.flatnonzero(second.distance_along_m == 20)[0])
        self.assertTrue(second.occupied[shifted])
        self.assertGreater(second.probabilities[shifted],
                           second.probabilities[old])
        self.assertAlmostEqual(second.distance_m, 15)

    def test_pipeline_is_optional_and_filters_current_points_when_enabled(self):
        prediction = straight_prediction()
        points = obstacle(20, 5)
        spatial = ObstacleFilterConfig(min_cluster_points=1)
        raw = ObstacleDetectionPipeline(spatial).update(1, points, prediction)
        self.assertTrue(raw.detected)
        self.assertIsNone(raw.occupancy)
        self.assertFalse(raw.bayesian_filtered)

        strict = BayesianFilterConfig(point_hit_probability=.51,
                                      detection_probability=.9)
        filtered = ObstacleDetectionPipeline(
            spatial, strict, bayesian_filter_enabled=True).update(
                1, points, prediction)
        self.assertFalse(filtered.detected)
        self.assertEqual(filtered.point_count, 0)
        self.assertEqual(len(filtered.rejected_points), len(points))
        self.assertIsNotNone(filtered.occupancy)

    def test_confirmed_occupancy_without_a_current_hit_does_not_alert(self):
        prediction = straight_prediction()
        points = obstacle(20, 5)
        spatial = ObstacleFilterConfig(min_cluster_points=1)
        config = BayesianFilterConfig(prior_probability=.08,
                                      point_hit_probability=.4,
                                      miss_probability=.07,
                                      detection_probability=.9,
                                      prediction_decay=.95,
                                      max_scan_hit_probability=.85)
        pipeline = ObstacleDetectionPipeline(
            spatial, config, bayesian_filter_enabled=True)
        self.assertFalse(pipeline.update(1, points, prediction).detected)
        self.assertTrue(pipeline.update(2, points, prediction).detected)
        stale = pipeline.update(3, np.empty((0, 3)), prediction)
        self.assertTrue(stale.occupancy.detected)
        self.assertFalse(stale.detected)
        self.assertIsNone(stale.distance_m)

    def test_immediate_policy_preserves_strong_near_and_sparse_far_hits(self):
        config = BayesianFilterConfig(prior_probability=.08,
                                      point_hit_probability=.4,
                                      miss_probability=.07,
                                      detection_probability=.9,
                                      prediction_decay=.95,
                                      max_scan_hit_probability=.85,
                                      immediate_close_distance_m=15,
                                      immediate_near_distance_m=50,
                                      immediate_near_probability=.75,
                                      immediate_far_distance_m=110)
        spatial = ObstacleFilterConfig(min_cluster_points=1)
        prediction = straight_prediction(150)

        def detected(distance):
            pipeline = ObstacleDetectionPipeline(
                spatial, config, bayesian_filter_enabled=True)
            return pipeline.update(1, obstacle(distance, 5), prediction).detected

        self.assertTrue(detected(40))
        self.assertFalse(detected(80))
        self.assertTrue(detected(120))

    def test_raw_centerline_adapter_matches_legacy_geometry(self):
        curve = np.column_stack((np.zeros(41), -np.arange(41),
                                 np.zeros(41))).astype(np.float32)
        curve[30:] = np.nan
        rails = np.array([[0, -5, 0], [np.nan, 0, 0]], np.float32)
        prediction = make_rail_prediction(curve, rails, 25)
        self.assertAlmostEqual(float(prediction.distance_along_m[-1]), 29)
        self.assertEqual(prediction.rail_points.shape, (1, 3))
        self.assertEqual(prediction.max_forward_m, 25)

    def test_kiss_adapter_subsamples_and_returns_engine_pose(self):
        class Engine:
            def __init__(self):
                self.last_pose = np.eye(4)
                self.last_pose[1, 3] = -3
                self.frames = []
            def register_frame(self, points, timestamps):
                self.frames.append((points.copy(), timestamps.copy()))

        engine = Engine()
        from railpath.occupancy_filter import KissICPOdometry
        odometry = KissICPOdometry(scan_stride=2, engine_factory=lambda: engine)
        points = np.arange(15, dtype=np.float64).reshape(5, 3)
        pose = odometry.update(points)
        np.testing.assert_array_equal(engine.frames[0][0], points[::2])
        self.assertEqual(engine.frames[0][1].shape, (0,))
        self.assertAlmostEqual(pose[1, 3], -3)

    def test_configuration_validation(self):
        prior = BayesianFilterConfig().prior_probability
        for kwargs in ({'grid_resolution_m': 0},
                       {'point_hit_probability': prior},
                       {'miss_probability': prior},
                       {'detection_probability': 1},
                       {'prediction_decay': 1.1},
                       {'prediction_diffusion': -0.1},
                       {'neighbor_hit_weight': 1.1},
                       {'immediate_near_distance_m': -1},
                       {'immediate_close_distance_m': 2, 'immediate_near_distance_m': 1},
                       {'immediate_near_probability': 1},
                       {'immediate_near_distance_m': 2, 'immediate_far_distance_m': 1},
                       {'max_scan_hit_probability': prior},
                       {'max_point_evidence': 0},
                       {'odometry_scan_stride': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BayesianFilterConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
