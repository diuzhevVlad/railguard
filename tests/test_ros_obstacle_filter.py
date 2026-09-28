import unittest
import numpy as np

from railguard.obstacle_filter import ObstacleFilterConfig, detect_obstacles, filter_obstacle_candidates


class ObstacleFilterTests(unittest.TestCase):
    def setUp(self):
        d = np.arange(1, 201, dtype=np.float32)
        self.center = np.column_stack((np.zeros_like(d), -d, np.zeros_like(d)))

    def group(self, distance, height=.3, x=0, count=5):
        return np.array([[x + i*.01, -distance, height] for i in range(count)], np.float32)

    def test_defaults_height_distance_rails_and_clusters(self):
        groups = [self.group(1), self.group(2, .2), self.group(3, .21),
                  self.group(10, .11), self.group(20, count=4), self.group(30),
                  self.group(160)]
        points = np.vstack(groups)
        before = points.copy()
        result = detect_obstacles(points, self.center, groups[5])
        np.testing.assert_array_equal(result.points, np.vstack(groups[2:4]))
        self.assertTrue(result.detected)
        self.assertEqual(result.excluded_rail_count, 5)
        self.assertAlmostEqual(result.distance_m, 3, places=3)
        np.testing.assert_array_equal(points, before)

    def test_dimensions_and_maximum_range(self):
        points = np.vstack((self.group(20, x=1.4), self.group(30, 3.5), self.group(175)))
        self.assertFalse(detect_obstacles(points, self.center).detected)
        config = ObstacleFilterConfig(train_width_m=4, train_height_m=4)
        result = detect_obstacles(points, self.center, config=config, max_forward_m=200)
        np.testing.assert_array_equal(result.points, points)

    def test_near_removal_preserves_far_baseline_decision(self):
        cross = np.array([[i*.01, -9.95, .15] for i in range(4)] + [[.04, -10, .15]], np.float32)
        result = detect_obstacles(cross, self.center)
        np.testing.assert_array_equal(result.points, cross[-1:])

    def test_radius_chains_and_dense_contact(self):
        near = np.array([[i*.4, -10, .3] for i in range(5)])
        far = near + [0, -130, 0]
        np.testing.assert_array_equal(filter_obstacle_candidates(np.vstack((near, far))),
                                      [False]*5 + [True]*5)
        self.assertTrue(filter_obstacle_candidates(np.zeros((10000, 3))).all())

    def test_empty_invalid_curve_and_parameters(self):
        empty = detect_obstacles(np.empty((0, 3)), self.center)
        self.assertFalse(empty.detected)
        self.assertIsNone(empty.distance_m)
        broken = self.center.copy()
        broken[5] = np.nan
        self.assertFalse(detect_obstacles(self.group(20), broken).detected)
        self.assertFalse(detect_obstacles(self.group(2), np.full((200, 3), np.nan)).detected)
        with self.assertRaises(ValueError):
            detect_obstacles(self.group(2), self.center, max_forward_m=float('nan'))
        with self.assertRaises(ValueError):
            ObstacleFilterConfig(train_width_m=0)


if __name__ == '__main__':
    unittest.main()
