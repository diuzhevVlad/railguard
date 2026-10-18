import unittest

import numpy as np

from railpath.obstacle_filter import RailPrediction
from tools.export_centerline_dataset import pack_centerlines, relative_sensor_poses


class ExportCenterlineDatasetTests(unittest.TestCase):
    def test_ragged_centerline_frame_slices(self):
        lines = [
            RailPrediction(np.array([[0, -1, 0], [1, -2, 0]], dtype=np.float32),
                           np.array([1, 2], dtype=np.float32)),
            RailPrediction(np.empty((0, 3), dtype=np.float32),
                           np.empty(0, dtype=np.float32)),
            RailPrediction(np.array([[2, -3, 0]], dtype=np.float32),
                           np.array([3], dtype=np.float32)),
        ]
        packed = pack_centerlines(lines)
        np.testing.assert_array_equal(packed["offsets"], [0, 2, 2, 3])
        np.testing.assert_array_equal(packed["xyz"][2:2], lines[1].center)
        np.testing.assert_array_equal(packed["xyz"][2:3], lines[2].center)
        np.testing.assert_array_equal(packed["distance_along_m"], [1, 2, 3])

    def test_pose_is_relative_to_first_sensor_frame(self):
        first = np.eye(4)
        first[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        first[:3, 3] = [10, 20, 1]
        movement = np.eye(4)
        movement[:3, 3] = [0, -5, 0]
        poses = np.stack([first, first @ movement])

        relative = relative_sensor_poses(poses)

        np.testing.assert_allclose(relative[0], np.eye(4), atol=1e-12)
        np.testing.assert_allclose(relative[1], movement, atol=1e-12)
        np.testing.assert_allclose(relative[1, :3, 3], [0, -5, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
