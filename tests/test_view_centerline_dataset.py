import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.view_centerline_dataset import CenterlineScene, scene_path


class ViewCenterlineDatasetTests(unittest.TestCase):
    def test_scene_reads_ragged_centerlines_by_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.npz"
            np.savez(
                path,
                timestamps_ns=np.array([1, 2], dtype=np.int64),
                sensor_pose_start=np.repeat(np.eye(4)[None], 2, axis=0),
                sensor_position_start_m=np.zeros((2, 3)),
                detected_xyz_sensor_m=np.array([[1, -2, 0], [3, -4, 0]],
                                                dtype=np.float32),
                detected_offsets=np.array([0, 1, 2]),
                gt_xyz_sensor_m=np.array([[0, -1, 0]], dtype=np.float32),
                gt_offsets=np.array([0, 1, 1]),
            )

            scene = CenterlineScene(path)

            self.assertEqual(scene.frame_count, 2)
            np.testing.assert_array_equal(scene.centerline("detected", 1), [[3, -4, 0]])
            np.testing.assert_array_equal(scene.centerline("gt", 1),
                                          np.empty((0, 3)))
            with self.assertRaises(IndexError):
                scene.centerline("gt", 2)

    def test_bag_name_resolves_to_npz_in_dataset(self):
        self.assertEqual(scene_path("eval_new_data", Path("dataset")),
                         Path("dataset/eval_new_data.npz"))
        self.assertEqual(scene_path("/tmp/other.npz", Path("dataset")),
                         Path("/tmp/other.npz"))


if __name__ == "__main__":
    unittest.main()
