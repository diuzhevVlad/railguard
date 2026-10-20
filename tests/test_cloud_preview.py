"""Preview retains Falcon-scale scans and does not alias periodic laser beams."""
from types import SimpleNamespace
import unittest

import numpy as np

from obstacle_app.cloud import preview_xyz


def message(points, height=1, padding=0, big_endian=False):
    points = np.asarray(points, dtype=">f4" if big_endian else "<f4")
    width = len(points) // height
    rows = [row.tobytes() + bytes(padding) for row in points.reshape(height, width, 3)]
    return SimpleNamespace(width=width, height=height, point_step=12,
        row_step=width * 12 + padding, data=b"".join(rows), is_bigendian=big_endian,
        fields=[SimpleNamespace(name=name, offset=i * 4, datatype=7, count=1)
                for i, name in enumerate(("x", "y", "z"))])


class CloudPreviewTest(unittest.TestCase):
    def test_falcon_sized_scan_keeps_all_returns_and_coordinates(self):
        xyz = np.random.default_rng(4).uniform(1, 400, (94848, 3)).astype(np.float32)
        msg = message(xyz)
        before = msg.data
        np.testing.assert_array_equal(preview_xyz(msg), xyz)
        self.assertEqual(msg.data, before)

    def test_sampling_covers_periodic_channels_and_is_deterministic(self):
        # A fixed stride of four would retain only channel zero.
        xyz = np.column_stack((np.arange(120000), np.arange(120000) % 4,
                               np.ones(120000))).astype(np.float32)
        msg = message(xyz)
        a = preview_xyz(msg, max_points=30000)
        self.assertEqual(len(a), 30000)
        np.testing.assert_array_equal(a, preview_xyz(msg, max_points=30000))
        self.assertTrue(np.all(np.diff(a[:, 0]) > 0))
        np.testing.assert_array_equal(np.unique(a[:, 1]), [0, 1, 2, 3])
        np.testing.assert_array_equal(a, xyz[a[:, 0].astype(int)])

    def test_padded_organized_big_endian_scan_and_synthetic_tail(self):
        xyz = np.array([[1, 2, 3], [np.nan, 0, 0], [0, 0, 0], [4, 5, 6],
                        [7, 8, 9], [10, 11, 12]], dtype=np.float32)
        msg = message(xyz, height=2, padding=7, big_endian=True)
        np.testing.assert_array_equal(preview_xyz(msg), xyz[[0, 3, 4, 5]])
        sampled = preview_xyz(msg, max_points=1, tail_points=2)
        self.assertEqual(len(sampled), 3)
        np.testing.assert_array_equal(sampled[-2:], xyz[-2:])

    def test_empty_cloud_and_invalid_budget(self):
        msg = message(np.empty((0, 3)))
        self.assertEqual(preview_xyz(msg).shape, (0, 3))
        with self.assertRaises(ValueError):
            preview_xyz(msg, max_points=0)


if __name__ == "__main__":
    unittest.main()
