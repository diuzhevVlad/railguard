"""Regressions for Seyond axis conversion at bag read/write boundaries."""
from types import SimpleNamespace
import unittest

import numpy as np

from obstacle_app.cloud import normalize_axes, preview_xyz, xyz_view


def cloud(big_endian=False):
    # Two organized rows, nontrivial field offsets and row padding.
    msg = SimpleNamespace(height=2, width=2, point_step=24, row_step=55,
        is_bigendian=big_endian, data=np.full(110, 171, dtype=np.uint8),
        fields=[SimpleNamespace(name=k, datatype=7, count=1, offset=o)
                for k, o in [('x',4),('y',12),('z',16)]])
    view = xyz_view(msg)
    view['x'] = [[10,20],[30,40]]
    view['y'] = [[1,-2],[3,-4]]
    view['z'] = [[-1,-2],[-3,-4]]
    return msg


class CloudAxesTest(unittest.TestCase):
    def test_iv_points_rotation_preserves_records_and_source(self):
        for endian in (False, True):
            for topic in ('/iv_points', 'iv_points'):
                with self.subTest(endian=endian, topic=topic):
                    msg = cloud(endian)
                    original = msg.data.copy()
                    result = normalize_axes(msg, topic)
                    np.testing.assert_array_equal(msg.data, original)
                    self.assertIsNot(result, msg)
                    a, b = xyz_view(msg), xyz_view(result)
                    np.testing.assert_array_equal(b['x'], a['y'])
                    np.testing.assert_array_equal(b['y'], -a['x'])
                    np.testing.assert_array_equal(b['z'], a['z'])
                    changed = np.zeros(110, dtype=bool)
                    for row in range(2):
                        for col in range(2):
                            for off in (4,12):
                                start = row*55+col*24+off
                                changed[start:start+4] = True
                    np.testing.assert_array_equal(result.data[~changed], original[~changed])
                    np.testing.assert_array_equal(normalize_axes(result, topic, inverse=True).data, original)
                    np.testing.assert_array_equal(preview_xyz(result),
                        [[1,-10,-1],[-2,-20,-2],[3,-30,-3],[-4,-40,-4]])

    def test_regular_topics_are_unchanged(self):
        msg = cloud()
        for topic in ('/lidar_points', '/sensing/lidar/hesai128/pointcloud', '/other/iv_points'):
            self.assertIs(normalize_axes(msg, topic), msg)
            self.assertIs(normalize_axes(msg, topic, inverse=True), msg)

    def test_read_only_input_is_supported(self):
        msg = cloud()
        msg.data = np.frombuffer(msg.data.tobytes(), dtype=np.uint8)
        result = normalize_axes(msg, '/iv_points')
        self.assertTrue(result.data.flags.writeable)
        np.testing.assert_array_equal(xyz_view(result)['y'], -xyz_view(msg)['x'])


if __name__ == '__main__':
    unittest.main()
