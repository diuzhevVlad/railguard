"""Also runnable inside the ROS image: python3 tests/test_ros_streaming.py."""
import unittest
from importlib.util import find_spec
from unittest.mock import patch

import numpy as np

try:
    import rclpy
    ros_node_available = find_spec('railguard.node') is not None
except ImportError:
    rclpy = None
    ros_node_available = False


@unittest.skipUnless(ros_node_available, "Run ROS integration tests inside the node image")
class TestRosStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[])

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        import railguard.node as module
        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass
            def predict(self, *args, **kwargs):
                d = np.arange(151, dtype=np.float32)
                return dict(centerline=np.column_stack([d*0, -d, d*0]), rails=np.empty((0, 3)))
        with patch.object(module, 'RailPath', FakeModel):
            self.node = module.RailGuardNode()
        self.results = []
        self.node.publish_detection = lambda header, result: self.results.append(result)

    def tearDown(self):
        self.node.destroy_node()

    def cloud(self, index, count=4):
        from std_msgs.msg import Header
        from sensor_msgs_py import point_cloud2
        header = Header()
        header.frame_id = 'lidar'
        header.stamp.sec, header.stamp.nanosec = 1, index*100_000_000
        points = np.column_stack([np.arange(count)*.001, np.full(count, -20.+index), np.ones(count)])
        return point_cloud2.create_cloud_xyz32(header, points)

    def pose(self, cloud, epoch='world'):
        from nav_msgs.msg import Odometry
        pose = Odometry()
        pose.header.stamp = cloud.header.stamp
        pose.header.frame_id = epoch
        pose.child_frame_id = cloud.header.frame_id
        pose.pose.pose.orientation.w = 1.
        pose.pose.pose.position.y = -cloud.header.stamp.nanosec / 100_000_000
        return pose

    def test_transforms_pairing_confirmation_and_new_epoch(self):
        self.assertFalse(hasattr(self.node, 'odometry'))
        for i in range(5):
            msg = self.cloud(i)
            if i % 2:
                self.node.on_cloud(msg)
                self.assertEqual(len(self.results), i)
                self.node.on_odometry(self.pose(msg))
            else:
                self.node.on_odometry(self.pose(msg))
                self.node.on_cloud(msg)
            self.assertEqual(self.results[-1].detected, i >= 3)
        msg = self.cloud(5)
        self.node.on_odometry(self.pose(msg, 'new_world'))
        self.node.on_cloud(msg)
        self.assertFalse(self.results[-1].detected)
        # Looping the bag clears tracking even if an external publisher reuses
        # its world-frame name.
        msg = self.cloud(0)
        self.node.on_odometry(self.pose(msg, 'new_world'))
        self.node.on_cloud(msg)
        self.assertFalse(self.results[-1].detected)

    def test_envelope_topic_matches_detection_dimensions_and_clearance(self):
        from types import SimpleNamespace
        from railguard.node import RailGuardNode
        markers = []
        self.node.pub_envelope = SimpleNamespace(publish=markers.append)
        msg = self.cloud(0)
        self.node.on_odometry(self.pose(msg))
        self.node.on_cloud(msg)
        RailGuardNode.publish_detection(self.node, msg.header, self.results[-1])
        marker = markers[-1]
        self.assertEqual(marker.header, msg.header)
        self.assertTrue(marker.points)
        xyz = np.array([[p.x, p.y, p.z] for p in marker.points])
        self.assertAlmostEqual(float(np.ptp(xyz[:, 0])), 2.1, places=5)
        self.assertAlmostEqual(float(xyz[:, 2].min()), .1, places=5)
        self.assertAlmostEqual(float(xyz[:, 2].max()), 3.2, places=5)

    def test_missing_pose_keeps_immediate_rule_but_breaks_support(self):
        now = [0.]
        self.node.pose_sync.clock = lambda: now[0]
        for i in range(4):
            msg = self.cloud(i)
            self.node.on_cloud(msg)
            self.node.on_odometry(self.pose(msg))
        self.assertTrue(self.results[-1].detected)
        self.node.on_cloud(self.cloud(4))
        now[0] += 1.
        self.node.pose_sync.flush()
        self.assertFalse(self.results[-1].detected)
        self.node.on_cloud(self.cloud(5, 51))
        now[0] += 1.
        self.node.pose_sync.flush()
        self.assertTrue(self.results[-1].detected)
        self.assertEqual(self.results[-1].point_count, 51)
        msg = self.cloud(6)
        self.node.on_cloud(msg)
        self.node.on_odometry(self.pose(msg))
        self.assertFalse(self.results[-1].detected)


if __name__ == '__main__':
    unittest.main()
