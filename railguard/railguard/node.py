"""RailGuard node: PointCloud2 in, track centerline (nav_msgs/Path, 200 m) and rail points (PointCloud2) out."""

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from railpath import RailPath
from railpath.bag_io import unpack


class RailGuardNode(Node):
    def __init__(self):
        super().__init__('railguard_node')
        topic = self.declare_parameter('input_topic', '/lidar_points').value
        self.model = RailPath(self.declare_parameter('weights', '').value, self.declare_parameter('refiner', '').value or None,
                              drop_intensity=self.declare_parameter('drop_intensity', True).value)
        self.pub_path = self.create_publisher(Path, '/railguard/centerline', 1)
        self.pub_rails = self.create_publisher(PointCloud2, '/railguard/rails', 1)
        # depth 1: while a scan is processed only the newest one waits, so the node never falls behind the sensor
        self.create_subscription(PointCloud2, topic, self.on_cloud, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info(f'listening on {topic}')

    def on_cloud(self, msg):
        xyz, intensity, ring, valid = unpack(msg)
        out = self.model.predict(xyz[valid], intensity[valid], ring[valid] if ring is not None else None)
        path = Path(header=msg.header)
        for x, y, z in out['centerline']:
            if x == x:                                             # skip NaN (beyond the supported range without the refiner)
                pose = PoseStamped(header=msg.header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(x), float(y), float(z)
                path.poses.append(pose)
        self.pub_path.publish(path)
        self.pub_rails.publish(point_cloud2.create_cloud_xyz32(msg.header, out['rails']))


def main():
    rclpy.init()
    rclpy.spin(RailGuardNode())


if __name__ == '__main__':
    main()
