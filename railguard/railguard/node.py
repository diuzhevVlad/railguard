"""RailGuard node: scan in; track, rails, and filtered obstacle detections out."""

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32

from railpath import RailPath
from railpath.bag_io import unpack
from railpath.obstacle_filter import ObstacleFilterConfig, detect_obstacles


class RailGuardNode(Node):
    def __init__(self):
        super().__init__('railguard_node')
        topic = self.declare_parameter('input_topic', '/lidar_points').value
        self.model = RailPath(self.declare_parameter('weights', '').value, self.declare_parameter('refiner', '').value or None,
                              drop_intensity=self.declare_parameter('drop_intensity', True).value)
        self.pub_path = self.create_publisher(Path, '/railguard/centerline', 1)
        self.pub_rails = self.create_publisher(PointCloud2, '/railguard/rails', 1)
        defaults = ObstacleFilterConfig()
        self.obstacle_config = ObstacleFilterConfig(**{
            name: self.declare_parameter(name, getattr(defaults, name)).value
            for name in ('train_width_m', 'train_height_m', 'min_forward_m',
                         'min_height_above_track_m', 'near_min_height_above_track_m',
                         'near_height_distance_m', 'min_cluster_points')})
        self.max_forward_m = self.declare_parameter('max_forward_m', 150.0).value
        self.pub_obstacles = self.create_publisher(PointCloud2, '/railguard/obstacles', 1)
        self.pub_detected = self.create_publisher(Bool, '/railguard/obstacle_detected', 1)
        self.pub_distance = self.create_publisher(Float32, '/railguard/obstacle_distance', 1)
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
        detection = detect_obstacles(xyz[valid], out['centerline'], out['rails'],
                                     self.obstacle_config, self.max_forward_m)
        self.pub_obstacles.publish(point_cloud2.create_cloud_xyz32(msg.header, detection.points))
        self.pub_detected.publish(Bool(data=detection.detected))
        self.pub_distance.publish(Float32(data=detection.distance_m if detection.detected else float('nan')))


def main():
    rclpy.init()
    rclpy.spin(RailGuardNode())


if __name__ == '__main__':
    main()
