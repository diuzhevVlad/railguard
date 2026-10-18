"""RailGuard node: scan in; track, rails, and filtered obstacle detections out."""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32

from railpath import RailPath
from railpath.bag_io import unpack
from railpath.detection_config import ObstacleDetectionConfig
from railpath.obstacle_filter import make_rail_prediction
from railpath.occupancy_filter import ObstacleDetectionPipeline


OCCUPANCY_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
]


def create_path(header, centerline):
    """Create a Path from finite local-LiDAR centerline samples."""
    path = Path(header=header)
    for x, y, z in centerline:
        if np.isfinite((x, y, z)).all():
            pose = PoseStamped(header=header)
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            path.poses.append(pose)
    return path


def create_occupancy_cloud(header, occupancy):
    """Create XYZ + probability-as-intensity samples for RViz."""
    if occupancy is None or not len(occupancy.center):
        rows = []
    else:
        rows = np.column_stack((occupancy.center, occupancy.probabilities)).tolist()
    return point_cloud2.create_cloud(header, OCCUPANCY_FIELDS, rows)


class RailGuardNode(Node):
    def __init__(self):
        super().__init__('railguard_node')
        topic = self.declare_parameter('input_topic', '/lidar_points').value
        self.model = RailPath(self.declare_parameter('weights', '').value,
                              self.declare_parameter('refiner', '').value or None,
                              drop_intensity=self.declare_parameter('drop_intensity', True).value)
        self.pub_path = self.create_publisher(Path, '/railguard/centerline', 1)
        self.pub_filtered_path = self.create_publisher(Path, '/railguard/centerline_filtered', 1)
        self.pub_rails = self.create_publisher(PointCloud2, '/railguard/rails', 1)

        defaults = ObstacleDetectionConfig()
        parameters = {
            name: self.declare_parameter(name, value).value
            for name, value in defaults.to_parameters().items()
        }
        detection_config = ObstacleDetectionConfig.from_parameters(parameters)
        self.obstacle_config = detection_config.obstacle
        self.bayesian_config = detection_config.bayesian
        self.centerline_config = detection_config.centerline
        self.max_forward_m = detection_config.max_forward_m
        self.bayesian_filter_enabled = detection_config.bayesian_filter_enabled
        self.centerline_filter_enabled = detection_config.centerline_filter_enabled
        self.odometry_enabled = detection_config.odometry_enabled
        self.obstacle_pipeline = ObstacleDetectionPipeline(
            self.obstacle_config, self.bayesian_config,
            self.bayesian_filter_enabled, self.odometry_enabled,
            centerline_config=self.centerline_config,
            centerline_filter_enabled=self.centerline_filter_enabled)

        self.pub_obstacles = self.create_publisher(PointCloud2, '/railguard/obstacles', 1)
        self.pub_occupancy = self.create_publisher(PointCloud2, '/railguard/occupancy', 1)
        self.pub_detected = self.create_publisher(Bool, '/railguard/obstacle_detected', 1)
        self.pub_distance = self.create_publisher(Float32, '/railguard/obstacle_distance', 1)
        # Depth 1: while a scan is processed only the newest one waits, so the
        # node never falls behind the sensor.
        self.create_subscription(PointCloud2, topic, self.on_cloud,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info(
            f'listening on {topic}; Bayesian filter={self.bayesian_filter_enabled}, '
            f'centerline filter={self.centerline_filter_enabled} '
            f'({self.centerline_config.algorithm}), '
            f'KISS-ICP odometry={self.odometry_enabled}')

    def on_cloud(self, msg):
        xyz, intensity, ring, valid = unpack(msg)
        valid_xyz = xyz[valid]
        out = self.model.predict(valid_xyz, intensity[valid],
                                 ring[valid] if ring is not None else None)
        self.pub_path.publish(create_path(msg.header, out['centerline']))
        self.pub_rails.publish(point_cloud2.create_cloud_xyz32(msg.header, out['rails']))

        prediction = make_rail_prediction(
            out['centerline'], out['rails'], self.max_forward_m, out.get('sigma'))
        timestamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        detection = self.obstacle_pipeline.update(timestamp_ns, valid_xyz, prediction)
        self.pub_obstacles.publish(
            point_cloud2.create_cloud_xyz32(msg.header, detection.points))
        self.pub_occupancy.publish(create_occupancy_cloud(msg.header, detection.occupancy))
        filtered_center = (detection.centerline_estimate.prediction.center
                           if detection.centerline_estimate is not None
                           else np.empty((0, 3), dtype=np.float32))
        self.pub_filtered_path.publish(create_path(msg.header, filtered_center))
        self.pub_detected.publish(Bool(data=detection.detected))
        self.pub_distance.publish(Float32(
            data=detection.distance_m if detection.detected else float('nan')))


def main():
    rclpy.init()
    rclpy.spin(RailGuardNode())


if __name__ == '__main__':
    main()
