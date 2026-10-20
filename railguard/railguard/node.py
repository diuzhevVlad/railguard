"""RailGuard node: scan in; track, rails, and filtered obstacle detections out."""

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32
from visualization_msgs.msg import Marker

from railpath import RailPath
from railpath.bag_io import unpack
from railpath.obstacle_filter import ObstacleFilterConfig, detect_obstacles
from railpath.tracked_detection import CANDIDATE_CONFIG, detect_candidates, StreamingConfirmation
from railpath.tracking_odometry import TrackingOdometry
from railpath.obstacle_tracking import TRACKING_SETTINGS


class RailGuardNode(Node):
    def __init__(self):
        super().__init__('railguard_node')
        topic = self.declare_parameter('input_topic', '/lidar_points').value
        self.model = RailPath(self.declare_parameter('weights', '').value, self.declare_parameter('refiner', '').value or None,
                              drop_intensity=self.declare_parameter('drop_intensity', True).value)
        self.pub_path = self.create_publisher(Path, '/railguard/centerline', 1)
        self.pub_rails = self.create_publisher(PointCloud2, '/railguard/rails', 1)
        defaults = CANDIDATE_CONFIG
        self.obstacle_config = ObstacleFilterConfig(**{
            name: self.declare_parameter(name, getattr(defaults, name)).value
            for name in ('train_width_m', 'train_height_m', 'min_forward_m',
                         'min_height_above_track_m', 'near_min_height_above_track_m',
                         'near_height_distance_m', 'min_cluster_points',
                         'count_unique_xyz')},
            stabilize_envelope=False, min_near_component_extent_m=0.)
        self.max_forward_m = self.declare_parameter('max_forward_m', 150.0).value
        settings = dict(TRACKING_SETTINGS)
        for name in ('match_distance_m', 'minimum_hits', 'window_scans', 'immediate_unique_points_gt', 'max_gap_s'):
            settings[name] = self.declare_parameter('tracking_' + name, settings[name]).value
        self.confirmation = StreamingConfirmation(settings)
        self.odometry = TrackingOdometry(
            self.declare_parameter('initial_speed_m_s', 0.).value, settings['max_gap_s'])
        self.sensor_frame = None
        self.pub_candidates = self.create_publisher(PointCloud2, '/railguard/obstacle_candidates', 1)
        self.pub_obstacles = self.create_publisher(PointCloud2, '/railguard/obstacles', 1)
        self.pub_detected = self.create_publisher(Bool, '/railguard/obstacle_detected', 1)
        self.pub_distance = self.create_publisher(Float32, '/railguard/obstacle_distance', 1)
        self.pub_nearest = self.create_publisher(Marker, '/railguard/nearest_obstacle', 1)
        # depth 1: while a scan is processed only the newest one waits, so the node never falls behind the sensor
        self.create_subscription(PointCloud2, topic, self.on_cloud, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info(f'listening on {topic}')

    def on_cloud(self, msg):
        xyz, intensity, ring, valid = unpack(msg)
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if msg.header.frame_id != self.sensor_frame:
            self.sensor_frame = msg.header.frame_id
            self.odometry.reset()
            self.confirmation.reset()
        pose, reset = self.odometry.update(xyz[valid], stamp)
        if reset:
            self.confirmation.reset()
        self.model.sparse = None
        out = self.model.predict(xyz[valid], intensity[valid], ring[valid] if ring is not None else None)
        candidate = detect_obstacles(xyz[valid], out['centerline'], out['rails'],
                                     self.obstacle_config, self.max_forward_m,
                                     candidate_detector=detect_candidates)
        detection = self.confirmation.update(stamp, candidate, pose)
        self.pub_candidates.publish(point_cloud2.create_cloud_xyz32(msg.header, candidate.points))
        centerline=detection.prediction.center if detection.prediction is not None else out['centerline']
        path = Path(header=msg.header)
        for x, y, z in centerline:
            if x == x:                                             # skip NaN (beyond the supported range without the refiner)
                pose = PoseStamped(header=msg.header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(x), float(y), float(z)
                path.poses.append(pose)
        self.pub_path.publish(path)
        self.pub_rails.publish(point_cloud2.create_cloud_xyz32(msg.header, out['rails']))
        self.pub_obstacles.publish(point_cloud2.create_cloud_xyz32(msg.header, detection.points))
        self.pub_detected.publish(Bool(data=detection.detected))
        self.pub_distance.publish(Float32(data=detection.distance_m if detection.detected else float('nan')))
        nearest = Marker()
        nearest.header = msg.header
        nearest.ns = 'nearest_obstacle'
        nearest.id = 0
        if detection.nearest_point is None:
            nearest.action = Marker.DELETE
        else:
            nearest.type = Marker.SPHERE
            nearest.action = Marker.ADD
            nearest.pose.position.x, nearest.pose.position.y, nearest.pose.position.z = map(float, detection.nearest_point)
            nearest.pose.orientation.w = 1.0
            nearest.scale.x = nearest.scale.y = nearest.scale.z = 0.5
            nearest.color.r, nearest.color.g, nearest.color.b, nearest.color.a = 1.0, 1.0, 0.0, 1.0
        self.pub_nearest.publish(nearest)


def main():
    rclpy.init()
    rclpy.spin(RailGuardNode())


if __name__ == '__main__':
    main()
