"""RailGuard node: scan in; track, rails, and filtered obstacle detections out."""

import threading

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
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
from railpath.pose_sync import PoseSynchronizer
from railpath.tracked_detection import confirm_detection
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
        self.odom_epoch = None
        # use_odometry=false: single-scan mode. No pose join, no waiting: each scan is confirmed by the immediate rule only
        # (large groups), exactly what the odometry mode falls back to when a pose is missing.
        self.use_odometry = self.declare_parameter('use_odometry', True).value
        self.pose_sync = PoseSynchronizer(self.on_aligned, self.reset_tracking,
            max_pending=self.declare_parameter('odometry_queue_size', 10).value,
            timeout_s=self.declare_parameter('odometry_wait_s', .5).value)
        # Inference blocks its callback for ~0.2 s per scan. Poses are received in their own callback group so that the
        # executor (two threads, see main) keeps taking them meanwhile; otherwise the pose queue lags behind the scans and
        # no scan ever meets its own pose. The synchroniser itself is not thread-safe, hence the lock.
        self.sync_lock = threading.Lock()
        self.pose_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(.02, self.flush_poses, callback_group=self.pose_group)
        odometry_topic = self.declare_parameter('odometry_topic', '/railguard/odometry').value
        self.create_subscription(Odometry, odometry_topic, self.on_odometry,
            QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT), callback_group=self.pose_group)
        self.pub_candidates = self.create_publisher(PointCloud2, '/railguard/obstacle_candidates', 1)
        self.pub_obstacles = self.create_publisher(PointCloud2, '/railguard/obstacles', 1)
        self.pub_detected = self.create_publisher(Bool, '/railguard/obstacle_detected', 1)
        self.pub_distance = self.create_publisher(Float32, '/railguard/obstacle_distance', 1)
        self.pub_nearest = self.create_publisher(Marker, '/railguard/nearest_obstacle', 1)
        # depth 1: while a scan is processed only the newest one waits, so the node never falls behind the sensor
        self.create_subscription(PointCloud2, topic, self.on_cloud, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info(f'listening on {topic}' + ('' if self.use_odometry else ' (single-scan mode, no odometry)'))

    def on_cloud(self, msg):
        xyz, intensity, ring, valid = unpack(msg)
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.model.sparse = None
        out = self.model.predict(xyz[valid], intensity[valid], ring[valid] if ring is not None else None)
        candidate = detect_obstacles(xyz[valid], out['centerline'], out['rails'],
                                     self.obstacle_config, self.max_forward_m,
                                     candidate_detector=detect_candidates)
        self.pub_candidates.publish(point_cloud2.create_cloud_xyz32(msg.header, candidate.points))
        centerline=candidate.prediction.center if candidate.prediction is not None else out['centerline']
        path = Path(header=msg.header)
        for x, y, z in centerline:
            if x == x:                                             # skip NaN (beyond the supported range without the refiner)
                pose = PoseStamped(header=msg.header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(x), float(y), float(z)
                path.poses.append(pose)
        self.pub_path.publish(path)
        self.pub_rails.publish(point_cloud2.create_cloud_xyz32(msg.header, out['rails']))
        if not self.use_odometry:
            self.publish_detection(msg.header, confirm_detection(candidate, [(0, stamp, candidate.points, np.eye(4))], self.confirmation.settings))
            return
        with self.sync_lock:
            self.pose_sync.add_cloud(stamp, msg.header.frame_id, (msg.header, candidate))

    def flush_poses(self):
        with self.sync_lock:
            self.pose_sync.flush()

    def reset_tracking(self):
        self.confirmation.reset()
        self.odom_epoch = None

    def on_odometry(self, msg):
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        try:
            quaternion = np.array([q.x, q.y, q.z, q.w])
            if not np.isfinite(quaternion).all() or not np.isfinite([p.x, p.y, p.z]).all():
                raise ValueError('Nonfinite odometry')
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
            matrix[:3, 3] = [p.x, p.y, p.z]
        except ValueError as exc:
            self.get_logger().warning(f'Ignoring invalid odometry: {exc}')
            return
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self.sync_lock:
            self.pose_sync.add_pose(stamp, msg.child_frame_id, (msg.header.frame_id, matrix))

    def on_aligned(self, stamp, value, odometry):
        header, candidate = value
        if odometry is None:
            # Do not guess a transform. Large current groups still satisfy the
            # immediate rule; small groups cannot acquire temporal confirmation.
            self.reset_tracking()
            detection = confirm_detection(candidate, [(0, stamp, candidate.points, np.eye(4))],
                                          self.confirmation.settings)
            self.get_logger().warning('No matching odometry: temporal confirmation reset', throttle_duration_sec=5.)
        else:
            epoch, matrix = odometry
            if epoch != self.odom_epoch:
                self.confirmation.reset()
                self.odom_epoch = epoch
            detection = self.confirmation.update(stamp, candidate, matrix)
        self.publish_detection(header, detection)

    def publish_detection(self, header, detection):
        self.pub_obstacles.publish(point_cloud2.create_cloud_xyz32(header, detection.points))
        self.pub_detected.publish(Bool(data=detection.detected))
        self.pub_distance.publish(Float32(data=detection.distance_m if detection.detected else float('nan')))
        nearest = Marker()
        nearest.header = header
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
    node = RailGuardNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
