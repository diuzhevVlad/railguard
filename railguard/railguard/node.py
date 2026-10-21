"""RailGuard node: scan in; track, rails, and filtered obstacle detections out."""

import threading
import os
import torch

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32, Header
from visualization_msgs.msg import Marker

from sensor_msgs.msg import PointField

from railpath import RailPath
from railpath.bag_io import unpack
from railpath.obstacle_filter import ObstacleFilterConfig, detect_obstacles
from railpath.tracked_detection import CANDIDATE_CONFIG, detect_candidates, StreamingConfirmation
from railpath.pose_sync import PoseSynchronizer
from railpath.tracked_detection import confirm_detection
from railpath.obstacle_tracking import TRACKING_SETTINGS


XYZ_FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]


def xyz_cloud(header, points):
    """PointCloud2 from an (N,3) array in one memcpy; sensor_msgs_py.create_cloud_xyz32 loops over points in Python."""
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    msg = PointCloud2(header=header, height=1, width=len(pts), fields=XYZ_FIELDS, is_bigendian=False,
                      point_step=12, row_step=12 * len(pts), is_dense=True)
    msg.data = pts.tobytes()
    return msg


class RailGuardNode(Node):
    def __init__(self):
        super().__init__('railguard_node')
        topic = self.declare_parameter('input_topic', '/lidar_points').value
        torch.set_num_threads(max(1, min(int(os.environ.get('RAILGUARD_CPU_THREADS', '4')), os.cpu_count() or 1)))
        self.model = RailPath(self.declare_parameter('weights', '').value, self.declare_parameter('refiner', '').value or None,
                              drop_intensity=self.declare_parameter('drop_intensity', True).value)
        if hasattr(self.model, 'warmup'):
            self.get_logger().info('Warming up model and refiner before accepting scans')
            self.model.warmup()
        self.pub_envelope = self.create_publisher(Marker, '/railguard/train_envelope', 1)
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
            max_pending=self.declare_parameter('odometry_queue_size', 100).value,
            timeout_s=self.declare_parameter('odometry_wait_s', .5).value)
        # Inference blocks its callback for ~0.15 s per scan, so poses are received by a separate node (PoseListener) spun
        # by its own executor in a background thread; a multi-threaded executor on this node stalled the scan callback
        # (GIL contention while the second thread kept waking on the ready scan subscription). The synchroniser is not
        # thread-safe, hence the lock.
        self.sync_lock = threading.Lock()
        self.odometry_topic = self.declare_parameter('odometry_topic', '/railguard/odometry').value
        # Re-publishing the processed scan (7 MB) costs a core; off by default, RViz shows the lidar topic instead.
        self.pub_detection_cloud = (self.create_publisher(PointCloud2, '/railguard/detection_points', QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
                                    if self.declare_parameter('publish_detection_cloud', False).value else None)
        self.pub_candidates = self.create_publisher(PointCloud2, '/railguard/obstacle_candidates', 1)
        self.pub_obstacles = self.create_publisher(PointCloud2, '/railguard/obstacles', 1)
        self.pub_detected = self.create_publisher(Bool, '/railguard/obstacle_detected', 1)
        self.pub_distance = self.create_publisher(Float32, '/railguard/obstacle_distance', 1)
        self.pub_nearest = self.create_publisher(Marker, '/railguard/nearest_obstacle', 1)
        # Obstacle points in the odometry frame (header.frame_id = the KISS-ICP epoch), consistent across scans.
        self.pub_obstacles_world = self.create_publisher(PointCloud2, '/railguard/obstacles_world', 1)
        # require_odometry=true: a scan whose pose never arrives (odometry_wait_s) is not reported at all, so every published
        # detection has a pose of the same stamp and can be placed in the world frame. false = fall back to the single-scan rule.
        self.require_odometry = self.declare_parameter('require_odometry', False).value   # the shipped config sets true
        # The lidar topic directly, best effort, newest scan only: the node never falls behind the sensor. Scans are joined
        # to KISS-ICP poses by timestamp (PoseSynchronizer). For bag evaluation with every scan, use the paced player
        # (railguard_play) instead of a RELIABLE queue, which throttles the publisher.
        self.create_subscription(PointCloud2, topic, self.on_cloud,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info(f'listening on {topic}' + ('' if self.use_odometry else ' (single-scan mode, no odometry)'))

    def on_cloud(self, msg):
        xyz, intensity, ring, valid = unpack(msg)
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        model_valid = valid & (np.einsum('ij,ij->i', xyz, xyz) > .25)
        out = self.model.predict(xyz[model_valid], ring=ring[model_valid] if ring is not None else None)
        # Candidates can only lie inside the train envelope around the predicted line, so the nearest-centre search gets
        # a broad sensor-frame window instead of the whole scan (half the points on a Pandar128 are behind or far aside).
        roi = valid & (xyz[:, 1] < 0) & (-xyz[:, 1] <= self.max_forward_m + 5.) & (np.abs(xyz[:, 0]) < 40.)
        candidate = detect_obstacles(xyz[roi], out['centerline'], out['rails'],
                                     self.obstacle_config, self.max_forward_m,
                                     candidate_detector=detect_candidates)
        self.pub_candidates.publish(xyz_cloud(msg.header, candidate.points))
        centerline = candidate.prediction.center if candidate.prediction is not None else out['centerline']
        centerline = centerline[np.isfinite(centerline).all(axis=1)]
        if len(centerline) > 400:                                  # the refined line is sampled every 0.25 m; 1 m is enough for the Path
            centerline = centerline[::4]
        path = Path(header=msg.header)
        for x, y, z in centerline:
            pose = PoseStamped(header=msg.header)
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(x), float(y), float(z)
            path.poses.append(pose)
        self.pub_path.publish(path)
        self.pub_rails.publish(xyz_cloud(msg.header, out['rails']))
        if not self.use_odometry:
            if self.pub_detection_cloud is not None:
                self.pub_detection_cloud.publish(msg)
            self.publish_detection(msg.header, confirm_detection(candidate, [(0, stamp, candidate.points, np.eye(4))], self.confirmation.settings))
            return
        with self.sync_lock:
            self.pose_sync.add_cloud(stamp, msg.header.frame_id, (msg if self.pub_detection_cloud is not None else msg.header, candidate))

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
        msg, candidate = value
        header = msg.header if isinstance(msg, PointCloud2) else msg
        if odometry is None:
            # Do not guess a transform. Large current groups still satisfy the
            # immediate rule; small groups cannot acquire temporal confirmation.
            self.reset_tracking()
            if self.require_odometry:
                self.get_logger().warning('No matching odometry: scan not reported', throttle_duration_sec=5.)
                return
            detection = confirm_detection(candidate, [(0, stamp, candidate.points, np.eye(4))],
                                          self.confirmation.settings)
            self.get_logger().warning('No matching odometry: temporal confirmation reset', throttle_duration_sec=5.)
            world = None
        else:
            epoch, matrix = odometry
            if epoch != self.odom_epoch:
                self.confirmation.reset()
                self.odom_epoch = epoch
            detection = self.confirmation.update(stamp, candidate, matrix)
            world = (epoch, matrix)
        if self.pub_detection_cloud is not None and isinstance(msg, PointCloud2):
            self.pub_detection_cloud.publish(msg)
        self.publish_detection(header, detection)
        if world is not None:
            epoch, matrix = world
            pts = np.asarray(detection.points, dtype=np.float64).reshape(-1, 3)
            world_header = Header(stamp=header.stamp, frame_id=epoch)
            self.pub_obstacles_world.publish(xyz_cloud(world_header, pts @ matrix[:3, :3].T + matrix[:3, 3]))

    def publish_detection(self, header, detection):
        envelope = Marker()
        envelope.header = header
        envelope.ns = 'train_envelope'
        envelope.id = 0
        envelope.type = Marker.LINE_LIST
        envelope.pose.orientation.w = 1.
        envelope.scale.x = .025
        envelope.color.r, envelope.color.g, envelope.color.b, envelope.color.a = .2, .9, .4, .8
        prediction = detection.prediction
        center = prediction.center if prediction is not None else np.empty((0, 3))
        center = center[np.isfinite(center).all(axis=1)]
        center = center[(-center[:, 1] >= self.obstacle_config.min_forward_m) &
                        (-center[:, 1] <= self.max_forward_m)]
        if len(center) > 200:
            center = center[::4]
        if len(center) < 2:
            envelope.action = Marker.DELETE
        else:
            cfg = self.obstacle_config
            center = center.copy()
            center[:, 2] += np.where(-center[:, 1] <= cfg.near_height_distance_m,
                                     cfg.near_min_height_above_track_m, cfg.min_height_above_track_m)
            tangent = np.gradient(center[:, :2], axis=0)
            normal = np.column_stack([tangent[:, 1], -tangent[:, 0], np.zeros(len(center))])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
            corners = [center + side*cfg.train_width_m/2*normal + [0, 0, height]
                       for side, height in [(-1, 0), (1, 0), (1, cfg.train_height_m), (-1, cfg.train_height_m)]]
            segments = []
            for edge in corners:
                segments.extend(zip(edge[:-1], edge[1:]))
            for i in (0, len(center)-1):
                segments.extend((corners[j][i], corners[(j+1)%4][i]) for j in range(4))
            envelope.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                               for segment in segments for p in segment]
        self.pub_envelope.publish(envelope)

        self.pub_obstacles.publish(xyz_cloud(header, detection.points))
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


class PoseListener(Node):
    """Receives KISS-ICP poses for the detection node; runs in its own executor thread so scans never block poses."""
    def __init__(self, detection):
        super().__init__('railguard_pose_listener')
        self.create_subscription(Odometry, detection.odometry_topic, detection.on_odometry,
                                 QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(.02, detection.flush_poses)


def main():
    rclpy.init()
    node = RailGuardNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    pose_executor = None
    if node.use_odometry:
        listener = PoseListener(node)
        pose_executor = SingleThreadedExecutor()
        pose_executor.add_node(listener)
        threading.Thread(target=pose_executor.spin, daemon=True).start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if pose_executor is not None:
            pose_executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
