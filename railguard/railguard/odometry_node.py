"""Streaming KISS-ICP publisher, independent of neural inference.

The seed speed comes from the same GT preset table as the web app (parameter `bag`), or from `initial_speed_m_s`.
Registration itself keeps the online defaults (point stride 8, no deskew, returns beyond 0.5 m, measured scan interval,
all cores): the app's offline settings (stride 1, deskew) registered too slowly online and deskew shortened the track.
"""
import uuid

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2

from railpath.bag_io import unpack
from railpath.tracking_odometry import TrackingOdometry
from obstacle_app.gt_odometry import GT_ODOMETRY_PRESETS, odometry_input


class KissIcpNode(Node):
    def __init__(self):
        super().__init__('railguard_odometry')
        topic = self.declare_parameter('input_topic', '/lidar_points').value
        output = self.declare_parameter('odometry_topic', '/railguard/odometry').value
        self.world_frame = self.declare_parameter('odometry_frame', 'railguard_odom').value
        bag = self.declare_parameter('bag', '').value
        speed = self.declare_parameter('initial_speed_m_s', 0.).value
        seed_dt = self.declare_parameter('seed_scan_dt_s', 0.).value          # 0 = measured from the first two scans
        preset = GT_ODOMETRY_PRESETS.get(bag) if bag else None
        if bag and preset is None:
            self.get_logger().warning(f'no GT odometry preset for bag {bag!r}: using initial_speed_m_s={speed}')
        if preset is not None:
            speed = preset[0]
        self.deskew = self.declare_parameter('deskew', False).value
        self.odometry = TrackingOdometry(
            speed, self.declare_parameter('tracking_max_gap_s', 2.).value,
            point_stride=self.declare_parameter('point_stride', 8).value, deskew=self.deskew,
            initial_scan_dt_s=seed_dt if seed_dt > 0 else None,
            min_distance_m=self.declare_parameter('min_distance_m', 0.5).value,
            max_num_threads=self.declare_parameter('max_num_threads', 0).value)
        self.get_logger().info(f'odometry: initial speed {speed} m/s' + (f' (preset {bag})' if preset else '') + (' with deskew' if self.deskew else ''))
        self.sensor_frame = None
        self.epoch = None
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(Odometry, output, qos)
        self.create_subscription(PointCloud2, topic, self.on_cloud,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info(f'KISS-ICP: {topic} -> {output}')

    def on_cloud(self, msg):
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        try:
            if msg.header.frame_id != self.sensor_frame:
                self.sensor_frame = msg.header.frame_id
                self.odometry.reset()
            if self.deskew:
                xyz, point_times = odometry_input(msg)
                matrix, reset = self.odometry.update(xyz, stamp, point_times)
            else:
                xyz, _, _, valid = unpack(msg)
                matrix, reset = self.odometry.update(xyz[valid], stamp)
            if reset or self.epoch is None:
                # A changed world frame explicitly invalidates the old track map,
                # including after a bag loop, gap, or this node restarting.
                self.epoch = f'{self.world_frame}_{uuid.uuid4().hex}'
            odom = Odometry()
            odom.header.stamp = msg.header.stamp
            odom.header.frame_id = self.epoch
            odom.child_frame_id = msg.header.frame_id
            position, orientation = odom.pose.pose.position, odom.pose.pose.orientation
            position.x, position.y, position.z = map(float, matrix[:3, 3])
            orientation.x, orientation.y, orientation.z, orientation.w = map(
                float, Rotation.from_matrix(matrix[:3, :3]).as_quat())
            self.publisher.publish(odom)
        except Exception as exc:
            self.odometry.reset()
            self.epoch = None
            self.get_logger().error(f'KISS-ICP scan failed: {exc}')


def main():
    rclpy.init()
    node = KissIcpNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
