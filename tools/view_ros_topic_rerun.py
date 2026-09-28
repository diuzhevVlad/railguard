"""Subscribe to a live ROS 2 PointCloud2 topic and show it in Rerun.

Start `ros2 bag play data/new_data` separately. This program never reads the bag
files; it receives the points published on the ROS network.
"""

from __future__ import annotations

import argparse
import sys
import time

import rerun as rr

from view_bag_rerun import extract_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/lidar_points', help='ROS 2 PointCloud2 topic')
    parser.add_argument('--max-points', type=int, default=80_000,
                        help='Approximate cap for points shown per message')
    parser.add_argument('--min-range', type=float, default=0.3,
                        help='Discard zero placeholders and near returns (m)')
    parser.add_argument('--max-range', type=float, default=300.0,
                        help='Maximum displayed distance (m)')
    parser.add_argument('--max-frames', type=int, help='Exit after this many messages')
    parser.add_argument('--output', help='Write .rrd instead of opening the Rerun viewer')
    parser.add_argument('--connect', help='Existing Rerun gRPC viewer URL, for example '
                        'rerun+http://127.0.0.1:9876/proxy')
    args = parser.parse_args()
    if args.max_points < 1 or args.min_range < 0 or args.max_range <= args.min_range:
        parser.error('Invalid point cap or range')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if args.output and args.connect:
        parser.error('Choose either --output or --connect')
    return args


def main() -> int:
    args = parse_args()
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import PointCloud2
    except ImportError as exc:
        print('ROS 2 Python packages are unavailable. Source your ROS 2 setup.bash '
              'and use its Python interpreter; see README-rerun.md.', file=sys.stderr)
        print(f'Import error: {exc}', file=sys.stderr)
        return 2

    rec = rr.RecordingStream('rail_guard_live_lidar')
    if args.output:
        rec.save(args.output)
    elif args.connect:
        rec.connect_grpc(args.connect)
    else:
        rec.spawn()
    rec.log('lidar', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Best-effort subscriber also accepts a reliable publisher. Volatile QoS
    # means we receive future frames, which is appropriate for live playback.
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)

    class ViewerNode(Node):
        def __init__(self) -> None:
            super().__init__('rail_guard_rerun_viewer')
            self.frames = 0
            self.create_subscription(PointCloud2, args.topic, self.on_cloud, qos)
            self.get_logger().info(f'Waiting for {args.topic} (sensor_msgs/msg/PointCloud2)')

        def on_cloud(self, msg: PointCloud2) -> None:
            try:
                positions, colors = extract_points(msg, args.max_points,
                                                   args.min_range, args.max_range)
                # Some bags have unrelated or invalid sensor header timestamps.
                # Arrival time makes the live Rerun timeline advance reliably.
                rec.set_time('received_at', timestamp=time.time())
                rec.log('lidar/points', rr.Points3D(positions, colors=colors, radii=0.025))
                self.frames += 1
                if self.frames == 1 or self.frames % 100 == 0:
                    self.get_logger().info(
                        f'Frame {self.frames}: {len(positions):,} points '
                        f'(sensor frame: {msg.header.frame_id})')
            except (ValueError, TypeError) as exc:
                self.get_logger().error(f'Skipping malformed point cloud: {exc}')

    rclpy.init()
    node = ViewerNode()
    try:
        try:
            while rclpy.ok() and (args.max_frames is None or node.frames < args.max_frames):
                rclpy.spin_once(node, timeout_sec=0.5)
        except KeyboardInterrupt:
            pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        rec.flush()
    print(f'Received {node.frames} frames.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
