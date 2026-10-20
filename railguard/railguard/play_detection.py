"""Replay every LiDAR scan with a bounded window of scans awaiting detection."""
import argparse
import time

import rclpy
import rosbag2_py
from rclpy.serialization import deserialize_message
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag')
    parser.add_argument('--topic', default='/lidar_points')
    parser.add_argument('--timeout', type=float, default=60.)
    parser.add_argument('--in-flight', type=int, default=3, help='maximum outstanding scans (1..10; 1 is sequential)')
    parser.add_argument('--max-scans', type=int, default=0, help='stop after this many scans; 0 plays the whole bag')
    args = parser.parse_args()
    if not 1 <= args.in_flight <= 10 or args.max_scans < 0 or args.timeout <= 0:
        parser.error('in-flight must be 1..10; max-scans >= 0; timeout > 0')
    rclpy.init(args=[])
    node = rclpy.create_node('railguard_paced_player')
    qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
    publisher = node.create_publisher(PointCloud2, args.topic, qos)
    pending = {}
    completed = [0]
    def receive(msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id)
        if key in pending:
            del pending[key]
            completed[0] += 1
    node.create_subscription(Marker, '/railguard/train_envelope', receive, qos)
    def wait_until(ready, error):
        deadline = time.monotonic() + args.timeout
        while not ready():
            if time.monotonic() >= deadline:
                raise RuntimeError(error)
            rclpy.spin_once(node, timeout_sec=.01)
    try:
        wait_until(lambda: publisher.get_subscription_count() > 0 and
                   node.count_publishers('/railguard/train_envelope') > 0,
                   'Start the RailGuard nodes and wait until model warmup completes')
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id='sqlite3'),
                    rosbag2_py.ConverterOptions('', ''))
        topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
        if topics.get(args.topic) != 'sensor_msgs/msg/PointCloud2':
            raise ValueError('Selected topic must be PointCloud2')
        reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))
        count = 0
        started = time.monotonic()
        def check_timeout():
            if pending and time.monotonic() - min(pending.values()) > args.timeout:
                raise RuntimeError('Missing detection acknowledgement; stopping without skipping scans')
        while reader.has_next() and (not args.max_scans or count < args.max_scans):
            while len(pending) >= args.in_flight:
                rclpy.spin_once(node, timeout_sec=.01)
                check_timeout()
            _, data, _ = reader.read_next()
            msg = deserialize_message(data, PointCloud2)
            key = (msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id)
            # Never let duplicate timestamp keys acknowledge multiple scans.
            while key in pending:
                rclpy.spin_once(node, timeout_sec=.01)
                check_timeout()
            pending[key] = time.monotonic()
            publisher.publish(msg)
            count += 1
            rclpy.spin_once(node, timeout_sec=0.)
            if count % 100 == 0:
                node.get_logger().info(f'Sent {count}, processed {completed[0]}, in flight {len(pending)}')
        while pending:
            rclpy.spin_once(node, timeout_sec=.01)
            check_timeout()
        elapsed = time.monotonic() - started
        node.get_logger().info(f'Finished: {completed[0]}/{count} scans, {elapsed:.2f} s, '
                               f'{count/max(elapsed, .001):.2f} scans/s')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
