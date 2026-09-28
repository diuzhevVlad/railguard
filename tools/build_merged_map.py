"""Build the same odometry/rail map used by the app, for the Rerun viewer."""
import argparse
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from obstacle_app.detection import build_merged_rail_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--initial-speed', type=float, default=0.)
    args = parser.parse_args()
    with AnyReader([args.bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        topics = [c.topic for c in reader.connections if c.msgtype == 'sensor_msgs/msg/PointCloud2']
    if len(topics) != 1:
        parser.error('Expected one PointCloud2 topic')
    build_merged_rail_map(args.bag, topics[0], args.initial_speed, args.output,
                         lambda progress, message: print(f'{progress:.0%} {message}', flush=True))


if __name__ == '__main__':
    main()
