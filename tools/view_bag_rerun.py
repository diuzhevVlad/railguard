"""Replay a ROS 2 PointCloud2 SQLite bag directly into Rerun.

This uses rosbags and needs no ROS installation. Points stay in the LiDAR frame;
the bag has no TF topic or odometry to place them in a world map.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
import rerun as rr
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


POINTFIELD_DTYPES = {
    1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
    5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8',
}


def extract_points(msg: object, max_points: int, min_range: float, max_range: float):
    """Return (N, 3) float32 points and optional (N, 3) uint8 intensity colors."""
    fields = {field.name: field for field in msg.fields}
    required = {'x', 'y', 'z'}
    if not required <= fields.keys():
        raise ValueError(f'PointCloud2 missing fields: {sorted(required - fields.keys())}')

    selected = [fields[name] for name in ('x', 'y', 'z', 'intensity') if name in fields]
    endian = '>' if msg.is_bigendian else '<'
    names, formats, offsets = [], [], []
    for field in selected:
        code = POINTFIELD_DTYPES.get(field.datatype)
        if code is None or field.count != 1 or field.offset + np.dtype(code).itemsize > msg.point_step:
            raise ValueError(f'Unsupported PointCloud2 field: {field}')
        names.append(field.name)
        formats.append(endian + code)
        offsets.append(field.offset)
    dtype = np.dtype({'names': names, 'formats': formats,
                      'offsets': offsets, 'itemsize': msg.point_step})
    if msg.row_step < msg.width * msg.point_step:
        raise ValueError('PointCloud2 row_step is smaller than its row data')
    required_bytes = msg.row_step * msg.height
    if len(msg.data) < required_bytes:
        raise ValueError('PointCloud2 data is shorter than its declared dimensions')
    cloud = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                       strides=(msg.row_step, msg.point_step))

    # Regular decimation keeps the spatial scan pattern without drawing every
    # distant return. Filter afterwards so zero placeholders never reach Rerun.
    stride = max(1, math.ceil(msg.width * msg.height / max_points))
    sample = cloud.reshape(-1)[::stride] if msg.row_step == msg.width * msg.point_step else cloud[:, ::stride].reshape(-1)
    xyz = np.column_stack((sample['x'], sample['y'], sample['z'])).astype(np.float32, copy=False)
    ranges_sq = np.einsum('ij,ij->i', xyz, xyz)
    valid = np.isfinite(xyz).all(axis=1)
    valid &= (ranges_sq >= min_range * min_range) & (ranges_sq <= max_range * max_range)
    xyz = xyz[valid]

    colors = None
    if 'intensity' in fields:
        intensity = np.asarray(sample['intensity'][valid], dtype=np.float32)
        # Stable scale across frames; most returns in this bag are below 50.
        brightness = np.clip(np.nan_to_num(intensity, nan=0.0) * 3 + 55, 55, 255).astype(np.uint8)
        colors = np.stack((brightness, brightness, brightness), axis=1)
    return xyz, colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path, help='ROS 2 bag directory with metadata.yaml')
    parser.add_argument('--topic', default='/lidar_points')
    parser.add_argument('--start', type=float, default=0.0, help='Seconds from beginning of bag')
    parser.add_argument('--duration', type=float, help='Seconds to play; default: rest of bag')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed; 0 sends without waiting')
    parser.add_argument('--max-points', type=int, default=80_000, help='Approximate point cap per scan')
    parser.add_argument('--min-range', type=float, default=0.3, help='Drop zero placeholders and near returns (m)')
    parser.add_argument('--max-range', type=float, default=300.0, help='Maximum displayed distance (m)')
    parser.add_argument('--max-frames', type=int, help='Stop after this many scans')
    parser.add_argument('--output', type=Path, help='Write a .rrd recording instead of opening the viewer')
    args = parser.parse_args()
    if args.start < 0 or args.duration is not None and args.duration <= 0:
        parser.error('--start must be nonnegative and --duration must be positive')
    if args.speed < 0 or args.max_points < 1 or args.min_range < 0 or args.max_range <= args.min_range:
        parser.error('Invalid speed, point cap or range')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if not (args.bag / 'metadata.yaml').is_file():
        parser.error(f'Not a ROS 2 bag directory: {args.bag}')
    return args


def main() -> int:
    args = parse_args()
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        matches = [c for c in reader.connections if c.topic == args.topic]
        if not matches:
            raise ValueError(f'Topic {args.topic!r} not found; available: {[c.topic for c in reader.connections]}')
        if any(c.msgtype != 'sensor_msgs/msg/PointCloud2' for c in matches):
            raise ValueError(f'{args.topic} is not PointCloud2')

        start_ns = reader.start_time + round(args.start * 1e9)
        stop_ns = None if args.duration is None else start_ns + round(args.duration * 1e9)
        rec = rr.RecordingStream('rail_guard_lidar_bag')
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            rec.save(args.output)
        else:
            rec.spawn()
        rec.log('lidar', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        print(f'Playing {args.topic} in frame hesai_lidar; start={args.start:g}s, speed={args.speed:g}x', flush=True)

        first_ns = None
        wall_start = None
        count = 0
        try:
            for conn, bag_ns, raw in reader.messages(connections=matches, start=start_ns, stop=stop_ns):
                if first_ns is None:
                    first_ns, wall_start = bag_ns, time.monotonic()
                if args.speed > 0:
                    delay = (bag_ns - first_ns) / 1e9 / args.speed - (time.monotonic() - wall_start)
                    if delay > 0:
                        time.sleep(delay)
                msg = reader.deserialize(raw, conn.msgtype)
                positions, colors = extract_points(msg, args.max_points, args.min_range, args.max_range)
                rec.set_time('bag_time', duration=(bag_ns - reader.start_time) / 1e9)
                rec.log('lidar/points', rr.Points3D(positions, colors=colors, radii=0.025))
                count += 1
                if count == 1 or count % 100 == 0:
                    print(f'frame {count}: {len(positions):,} visible points, bag time {(bag_ns-reader.start_time)/1e9:.1f}s', flush=True)
                if args.max_frames is not None and count >= args.max_frames:
                    break
        except KeyboardInterrupt:
            print('Stopped by user.', flush=True)
        finally:
            rec.flush()
        print(f'Done: {count} frames.' + (f' Saved {args.output}' if args.output else ''), flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
