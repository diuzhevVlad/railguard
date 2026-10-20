"""Bag odometry built in scan order, with replay of already processed history."""
from bisect import bisect_left
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import uuid

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from railpath.obstacle_tracking import TRACKING_SETTINGS
from railpath.tracked_detection import confirm_detection
from railpath.tracking_odometry import TrackingOdometry, ODOMETRY_VERSION
from .cloud import normalize_axes
from .gt_odometry import detection_initial_speed, gt_odometry_preset, odometry_input
from .world_map import source_fingerprint

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
POSE_CACHE_VERSION = 3


def pose_identity(bag, topic, initial_speed):
    from importlib.metadata import version
    preset = gt_odometry_preset(bag)
    return dict(source=source_fingerprint(bag), topic=topic,
                initial_speed=detection_initial_speed(bag, initial_speed),
                seed_scan_dt_s=preset[1] if preset else .1,
                point_stride=1, deskew=True, min_distance_m=0., max_num_threads=2,
                version=ODOMETRY_VERSION, cache_version=POSE_CACHE_VERSION, kiss_icp=version("kiss-icp"))


class BagOdometry:
    """One growing pose sequence per source/topic/speed, shared across playback.

    Only the next recorded scan may extend the sequence. Revisiting old scans
    never changes the ICP engine, which stays at the furthest processed scan.
    Complete caches survive restarts; incomplete engine state stays in memory.
    """
    def __init__(self, bag, topic, initial_speed, cache_root, odometry_factory=None):
        self.bag, self.topic = Path(bag), topic
        self.identity = pose_identity(bag, topic, initial_speed)
        self.key = json.dumps(self.identity, sort_keys=True)
        self.path = Path(cache_root)/(hashlib.sha256(self.key.encode()).hexdigest()+'.npz')
        self.lock = threading.RLock()
        self.odometry = (odometry_factory or TrackingOdometry)(
            self.identity['initial_speed'], point_stride=1, deskew=True,
            initial_scan_dt_s=self.identity['seed_scan_dt_s'], min_distance_m=0.,
            max_num_threads=2)
        self.timestamps, self.poses, self.resets = [], [], []
        self.sensor_frame = None
        self.reader = self.records = self.pending = None
        self.closed = False
        with AnyReader([self.bag], default_typestore=TYPESTORE) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            if len(connections) != 1 or connections[0].msgtype != 'sensor_msgs/msg/PointCloud2':
                raise ValueError('Select one PointCloud2 topic')
            self.total = connections[0].msgcount
            self.bag_start_ns = reader.start_time
        if not self.total:
            raise ValueError('Selected topic has no scans')
        self._load_cache()

    def _load_cache(self):
        if not self.path.is_file():
            return
        from zipfile import BadZipFile
        try:
            with np.load(self.path, allow_pickle=False) as data:
                stamps, poses, resets = (data[k].copy() for k in ('timestamps', 'poses', 'resets'))
                if (str(data['identity']) != self.key or stamps.shape != (self.total,)
                        or poses.shape != (self.total,4,4) or resets.shape != stamps.shape
                        or not np.all(np.diff(stamps) > 0) or not np.isfinite(poses).all()
                        or not np.allclose(poses[:,3], [0.,0.,0.,1.]) or not bool(resets[0])):
                    return
                self.timestamps, self.poses, self.resets = list(stamps), list(poses), list(resets)
        except (ValueError, OSError, KeyError, BadZipFile):
            pass

    def _next_record(self):
        if self.pending is None:
            if self.reader is None:
                self.reader = AnyReader([self.bag], default_typestore=TYPESTORE)
                self.reader.open()
                self.records = iter(self.reader.messages(connections=[c for c in self.reader.connections if c.topic == self.topic]))
            self.pending = next(self.records, None)
            if self.pending is None:
                raise ValueError('Recording ended before the expected scan count')
        return self.pending

    def _close_reader(self):
        self.records = self.pending = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def close(self):
        with self.lock:
            self.closed = True
            self._close_reader()
            self.odometry = None

    def _save(self):
        if source_fingerprint(self.bag) != self.identity['source']:
            raise ValueError('Recording changed while estimating detection odometry')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.stem+'.'+uuid.uuid4().hex+'.npz')
        try:
            np.savez_compressed(temporary, identity=self.key, timestamps=self.timestamps,
                                poses=self.poses, resets=self.resets)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def ensure(self, stamp, point_cloud):
        with self.lock:
            if self.closed:
                raise ValueError('Detection session has closed')
            index = bisect_left(self.timestamps, stamp)
            if index < len(self.timestamps) and self.timestamps[index] == stamp:
                return index
            if len(self.timestamps) == self.total:
                raise ValueError('Frame does not match the cached detection odometry')
            _, expected, _ = self._next_record()
            if stamp != expected:
                raise ValueError('Forward seeking is unavailable until odometry is complete. Resume from the processed portion.')
            if self.timestamps and stamp <= self.timestamps[-1]:
                raise ValueError('Detection requires increasing bag timestamps')
            frame = point_cloud.header.frame_id
            if self.sensor_frame is not None and frame != self.sensor_frame:
                self.odometry.reset()
            self.sensor_frame = frame
            xyz, point_times = odometry_input(point_cloud)
            pose, reset = self.odometry.update(xyz, stamp, point_times)
            self.timestamps.append(stamp)
            self.poses.append(pose)
            self.resets.append(reset)
            self.pending = None
            if len(self.timestamps) == self.total:
                self._close_reader()
                self._save()
            return len(self.timestamps)-1

    def status(self):
        with self.lock:
            complete = len(self.timestamps) == self.total
            return dict(complete=complete, processed_scans=len(self.timestamps), total_scans=self.total,
                        processed_until_s=(int(self.timestamps[-1])-self.bag_start_ns)/1e9 if self.timestamps else 0.)

    def validate_start(self, start_seconds):
        with self.lock:
            if len(self.timestamps) == self.total:
                return
            maximum = self.status()['processed_until_s']
            # The initial start=0 may precede the selected topic's first scan.
            if start_seconds > maximum + 1e-9:
                raise ValueError('Forward seeking is unavailable until odometry is complete. Resume from the processed portion.')


class TrackedBagDetector:
    """Deterministic confirmation using only current and preceding recorded scans."""
    def __init__(self, detector, odometry):
        self.detector, self.odometry = detector, odometry
        self.bag, self.topic = odometry.bag, odometry.topic
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def cached(self, stamp):
        with self.lock:
            return self.cache.get(stamp)

    def status(self):
        return self.odometry.status()

    def validate_start(self, seconds):
        self.odometry.validate_start(seconds)

    def detect(self, timestamp_ns, point_cloud):
        with self.lock:
            cached = self.cached(timestamp_ns)
            if cached is not None:
                return cached
            frame = self.odometry.ensure(timestamp_ns, point_cloud)
            start = max(0, frame - TRACKING_SETTINGS['window_scans'] + 1)
            breaks = np.flatnonzero(self.odometry.resets[start:frame+1])
            if len(breaks):
                start += int(breaks[-1])
            scans = []
            with AnyReader([self.bag], default_typestore=TYPESTORE) as reader:
                connections = [c for c in reader.connections if c.topic == self.topic]
                for i in range(start, frame+1):
                    stamp = int(self.odometry.timestamps[i])
                    result = self.detector.cached(stamp)
                    if result is None:
                        if i == frame:
                            msg = point_cloud
                        else:
                            record = next(reader.messages(connections=connections, start=stamp, stop=stamp+1), None)
                            if record is None or record[1] != stamp:
                                raise ValueError('Recorded history does not match detection odometry')
                            msg = normalize_axes(reader.deserialize(record[2], record[0].msgtype), self.topic)
                        result = self.detector.detect(stamp, msg)
                    prediction, candidate = result
                    scans.append((i, stamp, candidate.points, self.odometry.poses[i]))
            detection = confirm_detection(candidate, scans)
            prediction = replace(prediction, details=dict(prediction.details or {}, tracking=TRACKING_SETTINGS,
                tracking_mode='bag_history', tracking_history_scans=len(scans)))
            detection = replace(detection, prediction=prediction)
            result = prediction, detection
            self.cache[timestamp_ns] = result
            if len(self.cache) > 12:
                self.cache.popitem(last=False)
            return result
