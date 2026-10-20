"""Bag poses and deterministic five-scan confirmation for the production player."""
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
from .cloud import normalize_axes, valid_xyz
from .world_map import source_fingerprint

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def prepare_poses(bag, topic, initial_speed, cache_root, progress, odometry_factory=TrackingOdometry):
    """Stream one scan at a time; persist only stamps, poses and segment starts."""
    from importlib.metadata import version
    identity = dict(source=source_fingerprint(bag), topic=topic, initial_speed=initial_speed,
                    version=ODOMETRY_VERSION, kiss_icp=version("kiss-icp"))
    key = json.dumps(identity, sort_keys=True)
    path = Path(cache_root) / (hashlib.sha256(key.encode()).hexdigest() + ".npz")
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                stamps, poses, resets = (data[k].copy() for k in ("timestamps", "poses", "resets"))
                if (str(data["identity"]) == key and poses.shape == (len(stamps), 4, 4)
                        and resets.shape == stamps.shape and len(stamps)
                        and np.all(np.diff(stamps) > 0) and np.isfinite(poses).all()):
                    progress(1., "Loaded cached detection odometry")
                    return stamps, poses, resets
        except (ValueError, OSError, KeyError):
            pass
    odometry = odometry_factory(initial_speed)
    stamps, poses, resets = [], [], []
    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if len(connections) != 1 or connections[0].msgtype != "sensor_msgs/msg/PointCloud2":
            raise ValueError("Select one PointCloud2 topic")
        total = max(connections[0].msgcount, 1)
        for conn, stamp, raw in reader.messages(connections=connections):
            if stamps and stamp <= stamps[-1]:
                raise ValueError("Detection playback requires unique increasing bag timestamps")
            msg = normalize_axes(reader.deserialize(raw, conn.msgtype), topic)
            pose, reset = odometry.update(valid_xyz(msg), stamp)
            stamps.append(stamp)
            poses.append(pose)
            resets.append(reset)
            if len(stamps) % 10 == 0 or len(stamps) == total:
                progress(len(stamps) / total, f"Detection odometry: {len(stamps)}/{total} scans")
    if not stamps:
        raise ValueError("Selected topic has no scans")
    if source_fingerprint(bag) != identity["source"]:
        raise ValueError("Recording changed while estimating detection odometry")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + "." + uuid.uuid4().hex + ".npz")
    try:
        np.savez_compressed(temporary, identity=key, timestamps=stamps, poses=poses, resets=resets)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return np.asarray(stamps), np.asarray(poses), np.asarray(resets)


class TrackedBagDetector:
    """Use actual preceding scans even when playback skips, pauses, or seeks."""
    def __init__(self, detector, bag, topic, timestamps, poses, resets):
        self.detector, self.bag, self.topic = detector, Path(bag), topic
        self.timestamps, self.poses, self.resets = timestamps, poses, resets
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def cached(self, stamp):
        with self.lock:
            return self.cache.get(stamp)

    def detect(self, timestamp_ns, point_cloud):
        with self.lock:
            cached = self.cached(timestamp_ns)
            if cached is not None:
                return cached
            frame = int(np.searchsorted(self.timestamps, timestamp_ns))
            if frame >= len(self.timestamps) or self.timestamps[frame] != timestamp_ns:
                raise ValueError("Frame does not match the cached detection odometry")
            start = max(0, frame - TRACKING_SETTINGS["window_scans"] + 1)
            breaks = np.flatnonzero(self.resets[start:frame+1])
            if len(breaks):
                start += int(breaks[-1])
            scans = []
            with AnyReader([self.bag], default_typestore=TYPESTORE) as reader:
                connections = [c for c in reader.connections if c.topic == self.topic]
                for i in range(start, frame + 1):
                    stamp = int(self.timestamps[i])
                    result = self.detector.cached(stamp)
                    if result is None:
                        if i == frame:
                            msg = point_cloud
                        else:
                            record = next(reader.messages(connections=connections, start=stamp, stop=stamp+1), None)
                            if record is None or record[1] != stamp:
                                raise ValueError("Recorded history does not match detection odometry")
                            msg = normalize_axes(reader.deserialize(record[2], record[0].msgtype), self.topic)
                        result = self.detector.detect(stamp, msg)
                    prediction, candidate = result
                    scans.append((i, stamp, candidate.points, self.poses[i]))
            detection = confirm_detection(candidate, scans)
            prediction = replace(prediction, details=dict(prediction.details or {},
                                                          tracking=TRACKING_SETTINGS))
            detection = replace(detection, prediction=prediction)
            result = prediction, detection
            self.cache[timestamp_ns] = result
            if len(self.cache) > 12:
                self.cache.popitem(last=False)
            return result
