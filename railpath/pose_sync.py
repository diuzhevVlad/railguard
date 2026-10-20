"""Bounded, exact-timestamp join for independently processed clouds and poses.

No ROS dependency. Results are emitted in cloud arrival order. Missing poses
expire rather than being substituted with the latest available transform.
"""
from collections import OrderedDict
import time


class PoseSynchronizer:
    def __init__(self, emit, reset, max_pending=10, timeout_s=.5, clock=time.monotonic):
        if max_pending < 1 or timeout_s <= 0:
            raise ValueError("Pose queue size and timeout must be positive")
        self.emit, self.reset = emit, reset
        self.max_pending, self.timeout_s, self.clock = max_pending, timeout_s, clock
        self.pending, self.poses = OrderedDict(), OrderedDict()
        self.last_cloud = self.sensor_frame = None

    def add_cloud(self, stamp, frame, value):
        if (self.sensor_frame != frame or
                (self.last_cloud is not None and stamp <= self.last_cloud)):
            # At startup odometry can run ahead while the first inference warms up.
            # Preserve those future poses; on rewind retain only the exact match
            # to avoid reusing poses from the previous playback epoch.
            retained = OrderedDict((key, value) for key, value in self.poses.items()
                                   if key == (stamp, frame) or
                                   (self.last_cloud is None and key[1] == frame and key[0] >= stamp))
            self.pending.clear()
            self.poses.clear()
            self.poses.update(retained)
            self.reset()
        self.sensor_frame, self.last_cloud = frame, stamp
        self.pending[(stamp, frame)] = (self.clock(), value)
        self.flush()

    def add_pose(self, stamp, frame, value):
        self.poses[(stamp, frame)] = value
        self.poses.move_to_end((stamp, frame))
        while len(self.poses) > self.max_pending:
            self.poses.popitem(last=False)
        self.flush()

    def flush(self):
        while self.pending:
            key, (arrived, value) = next(iter(self.pending.items()))
            ready = key in self.poses
            if (not ready and len(self.pending) <= self.max_pending
                    and self.clock() - arrived < self.timeout_s):
                break
            self.pending.popitem(last=False)
            pose = self.poses.pop(key, None)
            self.emit(key[0], value, pose)
            # Old unmatched odometry cannot be paired to a later cloud.
            for old in list(self.poses):
                if old[0] <= key[0]:
                    del self.poses[old]
