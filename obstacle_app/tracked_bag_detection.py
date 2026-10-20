"""Incremental detection sessions for playback; never read or preprocess a bag."""
from dataclasses import replace
import threading
import time

from railpath.obstacle_tracking import TRACKING_SETTINGS
from railpath.tracked_detection import StreamingConfirmation
from railpath.tracking_odometry import TrackingOdometry
from .cloud import valid_xyz


class TrackedBagDetector:
    """One player's odometry and confirmation history, bounded to five scans.

    Create a new instance when seeking/reconnecting or switching detection jobs.
    Repeating the paused frame returns its last result without adding a hit.
    Raw neural predictions may be shared; odometry and confirmations never are.
    """
    def __init__(self, detector, initial_speed_m_s=0., odometry_factory=None):
        self.detector = detector
        self.odometry = (odometry_factory or TrackingOdometry)(initial_speed_m_s)
        self.confirmation = StreamingConfirmation()
        self.last_stamp = self.last_result = self.sensor_frame = None
        self.lock = threading.RLock()

    def cached(self, stamp):
        with self.lock:
            return self.last_result if stamp == self.last_stamp else None

    def detect(self, timestamp_ns, point_cloud):
        with self.lock:
            frame = point_cloud.header.frame_id
            if frame == self.sensor_frame and self.cached(timestamp_ns) is not None:
                return self.last_result
            # Drop the last result even if processing the next scan fails.
            self.last_stamp = self.last_result = None
            if frame != self.sensor_frame:
                self.sensor_frame = frame
                self.odometry.reset()
                self.confirmation.reset()
            started = time.perf_counter()
            try:
                pose, reset = self.odometry.update(valid_xyz(point_cloud), timestamp_ns)
                odometry_ms = (time.perf_counter() - started) * 1000
                if reset:
                    self.confirmation.reset()
                prediction, candidate = self.detector.detect(timestamp_ns, point_cloud)
                detection = self.confirmation.update(timestamp_ns, candidate, pose)
            except Exception:
                self.odometry.reset()
                self.confirmation.reset()
                raise
            prediction = replace(prediction, details=dict(prediction.details or {},
                tracking=TRACKING_SETTINGS, tracking_mode='online',
                tracking_history_scans=len(self.confirmation.scans),
                odometry_ms=round(odometry_ms, 2)))
            detection = replace(detection, prediction=prediction)
            self.last_stamp = timestamp_ns
            self.last_result = (prediction, detection)
            return self.last_result
