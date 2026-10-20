"""Incremental KISS-ICP poses for matching detections, never for model input."""
import numpy as np


ODOMETRY_VERSION = 1


class TrackingOdometry:
    def __init__(self, initial_speed_m_s=0., max_gap_s=2., factory=None):
        if not np.isfinite(initial_speed_m_s) or initial_speed_m_s < 0:
            raise ValueError("Initial speed must be finite and nonnegative")
        self.initial_speed = float(initial_speed_m_s)
        self.max_gap_ns = max_gap_s * 1e9
        self.factory = factory
        self.engine = None
        self.previous = None
        self.count = 0

    def reset(self):
        if self.factory is None:
            from kiss_icp.config import load_config
            from kiss_icp.kiss_icp import KissICP
            config = load_config(None)
            config.data.deskew = False
            self.engine = KissICP(config)
        else:
            self.engine = self.factory()
        self.count = 0
        self.previous = None

    def update(self, points, stamp_ns):
        reset = (self.engine is None or self.previous is None
                 or not 0 < stamp_ns - self.previous <= self.max_gap_ns)
        if reset:
            self.reset()
        xyz = np.asarray(points)
        xyz = xyz[np.isfinite(xyz).all(1) & (np.einsum('ij,ij->i', xyz, xyz) > .25)]
        if len(xyz) < 3:
            # An empty scan provides no alignment. Break support on both sides.
            self.engine = None
            self.previous = None
            return np.eye(4), True
        if self.count == 1:
            dt = (stamp_ns - self.previous) / 1e9
            self.engine.last_delta[1, 3] = -self.initial_speed * dt
        self.engine.register_frame(xyz[::8].astype(np.float64), np.empty(0, np.float64))
        pose = self.engine.last_pose.copy()
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            self.engine = None
            raise ValueError("Odometry returned an invalid pose")
        self.previous, self.count = stamp_ns, self.count + 1
        return pose, reset
