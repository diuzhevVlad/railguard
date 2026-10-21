"""Incremental KISS-ICP poses for matching detections, never for model input."""
import numpy as np


from kiss_icp.mapping import get_voxel_hash_map

ODOMETRY_VERSION = 1


class TrackingOdometry:
    def __init__(self, initial_speed_m_s=0., max_gap_s=2., factory=None, *,
                 point_stride=8, deskew=False, initial_scan_dt_s=None,
                 min_distance_m=.5, max_num_threads=0):
        if not np.isfinite(initial_speed_m_s) or initial_speed_m_s < 0:
            raise ValueError("Initial speed must be finite and nonnegative")
        self.initial_speed = float(initial_speed_m_s)
        self.max_gap_ns = max_gap_s * 1e9
        self.factory = factory
        if (not isinstance(point_stride, int) or point_stride < 1
                or not np.isfinite(min_distance_m) or min_distance_m < 0
                or (initial_scan_dt_s is not None and
                    (not np.isfinite(initial_scan_dt_s) or initial_scan_dt_s <= 0))):
            raise ValueError("Invalid odometry input settings")
        self.point_stride = point_stride
        self.deskew = deskew
        self.initial_scan_dt_s = initial_scan_dt_s
        self.min_distance_m = min_distance_m
        self.max_num_threads = max_num_threads
        self.engine = None
        self.previous = None
        self.count = 0
        self.blind = False

    def reset(self):
        if self.factory is None:
            from kiss_icp.config import load_config
            from kiss_icp.kiss_icp import KissICP
            config = load_config(None)
            config.data.deskew = self.deskew
            config.registration.max_num_threads = self.max_num_threads
            self.engine = KissICP(config)
        else:
            self.engine = self.factory()
        self.count = 0
        self.previous = None

    def update(self, points, stamp_ns, point_timestamps=None):
        reset = (self.engine is None or self.previous is None
                 or not 0 < stamp_ns - self.previous <= self.max_gap_ns)
        if reset:
            self.reset()
        xyz = np.asarray(points)
        # Blind spell: fewer than 2 % of the returns lie more than 5 m ahead (forward = -y), e.g. an obstacle fills the
        # view. Registration against the old map would lock onto the obstacle, so the map is dropped and the pose is
        # extrapolated with the last motion; the map is rebuilt from the first scan that sees ahead again. (KISS-ICP only.)
        blind = (self.factory is None and xyz.ndim == 2 and len(xyz) > 0
                 and np.count_nonzero(xyz[:, 1] < -5) < .02 * len(xyz))
        if blind and not self.blind:
            self.engine.local_map = get_voxel_hash_map(self.engine.config)
        self.blind = blind
        if blind:
            pose = self.engine.last_pose @ self.engine.last_delta
            self.engine.last_pose = pose
            self.previous, self.count = stamp_ns, self.count + 1   # keeps the gap check and the epoch intact
            return pose.copy(), reset
        times = np.asarray(point_timestamps, dtype=np.float64) if point_timestamps is not None else np.empty(0)
        if times.size and times.shape != (len(xyz),):
            raise ValueError("Point timestamps must match XYZ")
        valid = np.isfinite(xyz).all(1) & (np.einsum('ij,ij->i', xyz, xyz) > self.min_distance_m**2)
        if times.size:
            valid &= np.isfinite(times)
            times = times[valid][::self.point_stride]
        xyz = xyz[valid]
        if len(xyz) < 3:
            # An empty scan provides no alignment. Break support on both sides.
            self.engine = None
            self.previous = None
            return np.eye(4), True
        if self.count == 1:
            dt = self.initial_scan_dt_s if self.initial_scan_dt_s is not None else (stamp_ns - self.previous) / 1e9
            self.engine.last_delta[1, 3] = -self.initial_speed * dt
        self.engine.register_frame(xyz[::self.point_stride].astype(np.float64), times)
        pose = self.engine.last_pose.copy()
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            self.engine = None
            raise ValueError("Odometry returned an invalid pose")
        self.previous, self.count = stamp_ns, self.count + 1
        return pose, reset
