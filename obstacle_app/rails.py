"""Extract local rail points with RailGuard's detector."""

from __future__ import annotations

import numpy as np

from . import rail_guard_detector as detector


def detect_rail_points(xyz: np.ndarray) -> dict[str, np.ndarray]:
    """Return RailGuard detections in the original LiDAR frame."""
    for shift in (0.0, 0.4, -0.4, 0.6, -0.6, 0.2, -0.2, 0.8, -0.8):
        candidate = xyz if shift == 0 else xyz.copy()
        if shift:
            candidate[:, 2] += shift
        seeds = detector.detect_rails(candidate)
        if not all(side in seeds for side in ("left", "right")):
            continue
        try:
            rails = detector.propagate_rails(candidate, seeds)
        except IndexError:
            # Sparse seed clusters can leave no points in the propagation
            # window. Treat that scan as undetected instead of aborting a bag.
            continue
        if not all(len(rails[side]) >= 10 for side in ("left", "right")):
            continue
        if shift:
            rails = {side: points - np.array([0, 0, shift], dtype=np.float32)
                     for side, points in rails.items()}
        return rails
    return {}
