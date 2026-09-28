"""Place stationary objects at the train's position at a source-bag timestamp."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def timed_anchor(item, start_ns, timestamps, poses, track_bases, merged):
    target = start_ns + round(item.encounter_seconds * 1e9)
    times = np.asarray(timestamps, dtype=np.int64)
    if target < times[0] or target > times[-1]:
        raise ValueError(f"Obstacle {item.id}: time must fall within the recorded LiDAR scans")
    right = min(int(np.searchsorted(times, target)), len(times) - 1)
    left = max(0, right - 1)
    fraction = (target - times[left]) / max(1, int(times[right] - times[left]))
    position = poses[left][:3, 3] * (1 - fraction) + poses[right][:3, 3] * fraction
    rotation = poses[right][:3, :3]
    if right != left:
        rotation = Slerp([0, 1], Rotation.from_matrix(
            [poses[left][:3, :3], poses[right][:3, :3]]))([fraction]).as_matrix()[0]
    forward = rotation @ np.array([0., -1., 0.])
    forward[2] = 0
    forward /= np.linalg.norm(forward)

    # Project onto a segment, rather than snapping to a sampled route vertex.
    segments = np.diff(merged.center.astype(float), axis=0)
    delta = position[:2] - merged.center[:-1, :2]
    lengths2 = np.sum(segments[:, :2] ** 2, axis=1)
    weights = np.clip(np.sum(delta * segments[:, :2], axis=1)
                      / np.maximum(lengths2, 1e-12), 0, 1)
    projected = merged.center[:-1] + weights[:, None] * segments
    index = int(np.argmin(np.linalg.norm(projected[:, :2] - position[:2], axis=1)))
    base = projected[index]
    if np.linalg.norm(base[:2] - position[:2]) <= 2:
        direction = segments[index].copy()
        direction[2] = 0
        direction /= np.linalg.norm(direction)
        if direction @ forward < 0:
            direction *= -1
    else:
        # Global merging can truncate a route. Use paired rails at this time,
        # never silently move the object to the end of the shortened map.
        if track_bases[left] is None or track_bases[right] is None:
            raise ValueError(f"Obstacle {item.id}: no reliable rails at {item.encounter_seconds:g} s")
        base = track_bases[left] * (1 - fraction) + track_bases[right] * fraction
        direction = forward
    rightward = np.array([direction[1], -direction[0], 0])
    base = base + rightward * item.lateral_offset_m
    base[2] += item.height_above_track_m
    return base, direction, target
