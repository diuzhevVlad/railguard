"""Explain encounter times that map to the same place during a stop."""
import numpy as np


def placement_warnings(anchors, poses, timestamps):
    warnings = []
    used = set()
    poses, timestamps = np.asarray(poses), np.asarray(timestamps)
    for i, first in enumerate(anchors):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(anchors)):
            if j in used:
                continue
            second = anchors[j]
            if abs(second['encounter_seconds'] - first['encounter_seconds']) < 2:
                continue
            if np.linalg.norm(np.array(second['anchor_position_world_m']) -
                              first['anchor_position_world_m']) >= .5:
                continue
            lo, hi = sorted((first['anchor_timestamp_ns'], second['anchor_timestamp_ns']))
            positions = poses[(timestamps >= lo) & (timestamps <= hi), :3, 3]
            if len(positions) and np.max(np.linalg.norm(positions - positions[0], axis=1)) < .5:
                group.append(j)
        if len(group) > 1:
            used.update(group)
            times = ', '.join(f"{anchors[j]['encounter_seconds']:g}" for j in group)
            warnings.append(f"Obstacles at {times} s are placed within 0.5 m of the same location "
                "while the estimated train is nearly stationary. Choose encounter times outside "
                "this stop to place them along different parts of the route.")
    return warnings
