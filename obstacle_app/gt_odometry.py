"""Bag detection presets and scan input matching the GT KISS-ICP generator."""
import json
from pathlib import Path

import numpy as np

from .cloud import FIELD_FORMATS, xyz_view


# (initial forward speed m/s, seed scan interval s).
# Older recordings: RailGuard_rail_det/README.md, "Odometry initial speeds".
# Additional recordings: dataset odometry/*_v*/results and filtered pose matches.
# Platform: corrected GT regeneration manifest, initial_speed_mps=16.
# These are initialization seeds, not imposed velocities.
GT_ODOMETRY_PRESETS = {
    "roundT_doubleT": (10., .0805),
    "roundT_pressureGate_roundT": (10., .1102),
    "roundT_squareT_pressureGate_squareT": (10., .1026),
    "squareT_platform_squareT_switch": (15., .1043),
    "doubleT_platform": (0., .1149),
    "new_data": (0., .1013),
    "doubleT_obstacle": (0., .1001),
    "cloud_with_fake_obj": (0., .1),
    "synthetic_switch_cloud_only": (0., .1),
    "synthetic_tunnels_cloud_only": (10., .1),
    "synthetic_platform_cloud_only": (16., .1),
}


def gt_odometry_preset(bag):
    bag = Path(bag)
    name = bag.name
    scenario = bag / "obstacle_scenario.json"
    if scenario.is_file():
        name = Path(json.loads(scenario.read_text()).get("source_bag", name)).name
    return GT_ODOMETRY_PRESETS.get(name)


def detection_initial_speed(bag, fallback=None):
    preset = gt_odometry_preset(bag)
    return preset[0] if preset is not None else (float(fallback) if fallback is not None else None)


def odometry_input(msg):
    """All finite nonzero XYZ, with aligned per-point times for deskew.

    Matches the GT generator's read_valid_cloud; also handles organized clouds,
    row padding and big-endian fields without modifying the source message.
    """
    view = xyz_view(msg)
    xyz = np.column_stack([view[k].reshape(-1) for k in ("x", "y", "z")]).astype(np.float64)
    valid = np.isfinite(xyz).all(1) & (xyz != 0).any(1)
    timestamps = np.empty(0, dtype=np.float64)
    field = next((f for f in msg.fields if f.name in ("t", "timestamp", "time")), None)
    if field is not None:
        if (field.count != 1 or field.datatype not in FIELD_FORMATS or field.offset < 0):
            raise ValueError("Unsupported LiDAR timestamp field")
        dtype = np.dtype((">" if msg.is_bigendian else "<") + FIELD_FORMATS[field.datatype])
        if field.offset + dtype.itemsize > msg.point_step:
            raise ValueError("Invalid LiDAR timestamp offset")
        timestamps = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                                offset=field.offset, strides=(msg.row_step, msg.point_step)).reshape(-1)
        valid &= np.isfinite(timestamps)
        timestamps = timestamps[valid].astype(np.float64)
    return xyz[valid], timestamps
