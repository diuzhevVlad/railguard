"""PointCloud2 field handling for previews and synthetic points."""

from __future__ import annotations

import struct
from copy import copy

import numpy as np


FIELD_FORMATS = {1: "b", 2: "B", 3: "h", 4: "H", 5: "i", 6: "I", 7: "f", 8: "d"}


def normalize_axes(msg: object, topic: str, *, inverse: bool = False) -> object:
    """Use -Y forward for /iv_points; invert before writing its original topic.

    Preserve the source message, all non-XYZ fields, and row/record padding.
    Call once at each bag read boundary, before viewing or processing the scan.
    """
    if topic not in ("/iv_points", "iv_points"):
        return msg
    result = copy(msg)
    result.data = np.frombuffer(bytes(msg.data), dtype=np.uint8).copy()
    cloud = xyz_view(result)
    x, y = cloud["x"].copy(), cloud["y"].copy()
    cloud["x"], cloud["y"] = (-y, x) if inverse else (y, -x)
    return result


def xyz_view(msg: object) -> np.ndarray:
    fields = {field.name: field for field in msg.fields}
    for name in ("x", "y", "z"):
        if name not in fields or fields[name].datatype != 7 or fields[name].count != 1:
            raise ValueError("PointCloud2 needs float32 x, y and z fields")
        if fields[name].offset + 4 > msg.point_step:
            raise ValueError(f"Invalid {name} field offset")
    if msg.row_step < msg.width * msg.point_step or len(msg.data) < msg.row_step * msg.height:
        raise ValueError("Invalid PointCloud2 data length or row step")
    endian = ">" if msg.is_bigendian else "<"
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [endian + "f4"] * 3,
        "offsets": [fields[k].offset for k in ("x", "y", "z")],
        "itemsize": msg.point_step,
    })
    return np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                      strides=(msg.row_step, msg.point_step))


def valid_xyz(msg: object, stride: int = 1) -> np.ndarray:
    cloud = xyz_view(msg)
    # PointCloud2 rows can have padding; slicing rows keeps the declared stride.
    sample = cloud[:, ::stride]
    xyz = np.column_stack([sample[k].reshape(-1) for k in ("x", "y", "z")])
    good = np.isfinite(xyz).all(axis=1) & (np.einsum("ij,ij->i", xyz, xyz) >= 0.09)
    return xyz[good].astype(np.float32, copy=False)


def preview_xyz(msg: object, max_points: int = 100000, tail_points: int = 0) -> np.ndarray:
    """Keep scan detail without aliasing the LiDAR's ordered firing pattern.

    Small scans (including typical Falcon scans) retain every valid return.
    Larger scans use deterministic jittered strata rather than a fixed stride,
    which can repeatedly select the same beam in each firing group. The budget
    applies to background only; a known synthetic tail stays fully visible.
    """
    if max_points < 1:
        raise ValueError("Preview point budget must be positive")
    if tail_points < 0 or tail_points > msg.width * msg.height:
        raise ValueError("Invalid synthetic tail length")
    cloud = xyz_view(msg).reshape(-1)
    original = cloud[:-tail_points] if tail_points else cloud
    original_xyz = np.column_stack([original[k] for k in ("x", "y", "z")])
    good = np.isfinite(original_xyz).all(axis=1) & (np.einsum("ij,ij->i", original_xyz, original_xyz) >= 0.09)
    original_xyz = original_xyz[good]
    if len(original_xyz) > max_points:
        edges = np.arange(max_points + 1, dtype=np.int64) * len(original_xyz) // max_points
        # One point per disjoint interval, stable on replay, without touching
        # global RNG state or locking sampling to an individual laser channel.
        rng = np.random.default_rng(0)
        indices = edges[:-1] + (rng.random(max_points) * np.diff(edges)).astype(np.int64)
        original_xyz = original_xyz[indices]
    if not tail_points:
        return original_xyz.astype(np.float32, copy=False)
    tail = cloud[-tail_points:]
    tail_xyz = np.column_stack([tail[k] for k in ("x", "y", "z")])
    return np.concatenate((original_xyz, tail_xyz)).astype(np.float32, copy=False)


def filter_points(msg: object, keep: np.ndarray) -> object:
    """Keep complete records byte-for-byte, supporting organized/padded clouds."""
    xyz_view(msg)
    if keep.shape != (msg.height * msg.width,) or keep.dtype != np.bool_:
        raise ValueError("Point mask must contain one boolean per source record")
    if keep.all():
        return msg
    records = np.ndarray((msg.height, msg.width, msg.point_step), dtype=np.uint8,
                         buffer=msg.data, strides=(msg.row_step, msg.point_step, 1))
    msg.data = np.frombuffer(records.reshape(-1, msg.point_step)[keep].tobytes(), dtype=np.uint8)
    msg.height = 1
    msg.width = int(keep.sum())
    msg.row_step = msg.width * msg.point_step
    return msg


def append_xyz(msg: object, points: np.ndarray, rings: np.ndarray | None = None) -> object:
    """Append packed records, preserving every original point and its fields."""
    xyz_view(msg)  # Validate before modifying the message.
    if msg.height != 1 or msg.row_step != msg.width * msg.point_step:
        raise ValueError("Synthetic points require an unpadded, one-row PointCloud2")
    if len(points) == 0:
        return msg
    fields = {field.name: field for field in msg.fields}
    endian = ">" if msg.is_bigendian else "<"
    original = bytes(msg.data)
    extra = bytearray(len(points) * msg.point_step)
    intensity = 5.0
    if "intensity" in fields and fields["intensity"].datatype == 7:
        f = fields["intensity"]
        values = np.ndarray((msg.width,), dtype=np.dtype({"names": ["v"],
            "formats": [endian + "f4"], "offsets": [f.offset], "itemsize": msg.point_step}),
            buffer=msg.data)["v"]
        finite = values[np.isfinite(values)]
        if len(finite):
            intensity = float(np.median(finite))
    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
    if "timestamp" in fields and fields["timestamp"].datatype == 8:
        f = fields["timestamp"]
        values = np.ndarray((msg.width,), dtype=np.dtype({"names": ["v"],
            "formats": [endian + "f8"], "offsets": [f.offset], "itemsize": msg.point_step}),
            buffer=msg.data)["v"]
        finite = values[np.isfinite(values)]
        if len(finite):
            timestamp = float(np.median(finite))
    defaults = {"intensity": intensity, "ring": 0, "timestamp": timestamp}
    # Fill complete record columns in NumPy; zero initialization preserves
    # the previous defaults for unknown fields and record padding.
    for axis, name in enumerate(("x", "y", "z")):
        field = fields[name]
        column = np.ndarray((len(points),), dtype=endian + "f4", buffer=extra,
                            offset=field.offset, strides=(msg.point_step,))
        column[:] = points[:, axis]
    for name, value in defaults.items():
        field = fields.get(name)
        if field and field.count == 1 and field.datatype in FIELD_FORMATS:
            fmt = FIELD_FORMATS[field.datatype]
            if field.offset + struct.calcsize(fmt) <= msg.point_step:
                column = np.ndarray((len(points),), dtype=np.dtype(endian + fmt),
                                    buffer=extra, offset=field.offset, strides=(msg.point_step,))
                values = np.asarray(rings, dtype=np.int64) if name == "ring" and rings is not None else value
                if column.dtype.kind in "iu":
                    limits = np.iinfo(column.dtype)
                    if np.any(np.asarray(values) < limits.min) or np.any(np.asarray(values) > limits.max):
                        raise struct.error(f"{name} does not fit its PointCloud2 field")
                column[:] = values
    msg.data = np.frombuffer(original + extra, dtype=np.uint8)
    msg.width += len(points)
    msg.row_step = msg.width * msg.point_step
    return msg
