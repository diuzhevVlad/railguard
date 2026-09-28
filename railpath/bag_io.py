"""Decode PointCloud2 messages without requiring a ROS or bag-reader import."""
import numpy as np


_FORMATS = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def unpack(msg):
    """Return XYZ, intensity, optional ring IDs, and a finite/nonzero XYZ mask."""
    fields = {f.name: f for f in msg.fields}
    endian = '>' if msg.is_bigendian else '<'
    formats = [(endian + _FORMATS[f.datatype], (f.count,)) if f.count != 1
               else endian + _FORMATS[f.datatype] for f in fields.values()]
    dtype = np.dtype(dict(names=list(fields), formats=formats,
                         offsets=[f.offset for f in fields.values()], itemsize=msg.point_step))
    records = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                         strides=(msg.row_step, msg.point_step)).reshape(-1)
    xyz = np.column_stack([records[k] for k in ('x', 'y', 'z')]).astype(np.float32)
    valid = np.isfinite(xyz).all(axis=1) & (xyz != 0).any(axis=1)
    intensity = records['intensity'].astype(np.float32) if 'intensity' in fields else np.zeros(len(xyz), np.float32)
    ring = records['ring'].astype(np.int32) if 'ring' in fields else None
    return xyz, intensity, ring, valid
