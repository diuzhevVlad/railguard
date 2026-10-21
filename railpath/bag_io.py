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
    f = [fields.get(k) for k in ('x', 'y', 'z')]
    if (all(v is not None and v.datatype == 7 and v.count == 1 for v in f) and [v.offset for v in f] == [0, 4, 8]
            and not msg.is_bigendian and msg.point_step % 4 == 0 and msg.row_step == msg.width * msg.point_step):
        # Common layout (x, y, z float32 at the start of each record): one strided view and one copy instead of three
        # field gathers; same values as the generic path below.
        flat = np.ndarray((msg.height * msg.width, msg.point_step // 4), dtype=np.dtype('<f4'), buffer=msg.data)
        xyz = np.ascontiguousarray(flat[:, :3])
    else:
        xyz = np.column_stack([records[k] for k in ('x', 'y', 'z')]).astype(np.float32)
    valid = np.isfinite(xyz).all(axis=1) & (xyz != 0).any(axis=1)
    intensity = records['intensity'].astype(np.float32) if 'intensity' in fields else np.zeros(len(xyz), np.float32)
    ring = records['ring'].astype(np.int32) if 'ring' in fields else None
    return xyz, intensity, ring, valid
