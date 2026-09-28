"""Range-image cache access and training samples.

Cache layout (one directory per bag, built by build_range_cache.py):
  <frame>.npz  range, xyz, intensity (float16, 128 x W), rail (uint8), centerline (201 x 3, NaN beyond valid_m),
               valid_m, far / far_rail (points beyond 40 m, for accumulation)
  poses.npy, timestamps_ns.npy, calib.npz (ring elevations, azimuth window, segments), index.json

A sample is an 8-channel image: range, x, y, z, intensity, mask, accumulated range, accumulated count.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

ROWS = 200            # centerline rows: arc length 1..200 m
IGNORE = 255          # segmentation target for pixels without a return or beyond the label horizon
MAX_VOTE_OFFSET = 12  # metres; pixels farther from the centerline do not vote during training


def parse_cache_spec(spec):
    """'path' or 'path:start:end' (end exclusive) -> (Path, start, end)."""
    parts = str(spec).split(':')
    return Path(parts[0]), int(parts[1]) if len(parts) > 1 and parts[1] else 0, int(parts[2]) if len(parts) > 2 and parts[2] else None


def ring_from_elevation(elev_deg, table):
    order = np.argsort(table)
    pos = np.interp(elev_deg, table[order], order.astype(np.float64))
    return np.clip(np.rint(pos), 0, len(table) - 1).astype(np.int64)


def sparse_rings(valid):
    """Rings whose native azimuth step is coarser than the bin width: valid pixels rarely have a valid neighbour."""
    v = valid.astype(np.float32)
    return ((v[:, 1:] * v[:, :-1]).sum(1) / np.maximum(v.sum(1), 1)) < 0.2


def fill_sparse(images, valid, rings):
    """Copy each valid pixel of a sparse ring into an empty right neighbour so those rows are not dotted."""
    src = valid[rings, :-1] & ~valid[rings, 1:]
    for img in images:
        row = img[..., rings, :]
        row[..., 1:][..., src] = row[..., :-1][..., src]
        img[..., rings, :] = row
    v = valid.copy()
    v[rings, 1:] |= src
    return v


class RangeCache:
    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self.index = json.loads((self.dir / 'index.json').read_text())
        self.frames = self.index['frames']
        calib = np.load(self.dir / 'calib.npz')
        self.elev = calib['ring_elevation_deg']
        self.az_min, self.az_step, self.width = float(calib['az_min']), float(calib['az_step']), int(calib['width'])
        self.segments = [tuple(s) for s in calib['segments']]
        self.poses = np.load(self.dir / 'poses.npy')
        self.height = len(self.elev)
        self._sparse = None

    def __len__(self):
        return len(self.frames)

    def load(self, frame):
        return dict(np.load(self.dir / self.frames[frame]['file']))

    def sparse(self, valid):
        if self._sparse is None:
            self._sparse = np.flatnonzero(sparse_rings(valid))
        return self._sparse

    def project(self, xyz):
        """Image row/column of sensor-frame points; forward is -Y (azimuth 270 deg)."""
        az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360
        col = np.floor(((az - self.az_min) % 360) / self.az_step).astype(np.int64)
        row = ring_from_elevation(np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))), self.elev)
        return row, col, (col >= 0) & (col < self.width)

    def accumulate(self, frame, n_prev):
        """Far points of the previous n_prev scans (same odometry segment) projected into this frame."""
        acc_range = np.zeros((self.height, self.width), np.float32)
        acc_count = np.zeros((self.height, self.width), np.float32)
        acc_rail = np.zeros((self.height, self.width), np.uint8)
        start = next(s[0] for s in self.segments if s[0] <= frame < s[1])
        inv = np.linalg.inv(self.poses[frame])
        for j in range(max(start, frame - n_prev), frame):
            d = np.load(self.dir / self.frames[j]['file'])
            far = d['far'].astype(np.float32)
            if len(far) == 0:
                continue
            T = inv @ self.poses[j]
            xyz = far[:, :3] @ T[:3, :3].T + T[:3, 3]
            row, col, ok = self.project(xyz)
            rng = np.linalg.norm(xyz, axis=1)
            row, col, rng, rail = row[ok], col[ok], rng[ok], d['far_rail'][ok]
            np.add.at(acc_count, (row, col), 1)
            order = np.argsort(-rng, kind='stable')          # nearest return wins
            acc_range[row[order], col[order]] = rng[order]
            acc_rail[row[rail > 0], col[rail > 0]] = rail[rail > 0]
        return acc_range, acc_count, acc_rail


def build_sample(cache, frame, n_prev=0, augmenter=None, rng=None):
    """8-channel image plus targets for one frame; n_prev previous scans merged into the accumulation channels.
    augmenter (train/augment.py) transforms the raw per-pixel arrays before the image is assembled."""
    d = cache.load(frame)
    rng_img, xyz, intensity, rail = d['range'].astype(np.float32), d['xyz'].astype(np.float32), d['intensity'].astype(np.float32), d['rail'].copy()
    centerline = d['centerline'].astype(np.float32)
    valid = rng_img > 0
    if n_prev > 0:
        acc_range, acc_count, acc_rail = cache.accumulate(frame, n_prev)
        add = (acc_rail > 0) & ~valid
        rail[add] = acc_rail[add]
    else:
        acc_range, acc_count = np.zeros_like(rng_img), np.zeros_like(rng_img)
    if augmenter is not None:
        raw = augmenter(dict(range=rng_img, xyz=xyz, intensity=intensity, rail=rail, acc_range=acc_range, acc_count=acc_count, centerline=centerline), cache, rng)
        rng_img, xyz, intensity, rail, acc_range, acc_count, centerline = (raw[k] for k in ('range', 'xyz', 'intensity', 'rail', 'acc_range', 'acc_count', 'centerline'))
        valid = rng_img > 0
        sparse = np.flatnonzero(sparse_rings(valid))
    else:
        sparse = cache.sparse(valid)
    valid = fill_sparse([rng_img, xyz, intensity, rail, acc_range, acc_count], valid, sparse)
    image = np.concatenate([rng_img[None], xyz, intensity[None], valid[None].astype(np.float32), acc_range[None], acc_count[None]])
    return dict(image=image, rail=rail, centerline=centerline, valid_m=float(d['valid_m']))


def vote_targets(image, centerline, valid_m):
    """Per pixel: offset (3,H,W) to the nearest centerline point, its arc length s (H,W), and a training mask."""
    h, w = image.shape[1:]
    xyz = image[1:4].reshape(3, -1).T
    has = (image[5].reshape(-1) > 0) & (image[0].reshape(-1) <= valid_m)
    ok = np.isfinite(centerline[:, 0])
    dist, idx = cKDTree(centerline[ok]).query(xyz[has])
    off = np.zeros((len(xyz), 3), np.float32)
    s = np.zeros(len(xyz), np.float32)
    off[has] = centerline[ok][idx] - xyz[has]
    s[has] = np.flatnonzero(ok)[idx]
    mask = np.zeros(len(xyz), bool)
    mask[has] = dist < MAX_VOTE_OFFSET
    return off.reshape(h, w, 3).transpose(2, 0, 1), s.reshape(h, w), mask.reshape(h, w)


class RailDataset(Dataset):
    """Frames from several caches. stage 'seg' yields (image, rail target); stage 'path' adds vote targets and the centerline."""

    def __init__(self, cache_specs, stage='path', n_prev=(0, 9), min_valid_m=40.0, augment=False, seed=0, augmenter=None):
        specs = [parse_cache_spec(c) for c in cache_specs]
        self.caches = [RangeCache(path) for path, _, _ in specs]
        self.items = [(ci, f['frame']) for ci, (c, (_, start, end)) in enumerate(zip(self.caches, specs))
                      for f in c.frames if f['valid_m'] >= min_valid_m and start <= f['frame'] < (end if end is not None else len(c))]
        self.stage, self.n_prev, self.augment, self.augmenter, self.seed = stage, n_prev, augment, augmenter, seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ci, frame = self.items[i]
        n = self.n_prev if isinstance(self.n_prev, int) else int(self.rng.integers(self.n_prev[0], self.n_prev[1] + 1))
        if self.augmenter is not None:
            from .augment import sample_rng
            s = build_sample(self.caches[ci], frame, n, self.augmenter, sample_rng(self.seed, ci, frame))
        else:
            s = build_sample(self.caches[ci], frame, n)
        image, centerline = s['image'], s['centerline']
        target = s['rail'].astype(np.int64)
        target[(image[5] == 0) | (image[0] > s['valid_m'])] = IGNORE
        if self.augment and self.rng.random() < 0.5:          # mirror about the forward axis
            image = image[:, :, ::-1].copy()
            image[1] = -image[1]
            target = target[:, ::-1].copy()
            target = np.where(target == 1, 2, np.where(target == 2, 1, target))
            centerline = centerline.copy()
            centerline[:, 0] = -centerline[:, 0]
        out = dict(image=torch.from_numpy(np.ascontiguousarray(image)), rail=torch.from_numpy(target), valid_m=np.float32(s['valid_m']))
        if self.stage == 'path':
            off, sv, mask = vote_targets(image, centerline, s['valid_m'])
            out.update(off=torch.from_numpy(off), s=torch.from_numpy(sv), mask=torch.from_numpy(mask),
                       centerline=torch.from_numpy(np.nan_to_num(centerline, nan=0.0)),
                       centerline_valid=torch.from_numpy(np.isfinite(centerline[:, 0])))
        return out
