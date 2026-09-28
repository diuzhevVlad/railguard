"""Inference API: one LiDAR scan in, the track centerline to 200 m out. No ROS, no Rerun, no postprocessing.

    from railpath import RailPath
    rp = RailPath('results/path/best.pt', refiner='results/refine')      # refiner optional
    out = rp.predict(xyz, ring=None)                                       # xyz (N,3) float32 in the LiDAR frame, forward = -Y, Z up
    out['centerline']   # (200,3): x, y, z of the track centre at 1..200 m ahead (y = -d); NaN beyond the supported range without a refiner
    out['sigma']        # (200,): lateral 1-sigma of the centerline in metres (calibrated) when a refiner is loaded, else None
    out['horizon']      # metres up to which the votes support the curve
    out['rails']        # (M,3) points labelled as rail, out['rail_side'] (M,) 1 = left, 2 = right
    out['seg']          # (128,1000) segmentation image: 0 background, 1 left rail, 2 right rail; out['image'] the 8-channel input

Expects a Hesai Pandar128 layout: 128 rings; if `ring` is None the ring is taken from the elevation via ring_elevation_deg.npy.
About 35 ms per scan on a GPU (vote model 6 ms, decode 9 ms on the CPU, refiner 19 ms).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from .data import ROWS, fill_sparse, ring_from_elevation, sparse_rings  # noqa: E402
from .model import RailNet, decode_path, load_weights  # noqa: E402

RING_TABLE = np.load(Path(__file__).resolve().parent / 'ring_elevation_deg.npy')
RINGS, AZ_MIN, AZ_STEP, WIDTH = 128, 220.0, 0.1, 1000


def make_image(xyz, intensity=None, ring=None, sparse=None):
    """Single-scan 8-channel range image (range, x, y, z, intensity, mask, 0, 0), same layout as the training cache."""
    xyz = np.asarray(xyz, np.float32)
    intensity = np.zeros(len(xyz), np.float32) if intensity is None else np.asarray(intensity, np.float32)
    rng = np.sqrt(np.einsum('ij,ij->i', xyz, xyz))
    if ring is None:
        ring = ring_from_elevation(np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))), RING_TABLE)
    ring = np.asarray(ring, np.int64)
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360
    col = np.floor(((az - AZ_MIN) % 360) / AZ_STEP).astype(np.int64)
    idx = np.flatnonzero(np.isfinite(rng) & (rng > 0.5) & (col < WIDTH) & (ring >= 0) & (ring < RINGS))
    order = idx[np.argsort(-rng[idx])]                                        # farthest first, so the nearest return per pixel is written last
    pix = ring[order] * WIDTH + col[order]
    img_range = np.zeros(RINGS * WIDTH, np.float32); img_range[pix] = rng[order]; img_range = img_range.reshape(RINGS, WIDTH)
    img_xyz = np.zeros((3, RINGS * WIDTH), np.float32); img_xyz[:, pix] = xyz[order].T; img_xyz = img_xyz.reshape(3, RINGS, WIDTH)
    img_int = np.zeros(RINGS * WIDTH, np.float32); img_int[pix] = intensity[order]; img_int = img_int.reshape(RINGS, WIDTH)
    mask = img_range > 0
    if sparse is None:
        sparse = np.flatnonzero(sparse_rings(mask))
    acc = np.zeros_like(img_range)
    mask = fill_sparse([img_range, img_xyz, img_int, acc], mask, sparse)
    image = np.concatenate([img_range[None], img_xyz, img_int[None], mask[None].astype(np.float32), acc[None], acc.copy()[None]])
    return image, sparse


def make_image_gpu(xyz, intensity, ring, device, sparse=None):
    """make_image on the GPU: the nearest return per pixel by scatter-min, O(n) instead of a sort (3x faster on multi-return clouds)."""
    x = torch.from_numpy(np.ascontiguousarray(xyz, np.float32)).to(device, non_blocking=True)
    inten = torch.from_numpy(np.zeros(len(xyz), np.float32) if intensity is None else np.ascontiguousarray(intensity, np.float32)).to(device, non_blocking=True)
    if ring is None:
        ring = ring_from_elevation(np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))), RING_TABLE)
    ring_t = torch.from_numpy(np.asarray(ring, np.int64)).to(device, non_blocking=True)
    rng = x.norm(dim=1)
    # column index on the CPU with the same float32 NumPy arithmetic as training: Hesai azimuths sit exactly on 0.1 deg bin edges,
    # and the last-bit difference of the GPU atan2 would move whole rings by one column
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360
    col = torch.from_numpy(np.floor(((az - AZ_MIN) % 360) / AZ_STEP).astype(np.int64)).to(device, non_blocking=True)
    keep = torch.isfinite(rng) & (rng > 0.5) & (col < WIDTH) & (ring_t >= 0) & (ring_t < RINGS)
    x, inten, rng, pix = x[keep], inten[keep], rng[keep], ring_t[keep] * WIDTH + col[keep]
    n_pix = RINGS * WIDTH
    best = torch.full((n_pix,), float('inf'), device=device).scatter_reduce_(0, pix, rng, 'amin')
    win = rng == best[pix]                                                  # the nearest return of its pixel (ties: any one of them)
    idx = torch.full((n_pix,), -1, dtype=torch.long, device=device).scatter_reduce_(0, pix[win], torch.nonzero(win)[:, 0], 'amax')
    has = idx >= 0
    img = torch.zeros(5, n_pix, device=device)
    sel = idx[has]
    img[0, has] = rng[sel]; img[1:4, has] = x[sel].T; img[4, has] = inten[sel]
    img = img.view(5, RINGS, WIDTH).cpu().numpy()
    img_range, img_xyz, img_int = img[0], img[1:4], img[4]
    mask = img_range > 0
    if sparse is None:
        sparse = np.flatnonzero(sparse_rings(mask))
    acc = np.zeros_like(img_range)
    mask = fill_sparse([img_range, img_xyz, img_int, acc], mask, sparse)
    image = np.concatenate([img_range[None], img_xyz, img_int[None], mask[None].astype(np.float32), acc[None], acc.copy()[None]])
    return image, sparse


class RailPath:
    def __init__(self, weights, refiner=None, drop_intensity=True, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = RailNet().to(self.device); load_weights(self.model, weights, self.device); self.model.eval()
        self.drop_intensity = drop_intensity
        self.weights, self.refiner_dir, self.refiner, self.sparse = weights, Path(refiner) if refiner else None, None, None

    def predict_image(self, image):
        """Run on a prepared 8-channel image (e.g. a training-cache frame)."""
        image = image.copy()
        if self.drop_intensity:
            image[4] = 0
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
            seg, off, s = self.model(torch.from_numpy(image)[None].to(self.device))
        seg = seg[0].argmax(0).cpu().numpy()
        x, z, horizon, state = decode_path(off[0].float().cpu().numpy(), s[0].float().cpu().numpy(), image, seg, return_state=True)
        sigma = None
        if self.refiner_dir is not None and state is not None:
            if self.refiner is None:
                from .refiner import RefinerRunner
                self.refiner = RefinerRunner(self.refiner_dir, state['knots'], self.weights, self.device)
            x, sigma = self.refiner.run(image, z, horizon, state)
            zf = np.where(np.isfinite(z), z, np.nan)                        # height is decoded to horizon + 20 m; hold it beyond
            good = np.flatnonzero(np.isfinite(zf)); z = np.interp(np.arange(ROWS), good, zf[good]) if len(good) else np.zeros(ROWS)
        d = np.arange(1, ROWS + 1, dtype=np.float32)
        has = image[5] > 0
        pts = image[1:4].transpose(1, 2, 0)
        rail = (seg > 0) & has
        return dict(centerline=np.column_stack([x, -d, z]).astype(np.float32), sigma=sigma, horizon=float(horizon),
                    rails=pts[rail], rail_side=seg[rail], seg=seg, image=image)

    def predict(self, xyz, intensity=None, ring=None):
        build = make_image_gpu if self.device.type == 'cuda' else None
        image, self.sparse = (build(xyz, intensity, ring, self.device, self.sparse) if build else make_image(xyz, intensity, ring, self.sparse))
        return self.predict_image(image)
