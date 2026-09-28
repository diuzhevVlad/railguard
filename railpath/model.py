"""RailNet: U-Net on the range image with a rail-segmentation head and a centerline voting head.

Every pixel with a return predicts the 3D offset to the nearest centerline point and that point's arc length.
The path is decoded by binning the votes per metre and fitting a clothoid chain through the bin means.
"""

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import BSpline

from .data import IGNORE, ROWS

CLASS_WEIGHT = (1.0, 10.0, 10.0)   # rails are about 1 % of pixels
VOTE_BALANCE_POWER = 1.0           # each 10 m distance bin gets equal total weight in the vote loss
RAIL_Z_VOTES = 5                 # bins with at least this many rail-pixel votes anchor the centerline height
SPLINE_KNOT_STEP = 25.0            # metres between curvature knots of the clothoid decode (dev-selected)
MAX_CURVATURE = 1 / 200.0          # railway bound used by the decode (radius 200 m)
KINK_PENALTY = 1000.0              # L1 trend-filtering weight on second differences of curvature (sparse kinks; dev-selected)
KINK_ADAPTIVE = True               # scale the kink penalty by 1 / vote evidence around each knot (none where evidence is dense)
KINK_REF_VOTES = 800.0             # votes per knot neighbourhood at which the penalty scale is 1 (dev-selected)


class RailNet(nn.Module):
    def __init__(self, encoder='resnet34', in_channels=8, classes=3):
        super().__init__()
        self.net = smp.Unet(encoder, encoder_weights=None, in_channels=in_channels, classes=classes)
        dec = self.net.segmentation_head[0].in_channels
        self.vote_head = nn.Sequential(nn.Conv2d(dec, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 4, 1))

    @staticmethod
    def normalise(x):
        rng, xyz, inten, mask, acc_r, acc_c = x[:, :1], x[:, 1:4], x[:, 4:5], x[:, 5:6], x[:, 6:7], x[:, 7:8]
        return torch.cat([torch.log1p(rng) / 5, xyz[:, :1] / 20, xyz[:, 1:2] / 100, xyz[:, 2:3] / 5,
                          inten / 255, mask, torch.log1p(acc_r) / 5, torch.log1p(acc_c) / 3], dim=1)

    def features(self, x):
        """Decoder feature map (B,16,2H,W_pad): height upsampled x2, width padded to a multiple of 32."""
        w = x.shape[-1]
        x = F.pad(self.normalise(x), (0, (-w) % 32))
        x = F.interpolate(x, scale_factor=(2, 1), mode='nearest')   # keep 8 rows at the bottleneck
        return self.net.decoder(self.net.encoder(x))

    def forward(self, x):
        """x (B,8,H,W) -> seg logits (B,3,H,W), offsets (B,3,H,W) in metres, arc length (B,H,W) in metres."""
        w = x.shape[-1]
        dec = self.features(x)
        seg = F.avg_pool2d(self.net.segmentation_head(dec), (2, 1))[:, :, :, :w]
        v = F.avg_pool2d(self.vote_head(dec), (2, 1))[:, :, :, :w]
        return seg, v[:, :3] * 5.0, v[:, 3] * 50.0


def load_weights(model, path, device):
    """Load a checkpoint. Extra tensors (heads from earlier experiments) are ignored; missing ones are reported."""
    missing, unexpected = model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)
    return missing, unexpected


class SegLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHT, device=device), ignore_index=IGNORE)
        self.dice = smp.losses.DiceLoss('multiclass', classes=[1, 2], ignore_index=IGNORE)

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice(logits, target)


def vote_loss(off, s, off_t, s_t, mask, power=VOTE_BALANCE_POWER):
    """Smooth L1 on offsets and arc length, each pixel weighted by (pixels in its 10 m bin)^-power within its sample."""
    m = mask.float()
    b = (s_t / 10).long().clamp(0, ROWS // 10).view(s_t.shape[0], -1)
    counts = torch.zeros(s_t.shape[0], ROWS // 10 + 1, device=s.device).scatter_add_(1, b, m.view(m.shape[0], -1))
    w = counts.gather(1, b).view_as(m).clamp(min=1).pow(-power) * m
    w = w / w.sum().clamp(min=1e-6) * m.sum().clamp(min=1)
    n = w.sum().clamp(min=1)
    l_off = (F.smooth_l1_loss(off, off_t, reduction='none').sum(1) * w).sum() / n
    l_s = (F.smooth_l1_loss(s / 10, s_t / 10, reduction='none') * w).sum() / n
    return l_off + l_s


def _bin_votes(off, s, image, seg=None, min_votes=3, trim=3.0, max_gap=8.0):
    """Votes -> per-metre bins. Returns bin distances, mean x/z, weights, and the supported horizon (last bin before a gap)."""
    has = image[5] > 0
    pts = image[1:4] + off
    s_v, x_v, z_v = s[has].astype(np.float64), pts[0][has].astype(np.float64), pts[2][has].astype(np.float64)
    rail_v = (seg[has] > 0) if seg is not None else np.zeros(len(s_v), bool)
    w = np.ones(len(s_v))
    # vote quality: arc length should match the chord to the voted point; rail pixels are the most reliable near voters
    chord = np.hypot(x_v, pts[1][has].astype(np.float64))
    w *= np.exp(-((s_v - chord) / (2.0 + 0.03 * s_v)) ** 2)
    if seg is not None:
        w *= np.where(seg[has] > 0, 1.5, 1.0)
    keep = (s_v > 0) & (s_v < 220) & (w > 1e-3)
    s_v, x_v, z_v, w, rail_v = s_v[keep], x_v[keep], z_v[keep], w[keep], rail_v[keep]
    b = np.clip(np.rint(s_v), 1, ROWS).astype(int) - 1
    counts = np.bincount(b, minlength=ROWS).astype(float)
    mean_x = np.bincount(b, weights=x_v, minlength=ROWS) / np.maximum(counts, 1)
    keep = (counts[b] >= min_votes) & (np.abs(x_v - mean_x[b]) < trim)
    b, x_v, z_v, w, rail_v = b[keep], x_v[keep], z_v[keep], w[keep], rail_v[keep]
    n = np.bincount(b, minlength=ROWS)
    wsum = np.bincount(b, weights=w, minlength=ROWS)
    ok = n >= min_votes
    if ok.sum() < 6:
        return None
    d = np.arange(1, ROWS + 1, dtype=np.float64)
    gaps = np.diff(np.r_[0.0, d[ok]])
    stop = np.flatnonzero(gaps > max_gap)
    horizon = float(d[ok][stop[0] - 1]) if len(stop) else float(d[ok][-1])
    ok &= d <= horizon
    mx = np.bincount(b, weights=w * x_v, minlength=ROWS)[ok] / wsum[ok]
    mz = np.bincount(b, weights=w * z_v, minlength=ROWS) / np.maximum(wsum, 1e-9)
    # height: votes from wall and ceiling pixels carry the sensor-relative height of the training mount; rail pixels vote the
    # true rail height. Anchor each bin's height to the rail votes where there are enough, and carry the nearest measured
    # offset (rail-vote height minus all-vote height) to the other bins.
    n_rail = np.bincount(b[rail_v], minlength=ROWS)
    if (n_rail >= RAIL_Z_VOTES).any():
        z_rail = np.bincount(b[rail_v], weights=(w * z_v)[rail_v], minlength=ROWS) / np.maximum(np.bincount(b[rail_v], weights=w[rail_v], minlength=ROWS), 1e-9)
        sup = n_rail >= RAIL_Z_VOTES
        offset = np.interp(d, d[sup], (z_rail - mz)[sup])                  # nearest measured offset beyond the supported span
        mz = mz + offset
    return d[ok], mx, mz[ok], np.sqrt(np.minimum(n[ok], 50.0)), horizon, n[ok].astype(float)


def _clothoid_design(sq, knots):
    """Design matrix of the small-angle clothoid: x(s) = x0 + th0*s + double integral of piecewise-linear curvature."""
    d = np.arange(0, ROWS + 1, dtype=np.float64)                     # 0..200
    cols = [np.ones(len(d)), d]
    for k in range(len(knots)):
        hat = np.zeros(len(knots)); hat[k] = 1.0
        kap = np.interp(d, knots, hat)                                # holds the end values beyond the knots
        head = np.concatenate([[0.0], np.cumsum(0.5 * (kap[1:] + kap[:-1]))])
        cols.append(np.concatenate([[0.0], np.cumsum(0.5 * (head[1:] + head[:-1]))]))
    A = np.column_stack(cols)
    return np.column_stack([np.interp(sq, d, A[:, j]) for j in range(A.shape[1])])


def _integrate(params, knots, d=None):
    """Exact clothoid chain: x0, theta0, curvature at knots (held beyond the last knot) -> x(d), theta(d) at 1 m steps from 0."""
    d = np.arange(0, ROWS + 1, dtype=np.float64) if d is None else d
    kap = np.interp(d, knots, params[2:])
    th = params[1] + np.concatenate([[0.0], np.cumsum(0.5 * (kap[1:] + kap[:-1]))])
    x = params[0] + np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])))])
    return x, th


def _fit_clothoid(sd, mx, w, horizon, knot_step=SPLINE_KNOT_STEP, l1_kinks=KINK_PENALTY, ridge=100.0, irls=2, refine=True, counts=None, adaptive=KINK_ADAPTIVE, max_curvature=None, ref_votes=None):
    """Clothoid chain through the bin means.

    1. Bounded linear least squares in the small-angle model (x0, heading, curvature at knots every `knot_step` m),
       robust to outlier bins (IRLS on residuals) and with an L1 trend-filtering penalty on the second differences of
       curvature (IRLS on the penalty), which yields piecewise-linear curvature with few kinks: the alignment family
       of straight / clothoid / arc. Curvature is bounded to MAX_CURVATURE.
    2. If the heading is large enough for the small-angle error to matter, a Gauss-Newton refinement with the exact
       integral of sin(theta), seeded by step 1.
    Returns x(d) for d = 1..ROWS and the parameters.
    """
    from scipy.optimize import least_squares, lsq_linear
    n_knots = max(2, int(np.ceil(horizon / knot_step)) + 1)
    knots = np.linspace(0, max(horizon, knot_step), n_knots)
    A = _clothoid_design(sd, knots)
    n_par = A.shape[1]
    kmax = MAX_CURVATURE if max_curvature is None else max_curvature
    ref_votes = KINK_REF_VOTES if ref_votes is None else ref_votes
    lb = np.r_[-2.0, -0.4, -kmax * np.ones(n_knots)]
    ub = np.r_[2.0, 0.4, kmax * np.ones(n_knots)]
    D1 = np.zeros((n_knots - 1, n_par))
    for i in range(n_knots - 1):
        D1[i, 2 + i], D1[i, 3 + i] = -1.0, 1.0
    D2 = np.zeros((max(n_knots - 2, 0), n_par))
    for i in range(n_knots - 2):
        D2[i, 2 + i], D2[i, 3 + i], D2[i, 4 + i] = 1.0, -2.0, 1.0
    wt, pen = w.copy(), np.ones(len(D2))
    scale = np.ones(len(D2))
    if adaptive and counts is not None and len(D2):
        # evidence around each interior knot: votes in the bins within half a knot step; penalty ~ 1 / evidence
        for i in range(len(D2)):
            near = np.abs(sd - knots[i + 1]) <= knot_step / 2
            n = counts[near].sum() if near.any() else 0.0
            scale[i] = np.clip(ref_votes / max(n, 1.0), 0.0, 20.0)
        scale[scale < 1.0] = 0.0          # dense evidence: no prior at all
    p = np.zeros(n_par)
    for it in range(irls + 1):
        rows = [A * wt[:, None], ridge * D1]
        if len(D2):
            rows.append(l1_kinks * (scale * pen)[:, None] * D2)
        M = np.vstack(rows)
        rhs = np.r_[mx * wt, np.zeros(len(D1)), np.zeros(len(D2))]
        p = lsq_linear(M, rhs, bounds=(lb, ub), lsmr_tol='auto').x
        r = np.abs(A @ p - mx)
        wt = w * np.where(r < 1.0, 1.0, 1.0 / r)
        if len(D2):
            k2 = np.abs(D2 @ p)
            pen = 1.0 / np.sqrt(k2 ** 2 + 1e-8) / np.sqrt(1.0 / 1e-8)   # IRLS for the L1 norm, normalised so a zero kink weighs 1
    x, th = _integrate(p, knots)
    if refine and np.abs(th[: int(horizon) + 1]).max() > 0.15:
        def resid(q):
            xq, _ = _integrate(q, knots)
            out = [wt * (np.interp(sd, np.arange(0, ROWS + 1), xq) - mx), ridge * (D1 @ q)]
            if len(D2):
                out.append(l1_kinks * scale * pen * (D2 @ q))
            return np.concatenate(out)
        p = least_squares(resid, p, bounds=(lb, ub), max_nfev=40, x_scale='jac').x
        x, th = _integrate(p, knots)
    return x[1:], p


def decode_path(off, s, image, seg=None, extrapolate=20.0, l1_kinks=KINK_PENALTY, refine=True, adaptive=KINK_ADAPTIVE, return_state=False, **fit_kw):
    """Votes -> centerline x(d), z(d) for d = 1..ROWS (NaN where unsupported) and the supported horizon in metres.

    off (3,H,W) and s (H,W) are model outputs as numpy; image is the 8-channel input (xyz in 1:4, mask in 5);
    seg (H,W) is the optional segmentation argmax. The clothoid chain has piecewise-linear curvature with knots every
    25 m, bounded to a 200 m radius, and continues the last curvature for `extrapolate` metres past the horizon.
    """
    d = np.arange(1, ROWS + 1, dtype=np.float64)
    nan = np.full(ROWS, np.nan)
    bins = _bin_votes(off, s, image, seg)
    if bins is None:
        return (nan, nan, 0.0, None) if return_state else (nan, nan, 0.0)
    sd, mx, mz, w, horizon, counts = bins
    state = None
    x, p = _fit_clothoid(sd, mx, w, horizon, l1_kinks=l1_kinks, refine=refine, counts=counts, adaptive=adaptive, **fit_kw)
    if return_state:
        step = fit_kw.get('knot_step', SPLINE_KNOT_STEP)
        full = np.arange(0, ROWS + 1e-6, step)                      # full knot grid 0..200
        n_fit = len(p) - 2
        kap = np.r_[p[2:], np.full(len(full) - n_fit, p[-1])][:len(full)]   # hold the last curvature beyond the fit
        evidence = np.array([counts[np.abs(sd - k) <= step / 2].sum() for k in full])
        state = dict(params=np.r_[p[:2], kap], knots=full, evidence=evidence, horizon=horizon, bin_d=sd, bin_counts=counts)
    # height: piecewise-linear grade with knots every 50 m, least squares
    kz = np.linspace(0, max(horizon, 50.0), max(2, int(np.ceil(horizon / 50.0)) + 1))
    Bz = np.column_stack([np.interp(sd, kz, np.eye(len(kz))[i]) for i in range(len(kz))])
    pz = np.linalg.lstsq(Bz * w[:, None], mz * w, rcond=None)[0]
    z = np.column_stack([np.interp(d, kz, np.eye(len(kz))[i]) for i in range(len(kz))]) @ pz
    beyond = d > horizon + extrapolate
    x[beyond], z[beyond] = np.nan, np.nan
    if return_state:
        return x, z, horizon, state
    return x, z, horizon
