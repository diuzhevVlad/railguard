"""Scene-context refiner of the decoded centerline, with a probabilistic corridor (model and inference).

Input: the frozen path model's feature maps and the clothoid decoded from its votes. The refiner
  1. unrolls the scan into the frame of the current curve (arc length x lateral offset x height band grid: occupancy,
     mean decoder feature, mean height, intensity, range) so walls and ceiling become ridges that drift when the
     curvature is wrong;
  2. encodes the grid into arc-length tokens with a small 2D CNN;
  3. runs one query per movable knot through self-attention, cross-attention to the tokens and deformable attention
     into the range-image feature maps at the knot's projected pixel;
  4. outputs a curvature correction and a log-sigma per knot. Movable knots sit at a0 + 25 k (k = 1..M) where a0 is the
     last knot with dense vote evidence, so the correction only touches the part the votes do not pin down.
The mean is the exact integration of the corrected clothoid, the corridor is the linearised propagation of the knot
sigmas, scaled per band by calib.json produced during training.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ROWS
from .model import RailNet, load_weights

RING_TABLE = np.load(Path(__file__).resolve().parent / 'ring_elevation_deg.npy')
EVAL_D = (25, 50, 100, 150, 200)
DENSE_VOTES = 800.0                                # a knot with more votes than this is pinned by the decode
KNOT_STEP = 25.0                                   # spacing of the movable knots beyond the anchor
LAT_HALF, LAT_STEP, N_LAT = 10.0, 0.5, 40          # lateral extent and bin of the unrolled grid
H_BANDS = (0.5, 2.5)                               # height bands above the curve: floor | wall | ceiling
S_GRID = torch.arange(0, ROWS + 1, dtype=torch.float32)



# ------------------------------------------------------------------ model
def cumtrapz(v):
    return torch.cat([torch.zeros_like(v[..., :1]), torch.cumsum(0.5 * (v[..., 1:] + v[..., :-1]), -1)], -1)


def hat_basis(a0, M, device):
    """Piecewise-linear basis (B, M, 201) of the movable knots at a0 + 25 k, k = 1..M: zero below a0, held beyond the last knot."""
    s = S_GRID.to(device)[None, None]
    k = torch.arange(1, M + 1, device=device, dtype=torch.float32)[None, :, None]
    q_prev, q = a0[:, None, None] + KNOT_STEP * (k - 1), a0[:, None, None] + KNOT_STEP * k
    left = ((s - q_prev) / KNOT_STEP).clamp(0, 1)
    right = 1 - ((s - q) / KNOT_STEP).clamp(0, 1)
    right = torch.where(k == M, torch.ones_like(right), right)
    return left * right


class DeformAttn(nn.Module):
    """Single-scale deformable attention: each query samples `points` locations per head around its reference pixel."""

    def __init__(self, dim, heads=4, points=8, scale=0.15, canvas=(256, 1024)):
        super().__init__()
        self.h, self.p, self.scale, self.canvas = heads, points, scale, canvas
        self.off = nn.Linear(dim, heads * points * 2)
        self.wgt = nn.Linear(dim, heads * points)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.off.weight)
        ang = torch.arange(heads * points) * (2 * math.pi / points)
        self.off.bias.data = (torch.stack([ang.cos(), ang.sin()], -1) * torch.arange(1, points + 1).repeat(heads)[:, None] / points).flatten() * 0.3
        nn.init.zeros_(self.wgt.weight); nn.init.zeros_(self.wgt.bias)

    def forward(self, q, ref, val):
        """q (B,K,dim), ref (B,K,2) pixel (col,row) of the 128x1000 image, val (B,dim,H,W) covering the padded 256x1024 canvas."""
        B, K, D = q.shape
        Hf, Wf = val.shape[-2:]
        refn = torch.stack([ref[..., 0] * Wf / self.canvas[1] / (Wf - 1) * 2 - 1, ref[..., 1] * 2 * Hf / self.canvas[0] / (Hf - 1) * 2 - 1], -1)
        v = val.view(B * self.h, D // self.h, Hf, Wf)
        off = self.off(q).view(B, K, self.h, self.p, 2) * self.scale
        loc = (refn[:, :, None, None] + off).permute(0, 2, 1, 3, 4).reshape(B * self.h, K, self.p, 2)
        samp = F.grid_sample(v, loc.clamp(-1.2, 1.2), align_corners=True, padding_mode='zeros')
        w = self.wgt(q).view(B, K, self.h, self.p).softmax(-1).permute(0, 2, 1, 3).reshape(B * self.h, 1, K, self.p)
        out = (samp * w).sum(-1).view(B, self.h, D // self.h, K).permute(0, 3, 1, 2).reshape(B, K, D)
        return self.out(out)


class KnotLayer(nn.Module):
    def __init__(self, dim, heads, n_deform, drop=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.deform = nn.ModuleList([DeformAttn(dim, heads) for _ in range(n_deform)])
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * dim, dim))
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(dim) for _ in range(4))
        self.drop = nn.Dropout(drop)

    def forward(self, q, tokens, ref, feats):
        h = self.n1(q); q = q + self.drop(self.self_attn(h, h, h, need_weights=False)[0])
        h = self.n2(q); q = q + self.drop(self.cross(h, tokens, tokens, need_weights=False)[0])
        if self.deform:
            h = self.n3(q); q = q + self.drop(sum(m(h, ref, f) for m, f in zip(self.deform, feats)))
        return q + self.drop(self.ffn(self.n4(q)))


def conv_block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class Refiner(nn.Module):
    def __init__(self, knots, dim=256, heads=4, layers=3, M=7, n_iter=2, drop=0.1, az_min=220.0, az_step=0.1, width=1000, height=128):
        super().__init__()
        self.M, self.n_iter, self.dim = M, n_iter, dim
        self.register_buffer('knots', torch.tensor(knots, dtype=torch.float32))      # fixed decode grid (kappa rows, evidence)
        self.K = len(knots)
        kw = torch.zeros(ROWS + 1, self.K)
        for k in range(self.K):
            hat = np.zeros(self.K); hat[k] = 1.0
            kw[:, k] = torch.tensor(np.interp(S_GRID.numpy(), knots, hat), dtype=torch.float32)
        self.register_buffer('kap_w', kw)
        elev_grid = np.linspace(RING_TABLE.min() - 1, RING_TABLE.max() + 1, 4000)
        order = np.argsort(RING_TABLE)
        self.register_buffer('elev_lo', torch.tensor(elev_grid[0])); self.register_buffer('elev_step', torch.tensor(elev_grid[1] - elev_grid[0]))
        self.register_buffer('elev_row', torch.tensor(np.interp(elev_grid, RING_TABLE[order], order.astype(np.float64)), dtype=torch.float32))
        self.az_min, self.az_step, self.width, self.height = az_min, az_step, width, height
        self.enc_net = None
        cin = 3 + 16 + 1 + 2
        self.unroll_cnn = nn.Sequential(conv_block(cin, 64, (1, 2)), conv_block(64, 128, (2, 2)), conv_block(128, 128, (2, 2)))   # (200,40) -> (50,5)
        self.tok = nn.Linear(128 * 5 + 5, dim)
        self.tok_pe = nn.Parameter(torch.randn(50, dim) * 0.02)
        self.q_embed = nn.Parameter(torch.randn(M, dim) * 0.02)
        self.q_state = nn.Linear(8, dim)
        self.val_proj = nn.ModuleList([nn.Conv2d(16, dim, 1), nn.Conv2d(256, dim, 1)])
        self.layers = nn.ModuleList([KnotLayer(dim, heads, 2, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))
        self.glob = nn.Linear(dim, 2)
        self.register_buffer('glob_init', torch.log(torch.tensor([0.1, 0.005])))
        for lin in (self.head[-1], self.glob):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.register_buffer('prior_sigma', torch.tensor(3e-3))

    # ---- geometry
    def anchor(self, evidence):
        """Last knot with dense vote evidence (B,): the movable basis starts there."""
        dense = torch.expm1(evidence) > DENSE_VOTES
        pos = torch.where(dense, self.knots[None].expand_as(evidence), torch.zeros_like(evidence))
        return pos.max(1).values

    def kappa_rows(self, params):
        return params[:, 2:] @ self.kap_w.T

    def curve(self, x0, th0, kap):
        """Exact integration of curvature rows (B,201) -> x, heading, forward on d = 1..200."""
        th = th0[:, None] + cumtrapz(kap)
        x = x0[:, None] + cumtrapz(torch.sin(th))
        y = cumtrapz(torch.cos(th))
        return x[:, 1:], th[:, 1:], y[:, 1:]

    def project(self, P):
        az = torch.rad2deg(torch.atan2(P[..., 1], P[..., 0])) % 360.0
        col = ((az - self.az_min) % 360.0) / self.az_step
        elev = torch.rad2deg(torch.atan2(P[..., 2], torch.hypot(P[..., 0], P[..., 1])))
        idx = ((elev - self.elev_lo) / self.elev_step).clamp(0, len(self.elev_row) - 1).long()
        return torch.stack([col, self.elev_row[idx]], -1)

    def unroll(self, image, dec, x_c, y_c, th, z_c):
        """Scatter the scan into the curve frame: (B, 22, 200, 40)."""
        B = image.shape[0]
        px, py, pz = image[:, 1].flatten(1), -image[:, 2].flatten(1), image[:, 3].flatten(1)
        ok = (image[:, 5].flatten(1) > 0) & (py >= 0.5) & (py <= ROWS + 0.5)
        si = (py.round().long().clamp(1, ROWS) - 1)
        xc, yc, tc, zc = x_c.gather(1, si), y_c.gather(1, si), th.gather(1, si), z_c.gather(1, si)
        lat = (px - xc) * torch.cos(tc) + (py - yc) * torch.sin(tc)
        dz = pz - zc
        li = torch.floor((lat + LAT_HALF) / LAT_STEP).long()
        ok = ok & (li >= 0) & (li < N_LAT) & (dz > -1.5) & (dz < 6.0)
        band = torch.where(dz < H_BANDS[0], 0, torch.where(dz < H_BANDS[1], 1, 2))
        cell = (si * N_LAT + li).clamp(0, ROWS * N_LAT - 1)
        w = ok.float()
        occ = torch.zeros(B, 3 * ROWS * N_LAT, device=image.device).scatter_add_(1, band * ROWS * N_LAT + cell, w).view(B, 3, ROWS, N_LAT)
        cnt = occ.sum(1).flatten(1)
        f = F.avg_pool2d(dec, (2, 1))[:, :, :, :self.width].flatten(2) * w[:, None]
        fsum = torch.zeros(B, 16, ROWS * N_LAT, device=image.device).scatter_add_(2, cell[:, None].expand(-1, 16, -1), f)
        attr = torch.stack([dz, image[:, 4].flatten(1) / 255.0, torch.log1p(image[:, 0].flatten(1)) / 5.0], 1) * w[:, None]
        asum = torch.zeros(B, 3, ROWS * N_LAT, device=image.device).scatter_add_(2, cell[:, None].expand(-1, 3, -1), attr)
        den = cnt.clamp(min=1)[:, None]
        return torch.cat([torch.log1p(occ), (fsum / den).view(B, 16, ROWS, N_LAT),
                          (asum / den).view(B, 3, ROWS, N_LAT) * torch.tensor([1 / 3.0, 1.0, 1.0], device=image.device)[None, :, None, None]], 1)

    def step(self, image, dec, enc, x0, th0, kap, evidence, horizon, z, a0):
        B = kap.shape[0]
        x, th, y = self.curve(x0, th0, kap)
        hmap = self.unroll_cnn(self.unroll(image, dec, x, y, th, z))
        d = S_GRID.to(kap.device)[1:]
        ev_rows = torch.stack([torch.tensor(np.interp(d.cpu().numpy(), self.knots.cpu().numpy(), e.cpu().numpy()), dtype=torch.float32, device=kap.device) for e in evidence])
        geo = torch.stack([d.expand(B, -1) / ROWS, (d[None] <= horizon[:, None]).float(), ev_rows / 8.0, kap[:, 1:] * 100.0, x / 20.0], 1)
        geo = F.avg_pool1d(geo, 4)
        tokens = self.tok(torch.cat([hmap.permute(0, 2, 1, 3).flatten(2), geo.transpose(1, 2)], -1)) + self.tok_pe
        q_pos = a0[:, None] + KNOT_STEP * torch.arange(1, self.M + 1, device=kap.device, dtype=torch.float32)[None]
        qi = q_pos.clamp(1, ROWS).long() - 1
        xq, thq, zq, kq = (t.gather(1, qi) for t in (x, th, z, kap[:, 1:]))
        ref = self.project(torch.stack([xq, -y.gather(1, qi), zq], -1))
        state = torch.stack([q_pos / ROWS, torch.arange(1, self.M + 1, device=kap.device, dtype=torch.float32)[None].expand(B, -1) / self.M,
                             kq * 100.0, xq / 20.0, thq * 5.0, (q_pos <= ROWS).float(), horizon[:, None].expand(-1, self.M) / ROWS,
                             ev_rows.gather(1, (horizon.clamp(1, ROWS).long() - 1)[:, None]).expand(-1, self.M) / 8.0], -1)
        feats = [self.val_proj[0](F.avg_pool2d(dec, 2)), self.val_proj[1](enc)]
        q = self.q_embed[None] + self.q_state(state)
        for layer in self.layers:
            q = layer(q, tokens, ref, feats)
        out = self.head(self.norm(q))
        d_kappa = 2e-3 * torch.tanh(out[..., 0])
        log_sig_k = out[..., 1].clamp(-10, 4) - 8.0
        log_sig_g = self.glob(self.norm(q).mean(1)).clamp(-4, 4) + self.glob_init
        return d_kappa, log_sig_k, log_sig_g

    def forward(self, image, dec, enc, x0, th0, kap_in, evidence, horizon, z):
        """Per iteration: (cumulative d_kappa (B,M), log_sigma knots (B,M), log_sigma global (B,2), basis)."""
        a0 = self.anchor(evidence)
        basis = hat_basis(a0, self.M, kap_in.device)
        outs, dk_tot = [], torch.zeros(kap_in.shape[0], self.M, device=kap_in.device)
        for _ in range(self.n_iter):
            kap = kap_in + torch.einsum('bm,bms->bs', dk_tot, basis)
            dk, ls, lg = self.step(image, dec, enc, x0, th0, kap, evidence, horizon, z, a0)
            dk_tot = dk_tot + dk
            outs.append((dk_tot, ls, lg, basis))
        return outs

    def predict(self, x0, th0, kap_in, out):
        """Refined mean x(d), sigma(d) on d = 1..200 and the curvature rows."""
        dk, ls, lg, basis = out
        kap = kap_in + torch.einsum('bm,bms->bs', dk, basis)
        x, _, _ = self.curve(x0, th0, kap)
        A = cumtrapz(cumtrapz(basis))[:, :, 1:]
        d = S_GRID.to(kap.device)[1:]
        var = (A ** 2 * torch.exp(2 * ls)[:, :, None]).sum(1) + torch.exp(2 * lg[:, 0:1]) + torch.exp(2 * lg[:, 1:2]) * d[None] ** 2 + 0.15 ** 2
        return x, var.sqrt(), kap


def backbone_maps(backbone, image):
    """Frozen decoder map (B,16,256,1024) and the stride-16 encoder map (B,256,16,64) of the path model."""
    w = image.shape[-1]
    with torch.no_grad(), torch.autocast('cuda'):
        x = F.pad(backbone.normalise(image), (0, (-w) % 32))
        x = F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        enc = backbone.net.encoder(x)
        dec = backbone.net.decoder(enc)
    return dec.float(), enc[-2].float()


def run_model(model, backbone, b, kap_in=None):
    """One forward on a batch dict -> x0, th0, kap_in, outs."""
    x0, th0 = b['params'][:, 0], b['params'][:, 1]
    kap_in = model.kappa_rows(b['params']) if kap_in is None else kap_in
    dec, enc = backbone_maps(backbone, b['image'])
    return x0, th0, kap_in, model(b['image'], dec, enc, x0, th0, kap_in, b['evidence'], b['horizon'], b['z'])


def build(knots, weights, device):
    backbone = RailNet().to(device); load_weights(backbone, weights, device); backbone.eval()
    for q in backbone.parameters():
        q.requires_grad_(False)
    return backbone, Refiner(knots).to(device)


# ------------------------------------------------------------------ calibration of sigma
BANDS = (0.0, 25.0, 50.0, 75.0, np.inf)     # upper edges of "metres past the horizon"; rows inside the horizon are band 0


def band_index(d, horizon):
    return np.searchsorted(np.array(BANDS[:-1]), d - horizon, side='right')


def apply_calibration(sigma, horizon, calib):
    """sigma (ROWS,) -> calibrated sigma using the per-band scales of calib.json."""
    return sigma * np.array(calib['scale'])[band_index(np.arange(1, ROWS + 1), horizon)]


# ------------------------------------------------------------------ inference
class RefinerRunner:
    """Refine one decoded frame: refined x and calibrated sigma."""

    def __init__(self, run, knots, weights, device):
        self.device = device
        self.backbone, self.model = build(knots, weights, device)
        self.model.load_state_dict(torch.load(Path(run) / 'last.pt', map_location=device, weights_only=True)); self.model.eval()
        calib = Path(run) / 'calib.json'
        self.calib = json.loads(calib.read_text()) if calib.exists() else None

    def run(self, image, z, horizon, state):
        """Refined x(d) and calibrated sigma(d) for d = 1..200 m."""
        z_in = np.nan_to_num(z, nan=0.0).astype(np.float32)
        last = int(min(max(horizon + 20.0, 1.0), ROWS)) - 1
        z_in[last + 1:] = z_in[last]
        t = lambda a: torch.as_tensor(np.asarray(a, np.float32))[None].to(self.device)   # noqa: E731
        b = dict(image=t(image), params=t(state['params']), evidence=t(np.log1p(state['evidence'])), horizon=t(horizon)[0:1], z=t(z_in))
        with torch.no_grad():
            x0, th0, kap_in, outs = run_model(self.model, self.backbone, b)
            x, sigma, kap = self.model.predict(x0, th0, kap_in, outs[-1])
        x, sigma = x[0].cpu().numpy(), sigma[0].cpu().numpy()
        if self.calib:
            sigma = apply_calibration(sigma, horizon, self.calib)
        return x, sigma
