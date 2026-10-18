"""Forward-window operations shared by the dataset demo and RailGuard runtime."""

from __future__ import annotations

import numpy as np

if __package__:
    from .rolling_nurbs import Noise, RollingNURBSKF
else:  # Support ``python demo_dataset.py`` from this directory.
    from rolling_nurbs import Noise, RollingNURBSKF


def initialize_forward_window(measured: np.ndarray, n: int = 24,
                              behind_spans: int = 3,
                              noise: Noise | None = None) -> RollingNURBSKF:
    """Fit observed points, then add uncertain controls behind the sensor."""
    seed = RollingNURBSKF.from_points(measured, n=n - behind_spans, noise=noise)
    transform = np.zeros((n, seed.n))
    transform[behind_spans:] = np.eye(seed.n)
    birth = np.zeros((n, behind_spans))
    for row in range(behind_spans):
        distance = behind_spans - row
        transform[row, :2] = [1 + distance, -distance]
        birth[row] = np.maximum(distance - np.arange(behind_spans), 0)
    t = np.kron(transform, np.eye(2))
    b = np.kron(birth, np.eye(2))
    covariance = t @ seed.P @ t.T + seed.noise.birth**2 * (b @ b.T)
    return RollingNURBSKF(transform @ seed.cp, seed.spacing,
                          degree=seed.degree, covariance=covariance,
                          base=seed.base - behind_spans, noise=seed.noise)


def sample_forward_curve(xyz: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Sample an ordered detection at fixed sensor-forward (-Y) distances."""
    xy = np.asarray(xyz[:, :2], float)
    if len(xy) < 2 or not np.isfinite(xy).all():
        return np.empty((0, 2))
    forward = -xy[:, 1]
    if np.any(np.diff(forward) <= 0):
        return np.empty((0, 2))
    lateral = np.interp(targets, forward, xy[:, 0], left=np.nan, right=np.nan)
    return np.column_stack([lateral, -targets])


def detector_lateral_change(current_xyz: np.ndarray, old_xyz: np.ndarray,
                            current_pose: np.ndarray, old_pose: np.ndarray,
                            targets: np.ndarray) -> float:
    """RMS detector change after placing an older detection in the current frame."""
    old_world = old_xyz @ old_pose[:3, :3].T + old_pose[:3, 3]
    old_current = (old_world - current_pose[:3, 3]) @ current_pose[:3, :3]
    current_sample = sample_forward_curve(current_xyz, targets)
    old_sample = sample_forward_curve(old_current, targets)
    if len(current_sample) != len(targets) or len(old_sample) != len(targets):
        return np.inf
    good = np.isfinite(current_sample[:, 0]) & np.isfinite(old_sample[:, 0])
    if good.sum() < max(10, len(targets) // 2):
        return np.inf
    return float(np.sqrt(np.mean((current_sample[good, 0] -
                                  old_sample[good, 0])**2)))


def enforce_forward_reach(filt: RollingNURBSKF, minimum_forward_m: float,
                          tail_controls: int = 6) -> float:
    """Extend the far endpoint without changing the near control points."""
    endpoint_basis = filt.basis([filt.domain[1]])[0]
    endpoint_forward = float(-(endpoint_basis @ filt.cp[:, 1]))
    deficit = minimum_forward_m - endpoint_forward
    if deficit <= 0:
        return 0.
    count = min(tail_controls, filt.n)
    ramp = np.zeros(filt.n)
    ramp[-count:] = np.linspace(0., 1., count)
    response = float(endpoint_basis @ ramp)
    if response <= 1e-12:
        raise RuntimeError("Could not constrain the forward endpoint")
    direction = np.zeros(2 * filt.n)
    direction[1::2] = ramp / response
    observation = np.zeros(2 * filt.n)
    observation[1::2] = endpoint_basis
    filt.cp[:, 1] -= deficit * ramp / response
    transform = np.eye(2 * filt.n) - np.outer(direction, observation)
    filt.P = transform @ filt.P @ transform.T + .5**2 * np.outer(direction, direction)
    filt.P = (filt.P + filt.P.T) / 2
    return float(deficit)
