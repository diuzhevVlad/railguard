"""2-D moving-frame NURBS Kalman filter with a fixed-size control-point window.

Dependencies: numpy, scipy. See README.md for conventions and limitations.
State ordering: [P0.x, P0.y, P1.x, P1.y, ...]. No world-frame map is stored.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def sample_curve(curve, parameter_range=(0., 1.), count=60, dense_count=1000):
    """Accept any vectorized callable curve(t)->(len(t),2).

    Resample by approximate arc length; the detector's parameter is discarded.
    These samples are correlated observations, NOT independent measurements.
    """
    t = np.linspace(*parameter_range, dense_count)
    q = np.asarray(curve(t), dtype=float)
    if q.shape != (dense_count, 2) or not np.isfinite(q).all():
        raise ValueError('curve(t) must return finite (len(t), 2) points')
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-12]
    if s[-1] <= 1e-9:
        raise ValueError('Degenerate measurement curve')
    target = np.linspace(0., s[-1], count)
    return np.column_stack([np.interp(target, s[keep], q[keep, j]) for j in range(2)])


@dataclass
class Noise:
    # All standard deviations below are in the position unit (e.g. metres).
    white: float = .035
    frame_translation: float = .12
    smooth_shape: float = .07
    correlation_length: float = 4.
    process_per_sqrt_second: float = .003
    birth: float = .25  # uncertainty in a new CP's second difference


class RollingNURBSKF:
    def __init__(self, control_points, spacing, degree=3, weights=None,
                 covariance=None, base=None, noise=None):
        self.cp = np.asarray(control_points, float).copy()
        if self.cp.ndim != 2 or self.cp.shape[1] != 2:
            raise ValueError('control_points must have shape (n,2)')
        self.n = len(self.cp)
        self.degree = int(degree)
        self.spacing = float(spacing)
        if self.degree < 1 or self.n < self.degree + 2 or self.spacing <= 0:
            raise ValueError('Need degree>=1, n>=degree+2 and spacing>0')
        self.base = -self.degree if base is None else int(base)
        self.weights = np.ones(self.n) if weights is None else np.asarray(weights, float).copy()
        if self.weights.shape != (self.n,) or np.any(self.weights <= 0):
            raise ValueError('Need one positive fixed weight per control point')
        self.P = np.eye(2*self.n)*.3**2 if covariance is None else np.asarray(covariance, float).copy()
        if self.P.shape != (2*self.n, 2*self.n):
            raise ValueError('Incorrect covariance shape')
        self.noise = Noise() if noise is None else noise
        self.total_rolls = 0
        self._basis_cache = None

    @property
    def knots(self):
        # Non-clamped restriction of one persistent, uniform knot grid.
        return (self.base + np.arange(self.n+self.degree+1))*self.spacing

    @property
    def domain(self):
        return ((self.base+self.degree)*self.spacing,
                (self.base+self.n)*self.spacing)

    @property
    def control_ids(self):
        return self.base + np.arange(self.n)

    def basis(self, u, derivative=False):
        u = np.atleast_1d(np.asarray(u, float))
        if self._basis_cache is None:
            self._basis_cache = BSpline(self.knots, np.eye(self.n), self.degree,
                                       extrapolate=False)
        b = self._basis_cache(u)
        bw = b*self.weights
        den = bw.sum(axis=1, keepdims=True)
        if not derivative:
            return bw/den
        dbw = self._basis_cache(u, nu=1)*self.weights
        return (dbw*den-bw*dbw.sum(axis=1, keepdims=True))/den**2

    def evaluate(self, u):
        return self.basis(u) @ self.cp

    @classmethod
    def from_points(cls, points, n=18, degree=3, noise=None):
        """Initialize once from an ordered, reasonably clean first detection.

        Orientation defines increasing u. Arc length is an initial gauge only;
        it is never re-normalized on subsequent detections.
        """
        q = np.asarray(points, float)
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        if len(q) < n or s[-1] <= 1e-6:
            raise ValueError('Need at least n ordered points on a nonzero curve')
        obj = cls(np.zeros((n, 2)), s[-1]/(n-degree), degree, noise=noise)
        # Keep the endpoint exactly inside the floating-point base interval.
        s[-1] = obj.domain[1]
        b = obj.basis(s)
        d2 = np.diff(np.eye(n), n=2, axis=0)
        # Weak curvature regularization stabilizes the outer control points.
        obj.cp = np.linalg.lstsq(np.vstack([b, .02*d2]),
                                np.vstack([q, np.zeros((n-2, 2))]), rcond=None)[0]
        return obj

    def predict(self, translation, yaw, dt=1., odometry_cov=None):
        """Odometry convention: p_old = R(yaw) @ p_new + translation.

        translation is the NEW frame origin expressed in the OLD frame.
        odometry_cov: covariance of [translation_x,translation_y,yaw],
        additive errors in precisely those coordinates. Increment noise is
        assumed independent of the old state (not a joint SLAM estimator).
        """
        if dt < 0:
            raise ValueError('dt must be nonnegative')
        a = rotation(yaw).T
        self.cp = (self.cp-np.asarray(translation)) @ a.T
        f = np.kron(np.eye(self.n), a)
        self.P = f @ self.P @ f.T
        if odometry_cov is not None:
            # Shared pose error correlates EVERY control point.
            g = np.zeros((2*self.n, 3))
            for i, p in enumerate(self.cp):
                g[2*i:2*i+2, :2] = -a
                g[2*i:2*i+2, 2] = [p[1], -p[0]]
            self.P += g @ np.asarray(odometry_cov) @ g.T
        self.P += np.eye(2*self.n)*self.noise.process_per_sqrt_second**2*dt
        self.P = (self.P+self.P.T)/2

    def roll(self, direction=1):
        """Discard one CP, retain its covariance marginal, birth one new CP.

        Exact geometry AND curve covariance on the common valid interval.
        New CP prior: P_new = 2 P_end - P_penultimate + acceleration noise.
        Boundary weights are continued as constants. Returns no world map.
        """
        if direction not in (-1, 1):
            raise ValueError('direction must be +1 or -1')
        s = np.zeros((self.n, self.n))
        if direction == 1:
            s[:-1, 1:] = np.eye(self.n-1)
            s[-1, -2:] = [-1., 2.]
            self.weights = np.r_[self.weights[1:], self.weights[-1]]
            newborn = self.n-1
        else:
            s[1:, :-1] = np.eye(self.n-1)
            s[0, :2] = [2., -1.]
            self.weights = np.r_[self.weights[0], self.weights[:-1]]
            newborn = 0
        transform = np.kron(s, np.eye(2))
        self.cp = s @ self.cp
        self.P = transform @ self.P @ transform.T
        j = slice(2*newborn, 2*newborn+2)
        self.P[j, j] += np.eye(2)*self.noise.birth**2
        self.base += direction
        self.total_rolls += 1
        self._basis_cache = None

    def project(self, points, grid_count=250):
        """Nearest-point association. Intended for simple, nonbranching curves.

        Grid initialization plus local bounded scalar minimization; no global
        correspondence guarantee at self intersections or nearby branches.
        """
        q = np.atleast_2d(np.asarray(points, float))
        grid = np.linspace(*self.domain, grid_count)
        c = self.evaluate(grid)
        near = np.argmin(((q[:, None]-c[None])**2).sum(axis=2), axis=1)
        u = np.empty(len(q))
        for i, j in enumerate(near):
            lo, hi = grid[max(0, j-1)], grid[min(grid_count-1, j+1)]
            result = minimize_scalar(lambda v: np.sum((self.evaluate([v])[0]-q[i])**2),
                                     bounds=(lo, hi), method='bounded',
                                     options={'xatol': 1e-8})
            u[i] = result.x
        return u

    def recenter(self, reference=(0., 0.), anchor_fraction=.25, max_shifts=None):
        """Roll until the nearest curve point to reference is near the anchor.

        reference is in the current moving frame (usually the sensor origin).
        Call after predict(), before update(). Requires motion per frame small
        enough that the reference still projects onto the old valid window.
        """
        if not 0 < anchor_fraction < 1:
            raise ValueError('anchor_fraction must be between zero and one')
        u = self.project([reference])[0]
        lo, hi = self.domain
        if min(u-lo, hi-u) < 1e-4*self.spacing:
            raise RuntimeError('Reference left window; reinitialize or use an external anchor parameter')
        shift = int(np.rint((u-(lo+anchor_fraction*(hi-lo)))/self.spacing))
        max_shifts = self.n-self.degree-1 if max_shifts is None else int(max_shifts)
        if abs(shift) > max_shifts:
            raise RuntimeError('Motion too large for rolling association; reinitialize')
        for _ in range(abs(shift)):
            self.roll(1 if shift > 0 else -1)
        return shift

    def update_points(self, points, covariance_xy=None, max_distance=.9, gate=5.,
                      robust_threshold=None):
        """One frozen-association EKF / point-to-normal Kalman correction.

        covariance_xy optionally gives the FULL (2m,2m) covariance of the
        sampled curve in interleaved coordinates. Otherwise use the correlated
        detector model in Noise. Do not reuse this frame as independent data.
        If robust_threshold is set, large normal residuals receive extra
        independent variance, limiting the effect of detector outliers.
        """
        q = np.asarray(points, float)
        if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
            raise ValueError('points must be finite (m,2)')
        if robust_threshold is not None and (not np.isfinite(robust_threshold) or
                                             robust_threshold <= 0):
            raise ValueError('robust_threshold must be positive and finite')
        u = self.project(q)
        b = self.basis(u)
        c = b @ self.cp
        tangent = self.basis(u, derivative=True) @ self.cp
        speed = np.linalg.norm(tangent, axis=1)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / np.maximum(speed[:, None], 1e-12)
        h = (b[:, :, None]*normal[:, None, :]).reshape(len(q), 2*self.n)
        residual = np.einsum('ij,ij->i', normal, q-c)
        if covariance_xy is None:
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
            length = self.noise.correlation_length
            if length <= 0:
                raise ValueError('correlation_length must be positive')
            # Shared XY translation + smooth isotropic vector-field jitter.
            dots = normal @ normal.T
            kernel = np.exp(-.5*((arc[:, None]-arc[None, :])/length)**2)
            r = (self.noise.white**2*np.eye(len(q)) +
                 self.noise.frame_translation**2*dots +
                 self.noise.smooth_shape**2*kernel*dots)
        else:
            cov = np.asarray(covariance_xy, float)
            if cov.shape != (2*len(q), 2*len(q)):
                raise ValueError('covariance_xy must be (2m,2m)')
            projection = np.zeros((len(q), 2*len(q)))
            for i in range(len(q)):
                projection[i, 2*i:2*i+2] = normal[i]
            r = projection @ cov @ projection.T
        innovation_var = np.einsum('ij,jk,ik->i', h, self.P, h)+np.diag(r)
        lo, hi = self.domain
        keep = ((u > lo+1e-3*self.spacing) & (u < hi-1e-3*self.spacing) &
                (speed > 1e-8) & (np.linalg.norm(q-c, axis=1) < max_distance) &
                (residual**2 < gate**2*innovation_var))
        if not np.any(keep):
            return {'accepted': 0, 'total': len(q), 'rms_normal_residual': np.nan}
        h, residual, r = h[keep], residual[keep], r[np.ix_(keep, keep)]
        if robust_threshold is not None:
            extra = np.maximum(residual**2-robust_threshold**2, 0.)
            r = r + np.diag(extra)
        s = h @ self.P @ h.T+r
        k = cho_solve(cho_factor(s, lower=True), h @ self.P).T
        self.cp += (k @ residual).reshape(self.n, 2)
        ikh = np.eye(2*self.n)-k @ h
        self.P = ikh @ self.P @ ikh.T+k @ r @ k.T  # Joseph form
        self.P = (self.P+self.P.T)/2
        return {'accepted': int(keep.sum()), 'total': len(q),
                'rms_normal_residual': float(np.sqrt(np.mean(residual**2)))}

    def update_path_position(self, position=(0., 0.), lateral_sigma=.5,
                             max_distance=2., gate=5.):
        """Treat one sensor-path position as a soft point on the curve.

        Only the curve-normal offset is observed. This does not constrain the
        curve's parameter or force the sensor to lie exactly on it. Call once
        for each NEW position; replaying old positions would double-count them.
        """
        q = np.asarray(position, float)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError('position must be a finite 2-D point')
        if not np.isfinite(lateral_sigma) or lateral_sigma <= 0:
            raise ValueError('lateral_sigma must be positive and finite')
        if not np.isfinite(max_distance) or max_distance <= 0 or not np.isfinite(gate) or gate <= 0:
            raise ValueError('max_distance and gate must be positive and finite')
        u = self.project(q[None])[0]
        lo, hi = self.domain
        b = self.basis([u])[0]
        c = b @ self.cp
        tangent = self.basis([u], derivative=True)[0] @ self.cp
        speed = np.linalg.norm(tangent)
        if (u <= lo+1e-3*self.spacing or u >= hi-1e-3*self.spacing or
                speed <= 1e-8 or np.linalg.norm(q-c) >= max_distance):
            return {'accepted': 0, 'normal_residual': np.nan}
        normal = np.array([-tangent[1], tangent[0]]) / speed
        h = (b[:, None]*normal).ravel()
        residual = float(normal @ (q-c))
        r = lateral_sigma**2
        s = float(h @ self.P @ h+r)
        if residual**2 >= gate**2*s:
            return {'accepted': 0, 'normal_residual': residual}
        k = self.P @ h / s
        self.cp += (k*residual).reshape(self.n, 2)
        ikh = np.eye(2*self.n)-np.outer(k, h)
        self.P = ikh @ self.P @ ikh.T+r*np.outer(k, k)
        self.P = (self.P+self.P.T)/2
        return {'accepted': 1, 'normal_residual': residual}

    def update_curve(self, curve, parameter_range=(0., 1.), count=60, **kwargs):
        return self.update_points(sample_curve(curve, parameter_range, count), **kwargs)
