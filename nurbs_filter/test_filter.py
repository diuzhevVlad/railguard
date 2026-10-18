"""Numerical invariants: run `python -m unittest -v test_filter.py`."""
import unittest
import numpy as np
if __package__:
    from .rolling_nurbs import RollingNURBSKF, Noise, rotation, sample_curve
else:
    from rolling_nurbs import RollingNURBSKF, Noise, rotation, sample_curve


class FilterTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(11)
        cp = np.column_stack([np.arange(15)*2., rng.normal(size=15)])
        a = rng.normal(size=(30, 30))
        return RollingNURBSKF(cp, 2., weights=np.exp(rng.normal(0, .2, 15)),
                              covariance=.01*a@a.T,
                              noise=Noise(process_per_sqrt_second=0.))

    def test_exact_stationarity_under_odometry(self):
        f = self.fixture()
        u = np.linspace(*f.domain, 70)
        world = f.evaluate(u)
        p0 = f.P.copy()
        translation, yaw = np.array([1.3, -.4]), .28
        f.predict(translation, yaw)
        np.testing.assert_allclose(f.evaluate(u) @ rotation(yaw).T+translation,
                                   world, atol=1e-12)
        g = np.kron(np.eye(f.n), rotation(yaw).T)
        np.testing.assert_allclose(f.P, g @ p0 @ g.T, atol=1e-12)

    def test_roll_preserves_geometry_and_covariance_on_overlap(self):
        for direction in [-1, 1]:
            f = self.fixture()
            lo, hi = f.domain
            u = np.linspace(lo+(2.01 if direction == 1 else .01),
                            hi-(2.01 if direction == -1 else .01), 70)
            c = f.evaluate(u)
            h = np.kron(f.basis(u), np.eye(2))
            cov = h @ f.P @ h.T
            f.roll(direction)
            np.testing.assert_allclose(f.evaluate(u), c, atol=1e-12)
            h = np.kron(f.basis(u), np.eye(2))
            np.testing.assert_allclose(h @ f.P @ h.T, cov, atol=1e-12)

    def test_odometry_jacobian(self):
        f = self.fixture()
        old = f.cp.copy()
        t, yaw = np.array([.4, -.2]), .12
        f.predict(t, yaw)
        g = np.zeros((2*f.n, 3))
        for i, p in enumerate(f.cp):
            g[2*i:2*i+2, :2] = -rotation(yaw).T
            g[2*i:2*i+2, 2] = [p[1], -p[0]]
        for j in range(3):
            d = np.zeros(3)
            d[j] = 1e-6
            plus = (old-(t+d[:2])) @ rotation(yaw+d[2])
            minus = (old-(t-d[:2])) @ rotation(yaw-d[2])
            np.testing.assert_allclose(((plus-minus)/2e-6).ravel(), g[:, j], atol=1e-8)

    def test_common_odometry_noise_is_correlated(self):
        f = self.fixture()
        f.P[:] = 0
        f.predict([0, 0], 0, odometry_cov=np.diag([.1, .2, 0]))
        np.testing.assert_allclose(f.P[:2, -2:], np.diag([.1, .2]))

    def test_detector_reparametrization(self):
        def curve(t):
            return np.column_stack([25*t, np.sin(2*t)])
        a = sample_curve(curve)
        b = sample_curve(lambda t: curve(t**2.3))
        np.testing.assert_allclose(a, b, atol=5e-6)

    def test_filter_reduces_normal_error_and_keeps_psd(self):
        q = np.column_stack([np.linspace(-8, 28, 70), np.zeros(70)])
        f = RollingNURBSKF.from_points(q)
        f.cp[:, 1] += .25
        result = f.update_points(q)
        self.assertGreater(result['accepted'], 60)
        self.assertLess(np.mean(abs(f.evaluate(np.linspace(2, 30, 50))[:, 1])), .25)
        self.assertGreater(np.linalg.eigvalsh(f.P).min(), -1e-10)

    def test_sensor_path_position_is_soft_lateral_measurement(self):
        cp = np.column_stack([np.full(15, .8), 15.-5.*np.arange(15)])
        f = RollingNURBSKF(cp, spacing=5.)
        before = f.evaluate([f.project([[0., 0.]])[0]])[0, 0]

        result = f.update_path_position(lateral_sigma=.3)

        after = f.evaluate([f.project([[0., 0.]])[0]])[0, 0]
        self.assertEqual(result['accepted'], 1)
        self.assertLess(abs(after), abs(before))
        self.assertGreater(abs(after), 0.)  # The observation has finite uncertainty.
        self.assertGreaterEqual(np.linalg.eigvalsh(f.P).min(), -1e-10)
        rejected = f.update_path_position([20., 0.], max_distance=2.)
        self.assertEqual(rejected['accepted'], 0)

    def test_robust_detector_update_downweights_large_residual(self):
        clean = np.column_stack([np.linspace(-8., 28., 70), np.zeros(70)])
        observed = clean.copy()
        observed[35, 1] = 5.
        ordinary = RollingNURBSKF.from_points(clean)
        robust = RollingNURBSKF.from_points(clean)

        ordinary.update_points(observed, max_distance=20., gate=100.)
        robust.update_points(observed, max_distance=20., gate=100.,
                             robust_threshold=1.)

        u = ordinary.project(clean[35:36])
        ordinary_offset = ordinary.evaluate(u)[0, 1]
        robust_offset = robust.evaluate(u)[0, 1]
        self.assertLess(abs(robust_offset), .1*abs(ordinary_offset))
        self.assertGreaterEqual(np.linalg.eigvalsh(robust.P).min(), -1e-10)


if __name__ == '__main__':
    unittest.main()
