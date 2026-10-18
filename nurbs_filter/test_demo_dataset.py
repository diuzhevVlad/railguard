"""Checks for the dataset demo's forward-range constraint."""

import unittest

import numpy as np

if __package__:
    from .forward_filter import detector_lateral_change, enforce_forward_reach
    from .rolling_nurbs import RollingNURBSKF
else:
    from forward_filter import detector_lateral_change, enforce_forward_reach
    from rolling_nurbs import RollingNURBSKF


class ForwardReachTests(unittest.TestCase):
    def test_constraint_reaches_150_m_and_preserves_near_controls(self):
        controls = np.column_stack([np.zeros(24), -np.arange(24)*4.])
        filt = RollingNURBSKF(controls, spacing=7.)
        near_before = filt.cp[:-6].copy()

        correction = enforce_forward_reach(filt, 150.01)

        self.assertGreater(correction, 0.)
        self.assertGreaterEqual(-filt.evaluate([filt.domain[1]])[0, 1], 150.)
        np.testing.assert_array_equal(filt.cp[:-6], near_before)
        np.testing.assert_allclose(filt.P, filt.P.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(filt.P).min(), -1e-12)
        self.assertEqual(enforce_forward_reach(filt, 150.), 0.)

    def test_detector_change_uses_pose_alignment(self):
        old = np.column_stack([np.zeros(151), -np.arange(151.), np.zeros(151)])
        old_pose = np.eye(4)
        current_pose = np.eye(4)
        current_pose[1, 3] = -2.
        current = old.copy()
        current[:, 1] += 2.
        targets = np.arange(10., 140., 5.)

        self.assertAlmostEqual(detector_lateral_change(
            current, old, current_pose, old_pose, targets), 0.)
        current[:, 0] += 2.
        self.assertAlmostEqual(detector_lateral_change(
            current, old, current_pose, old_pose, targets), 2.)
        self.assertTrue(np.isinf(detector_lateral_change(
            current[::-1], old, current_pose, old_pose, targets)))


if __name__ == "__main__":
    unittest.main()
