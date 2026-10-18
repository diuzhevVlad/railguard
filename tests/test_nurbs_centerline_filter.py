import unittest
from pathlib import Path

import numpy as np

from railpath.nurbs_centerline_filter import RollingNURBSCenterlineFilter
from railpath.centerline_filter import BayesianCenterlineFilter, CenterlineFilterConfig
from railpath.detection_config import ObstacleDetectionConfig
from railpath.obstacle_filter import RailPrediction
from railpath.occupancy_filter import ObstacleDetectionPipeline
from nurbs_filter.demo_dataset import score_frame, truth_anchors


DATASET_SCENE = (Path(__file__).resolve().parents[1] /
                 "outputs/centerline_dataset_all_generated/eval_new_data.npz")


def prediction(lateral: float) -> RailPrediction:
    distance = np.arange(1., 150., .25, dtype=np.float32)
    center = np.column_stack((np.full(len(distance), lateral),
                              -distance, np.full(len(distance), .3))).astype(np.float32)
    return RailPrediction(center, distance, details={})


class RollingNURBSCenterlineFilterTests(unittest.TestCase):
    def test_shared_pipeline_uses_rolling_nurbs(self):
        pipeline = ObstacleDetectionPipeline(centerline_filter_enabled=True)
        self.assertIsInstance(pipeline.centerline_filter, RollingNURBSCenterlineFilter)
        result = pipeline.update(1_000_000_000, np.empty((0, 3)), prediction(0.0))
        self.assertTrue(result.centerline_filtered)
        self.assertFalse(result.odometry_used)
        self.assertEqual(result.centerline_estimate.prediction.details[
            "centerline_filter_type"], "rolling_nurbs")

    def test_shared_pipeline_can_select_legacy_filter(self):
        config = CenterlineFilterConfig(algorithm="legacy_pointwise")
        pipeline = ObstacleDetectionPipeline(
            centerline_config=config, centerline_filter_enabled=True)
        self.assertIsInstance(pipeline.centerline_filter, BayesianCenterlineFilter)
        result = pipeline.update(1_000_000_000, np.empty((0, 3)), prediction(0.0))
        self.assertEqual(result.centerline_estimate.prediction.details[
            "centerline_filter_type"], "legacy_pointwise")

    def test_algorithm_round_trips_through_shared_parameters(self):
        params = ObstacleDetectionConfig().to_parameters()
        self.assertEqual(params["centerline_filter_algorithm"], "rolling_nurbs")
        self.assertEqual(params["centerline_nurbs_measurement_spacing_m"], 2.0)
        self.assertEqual(params["centerline_nurbs_behind_spans"], 3)
        self.assertTrue(params["centerline_nurbs_path_measurement_enabled"])
        self.assertEqual(params["centerline_nurbs_output_strength"], 0.2)
        params["centerline_filter_algorithm"] = "legacy_pointwise"
        config = ObstacleDetectionConfig.from_parameters(params)
        self.assertEqual(config.centerline.algorithm, "legacy_pointwise")
        with self.assertRaises(ValueError):
            CenterlineFilterConfig(algorithm="unknown")

    def test_temporal_update_smooths_lateral_shift_and_preserves_z(self):
        filt = RollingNURBSCenterlineFilter()
        filt.update(prediction(0.0), np.eye(4), odometry_used=True)
        second = prediction(0.5)
        result = filt.update(second, np.eye(4), odometry_used=True)
        middle = result.prediction.center[100:-100, 0]
        self.assertGreater(float(np.mean(middle)), 0.0)
        self.assertLess(float(np.mean(middle)), 0.5)
        np.testing.assert_array_equal(result.prediction.center[:, 2],
                                      second.center[:, 2])
        self.assertEqual(result.prediction.details["centerline_filter_type"],
                         "rolling_nurbs")
        self.assertEqual(len(result.variances_m2), len(second.center))

    def test_tuned_output_blends_detector_and_full_nurbs_curve(self):
        tuned = RollingNURBSCenterlineFilter()
        full = RollingNURBSCenterlineFilter(
            CenterlineFilterConfig(nurbs_output_strength=1.0))
        for filt in (tuned, full):
            filt.update(prediction(0.0), np.eye(4), odometry_used=True,
                        timestamp_ns=1)
        measured = prediction(0.5)
        tuned_center = tuned.update(
            measured, np.eye(4), odometry_used=True,
            timestamp_ns=2).prediction.center
        full_center = full.update(
            measured, np.eye(4), odometry_used=True,
            timestamp_ns=2).prediction.center
        np.testing.assert_allclose(
            tuned_center[:, 0],
            0.8 * measured.center[:, 0] + 0.2 * full_center[:, 0],
            rtol=0, atol=1e-6)

    def test_short_centerline_returns_measurement(self):
        filt = RollingNURBSCenterlineFilter()
        short = RailPrediction(np.array([[0., -1., 0.], [0., -2., 0.]],
                                        dtype=np.float32),
                               np.array([1., 2.], dtype=np.float32))
        estimate = filt.update(short)
        self.assertIs(estimate.prediction, short)
        self.assertEqual(len(estimate.variances_m2), 2)

    def test_nonmonotonic_timestamp_reinitializes_state(self):
        filt = RollingNURBSCenterlineFilter()
        filt.update(prediction(0.0), timestamp_ns=200)
        result = filt.update(prediction(0.1), timestamp_ns=100)
        self.assertEqual(result.prediction.details["centerline_filter_status"],
                         "initialized")

    def test_full_nurbs_output_replaces_far_detector_outlier(self):
        config = CenterlineFilterConfig(
            nurbs_path_measurement_enabled=False, nurbs_output_strength=1.0)
        filt = RollingNURBSCenterlineFilter(config)
        first = prediction(0.0)
        filt.update(first, np.eye(4), odometry_used=True, timestamp_ns=1)
        corrupted = first.center.copy()
        corrupted[first.distance_along_m > 80., 0] = 5.
        second = RailPrediction(corrupted, first.distance_along_m,
                                max_forward_m=150., details={})
        result = filt.update(second, np.eye(4), odometry_used=True,
                             timestamp_ns=2).prediction
        output = result.center
        self.assertLess(float(np.mean(output[first.distance_along_m > 100., 0])), 1.)
        np.testing.assert_array_equal(output[:, 1:], corrupted[:, 1:])
        self.assertTrue(np.all(np.diff(result.distance_along_m) > 0))
        self.assertLess(float(result.distance_along_m[-1]),
                        float(second.distance_along_m[-1]) + 2.)

    @unittest.skipUnless(DATASET_SCENE.is_file(), "exported scene not available")
    def test_runtime_matches_supplied_demo_scene(self):
        with np.load(DATASET_SCENE, allow_pickle=False) as file:
            arrays = {name: file[name] for name in file.files}
        filt = RollingNURBSCenterlineFilter(
            CenterlineFilterConfig(nurbs_output_strength=1.0))
        detected = arrays["detected_xyz_sensor_m"]
        detected_distance = arrays["detected_distance_along_m"]
        detected_offsets = arrays["detected_offsets"]
        gt = arrays["gt_xyz_sensor_m"]
        gt_distance = arrays["gt_distance_along_m"]
        gt_offsets = arrays["gt_offsets"]
        errors = []
        for frame, timestamp in enumerate(arrays["timestamps_ns"]):
            da, db = detected_offsets[frame:frame + 2]
            ga, gb = gt_offsets[frame:frame + 2]
            raw = detected[da:db]
            prediction = RailPrediction(
                raw, detected_distance[da:db], max_forward_m=150.)
            output = filt.update(
                prediction, arrays["sensor_pose_start"][frame],
                odometry_used=True, timestamp_ns=int(timestamp)).prediction.center
            anchors, normals = truth_anchors(
                gt[ga:gb], gt_distance[ga:gb], 150., 2.)
            visible = -raw[:, 1] <= 150.
            _, filtered_error, _ = score_frame(
                anchors, normals, raw[visible, :2], output[visible, :2])
            errors.append(filtered_error)
        pooled = np.concatenate(errors)
        self.assertEqual(len(pooled), 10_997)
        self.assertAlmostEqual(float(np.sqrt(np.mean(pooled**2))),
                               0.19134, places=4)


if __name__ == "__main__":
    unittest.main()
