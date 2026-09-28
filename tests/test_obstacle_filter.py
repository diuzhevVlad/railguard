import unittest
import numpy as np
from obstacle_app.obstacle_filter import ObstacleFilterConfig, filter_obstacle_candidates


class ObstacleFilterTests(unittest.TestCase):
    def test_reject_single_and_pair_preserve_chain_and_input_order(self):
        points = np.array([[0,-10,0], [.15,-10,0], [.3,-10,0],
                           [2,-10,0], [4,-10,0], [4.1,-10,0]], dtype=float)
        before = points.copy()
        np.testing.assert_array_equal(filter_obstacle_candidates(points, ObstacleFilterConfig(min_cluster_points=3)),
                                      [True, True, True, False, False, False])
        np.testing.assert_array_equal(points, before)
        permutation = [5,2,3,0,4,1]
        np.testing.assert_array_equal(filter_obstacle_candidates(points[permutation], ObstacleFilterConfig(min_cluster_points=3)),
                                      np.array([True,True,True,False,False,False])[permutation])

    def test_radius_grows_smoothly_with_distance_and_uses_3d(self):
        points = np.array([[0,-10,0], [.4,-10,0], [.8,-10,0],
                           [0,-140,0], [.4,-140,0], [.8,-140,0],
                           [0,-20,0], [0,-20,1], [0,-20,2]])
        np.testing.assert_array_equal(filter_obstacle_candidates(points, ObstacleFilterConfig(min_cluster_points=3)),
                                      [False]*3 + [True]*3 + [False]*3)
        knots = ObstacleFilterConfig()
        self.assertAlmostEqual(np.interp(55,knots.distance_knots_m,knots.radius_knots_m),.275)

    def test_empty_duplicates_and_config(self):
        self.assertEqual(len(filter_obstacle_candidates(np.empty((0,3)))),0)
        self.assertTrue(filter_obstacle_candidates(np.zeros((5,3))).all())
        self.assertFalse(filter_obstacle_candidates(np.zeros((3,3)),
                         ObstacleFilterConfig(min_cluster_points=4)).any())
        with self.assertRaises(ValueError):ObstacleFilterConfig(min_cluster_points=0)
        with self.assertRaises(ValueError):ObstacleFilterConfig(radius_knots_m=(.2,))
        with self.assertRaises(ValueError):filter_obstacle_candidates([[np.nan,0,0]])

    def test_low_nonrail_objects_remain_eligible_without_height_tolerance(self):
        from obstacle_app.detection import RailPrediction, detect_obstacle_points
        depth = np.arange(1,151,dtype=np.float32)
        prediction = RailPrediction(np.column_stack((np.zeros_like(depth), -depth,
                                      np.full_like(depth, -1.1))), depth)
        low = np.array([[0,-2,-1.09],[.02,-2,-1.08],[.04,-2,-1.07]])
        found = detect_obstacle_points(low,prediction,ObstacleFilterConfig(min_cluster_points=3,min_height_above_track_m=0,near_min_height_above_track_m=0))
        np.testing.assert_allclose(found.points,low)
        self.assertAlmostEqual(found.distance_m,2)
        self.assertEqual(len(found.rejected_points),0)

    def test_segmented_rails_excluded_without_erasing_nearby_objects(self):
        from obstacle_app.detection import RailPrediction, detect_obstacle_points
        from obstacle_app.scan_detection import rejected_payload
        depth = np.arange(1,151,dtype=np.float32)
        rail = np.array([[0,-2,-.8], [.02,-2,-.8], [.04,-2,-.8]],dtype=np.float32)
        nearby_object = rail + [0,0,.02]
        prediction = RailPrediction(np.column_stack((np.zeros_like(depth),-depth,
                                      np.full_like(depth,-1.1))),depth,rail_points=rail)
        only_rails = detect_obstacle_points(rail,prediction)
        self.assertFalse(only_rails.detected)
        self.assertEqual(only_rails.excluded_rail_count,3)
        self.assertEqual(len(only_rails.rejected_points),0)
        combined = detect_obstacle_points(np.concatenate((rail,nearby_object)),prediction,ObstacleFilterConfig(min_cluster_points=3))
        np.testing.assert_allclose(combined.points,nearby_object)
        self.assertEqual(combined.excluded_rail_count,3)
        self.assertEqual(rejected_payload(combined)['excluded_rail_points'],3)
        # A segmented rail cannot supply the third point to a two-point object.
        small = detect_obstacle_points(np.concatenate((rail,nearby_object[:2])),prediction)
        self.assertFalse(small.detected)
        self.assertEqual(len(small.rejected_points),2)

    def test_default_rejects_four_and_keeps_five_point_clusters(self):
        points = np.array([[i*.1,-10,0] for i in range(4)] +
                          [[i*.1,-20,0] for i in range(5)])
        np.testing.assert_array_equal(filter_obstacle_candidates(points),
                                      [False]*4 + [True]*5)

    def test_capped_graph_matches_full_graph_including_variable_radius_boundaries(self):
        from scipy.spatial.distance import cdist
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix
        rng = np.random.default_rng(42)
        for minimum in [2,3,5,10]:
            config = ObstacleFilterConfig(min_cluster_points=minimum,
                distance_knots_m=(0.,1.,2.,3.), radius_knots_m=(.05,.7,.1,.6))
            for _ in range(10):
                points = rng.uniform([0,-3,0],[1,0,1],size=(100,3))
                radii = np.interp(-points[:,1],config.distance_knots_m,config.radius_knots_m)
                graph = cdist(points,points) <= np.minimum(radii[:,None],radii[None,:])
                _,labels = connected_components(csr_matrix(graph),directed=False)
                expected = np.bincount(labels)[labels]>=minimum
                np.testing.assert_array_equal(filter_obstacle_candidates(points,config),expected)

    def test_dense_contact_cloud_keeps_all_returns(self):
        rng = np.random.default_rng(4)
        points = rng.uniform([0,-2,0],[.01,-1.99,.01],size=(10000,3))
        self.assertTrue(filter_obstacle_candidates(points).all())

    def test_default_height_threshold_removes_low_clusters(self):
        from obstacle_app.detection import RailPrediction, detect_obstacle_points
        depth = np.arange(1,151,dtype=np.float32)
        prediction = RailPrediction(np.column_stack((np.zeros_like(depth),-depth,
                                      np.zeros_like(depth))),depth)
        low = np.array([[i*.02,-2,.05] for i in range(5)])
        high = np.array([[i*.02,-20,.11] for i in range(5)])
        found = detect_obstacle_points(np.concatenate((low,high)),prediction)
        np.testing.assert_allclose(found.points,high)
        np.testing.assert_allclose(found.rejected_points,low)
        self.assertAlmostEqual(found.distance_m,20)
        self.assertFalse(detect_obstacle_points(low,prediction).detected)
        with self.assertRaises(ValueError):ObstacleFilterConfig(min_height_above_track_m=-1)

    def test_sparse_points_attached_to_dense_cell_remain_accepted(self):
        # Four extra points straddle cells and connect to a proven dense cell.
        points = np.array([[0,-10,i*.001] for i in range(5)] +
                          [[.19,-10,0],[.38,-10,0],[.57,-10,0],[.76,-10,0],
                           [2,-10,0],[2.1,-10,0]])
        np.testing.assert_array_equal(filter_obstacle_candidates(points),[True]*9+[False]*2)

    def test_min_forward_boundary_and_clustering_without_near_sensor_support(self):
        from obstacle_app.detection import RailPrediction, detect_obstacle_points
        depth = np.arange(1,151,.25,dtype=np.float32)
        prediction = RailPrediction(np.column_stack((np.zeros_like(depth),-depth,
                                      np.zeros_like(depth))),depth)
        close = np.array([[i*.01,-1.49,.3] for i in range(5)])
        boundary = np.array([[i*.01,-1.5,.3] for i in range(5)])
        far = np.array([[i*.01,-10,.3] for i in range(5)])
        result = detect_obstacle_points(np.concatenate((close,boundary,far)),prediction)
        np.testing.assert_allclose(result.points,np.concatenate((boundary,far)))
        self.assertAlmostEqual(result.distance_m,1.5)
        self.assertFalse(detect_obstacle_points(close,prediction).detected)
        self.assertFalse(detect_obstacle_points(np.concatenate((close,boundary[:4])),prediction).detected)
        self.assertTrue(detect_obstacle_points(close,prediction,
            ObstacleFilterConfig(min_forward_m=0)).detected)
        with self.assertRaises(ValueError):ObstacleFilterConfig(min_forward_m=-1)


    def test_near_height_and_exact_ten_meter_boundary(self):
        from obstacle_app.detection import RailPrediction, detect_obstacle_points
        depth=np.arange(1,151,.1,dtype=np.float32)
        prediction=RailPrediction(np.column_stack((np.zeros_like(depth),-depth,np.zeros_like(depth))),depth)
        groups=[np.array([[i*.01,-d,h] for i in range(5)]) for d,h in [(2,.19),(4,.20),(6,.21),(10,.11),(20,.11)]]
        result=detect_obstacle_points(np.concatenate(groups),prediction)
        np.testing.assert_allclose(result.points,np.concatenate(groups[2:]))
        for kwargs in [dict(near_min_height_above_track_m=-1),dict(near_height_distance_m=float('nan'))]:
            with self.assertRaises(ValueError):ObstacleFilterConfig(**kwargs)

    def test_low_points_cannot_support_near_cluster_and_far_decisions_preserved(self):
        from obstacle_app.detection import RailPrediction, detect_obstacle_points
        depth=np.arange(1,151,.1,dtype=np.float32)
        prediction=RailPrediction(np.column_stack((np.zeros_like(depth),-depth,np.zeros_like(depth))),depth)
        # Five baseline points span 10 m. Removing four low near points must
        # preserve the far point's previously accepted decision.
        cross=np.array([[i*.01,-9.95,.15] for i in range(4)]+[[.04,-10,.15]])
        result=detect_obstacle_points(cross,prediction)
        np.testing.assert_allclose(result.points,cross[-1:])
        near=np.array([[i*.01,-3,.25] for i in range(4)]+[[.04,-3,.15]])
        self.assertFalse(detect_obstacle_points(near,prediction).detected)

    def test_regional_passes_match_full_scan_two_pass_filter(self):
        from scipy.spatial import cKDTree
        from obstacle_app.detection import RailPrediction, envelope_candidates, detect_obstacle_points
        rng = np.random.default_rng(731)
        depth = np.arange(1, 31, .1, dtype=np.float32)
        prediction = RailPrediction(np.column_stack((np.zeros_like(depth), -depth,
                                                     np.zeros_like(depth))), depth)
        for minimum in (1, 3, 5, 10):
            config = ObstacleFilterConfig(train_width_m=6, min_cluster_points=minimum)
            for _ in range(8):
                points = rng.uniform([-2.5, -25, .11], [2.5, -1.5, .3], (1500, 3))
                # Chains straddling the threshold must retain far baseline
                # support, even when low near members are removed.
                chains = np.array([[x, -d, h] for x in (-1, 0, 1)
                                   for d, h in zip(np.arange(8, 12, .15),
                                                   np.tile([.15, .25, .25], 9))])
                points = np.vstack((points, chains))
                candidates, _ = envelope_candidates(points, prediction, 6, 3)
                _, nearest = cKDTree(prediction.center[:, :2]).query(candidates[:, :2])
                heights = candidates[:, 2] - prediction.center[nearest, 2]
                keep = (heights > .1) & (-candidates[:, 1] >= 1.5)
                keep[keep] = filter_obstacle_candidates(candidates[keep], config)
                active = np.flatnonzero(keep)
                near = -candidates[active, 1] < 10
                eligible = ~near | (heights[active] > .2)
                again = np.zeros(len(active), dtype=bool)
                again[eligible] = filter_obstacle_candidates(candidates[active[eligible]], config)
                keep[active[near]] = again[near]
                result = detect_obstacle_points(points, prediction, config)
                np.testing.assert_array_equal(result.points, candidates[keep])
                np.testing.assert_array_equal(result.rejected_points, candidates[~keep])

    def test_bounded_rail_match_keeps_nearby_nonrail_points(self):
        from obstacle_app.detection import RailPrediction, detect_obstacle_points
        depth = np.arange(1, 31, .25, dtype=np.float32)
        prediction = RailPrediction(np.column_stack((np.zeros_like(depth), -depth,
                                                     np.zeros_like(depth))), depth,
                                    rail_points=np.array([[0., -5., .3]], dtype=np.float32))
        points = np.array([[0., -5., .3], [1e-5, -5., .3],
                           [2e-5, -5., .3], [2., -5., .3]])
        result = detect_obstacle_points(points, prediction,
            ObstacleFilterConfig(train_width_m=6, min_cluster_points=1))
        self.assertEqual(result.excluded_rail_count, 2)
        np.testing.assert_array_equal(result.points, points[2:].astype(np.float32))
