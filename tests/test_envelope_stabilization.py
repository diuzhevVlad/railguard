import unittest
from dataclasses import replace
import numpy as np

from railpath.obstacle_filter import (RailPrediction, ObstacleFilterConfig,
    detect_obstacle_points, detect_obstacles, filter_obstacle_candidates)
from railpath.envelope_stabilization import stabilize_rail_prediction


def scene(bias=.2, height=0., curve=0., length=50):
    d=np.arange(1.,length+.01,.25)
    center=np.column_stack((bias+curve*d*d,-d,np.full_like(d,height))).astype(np.float32)
    rd=np.arange(2.,40.,.1)
    rails=np.concatenate([np.column_stack((curve*rd*rd+side*.76,-rd,np.zeros_like(rd))) for side in (-1,1)]).astype(np.float32)
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(center,axis=0),axis=1))]+1
    return RailPrediction(center,lengths,rail_points=rails,max_forward_m=150.),rails


class EnvelopeStabilizationTests(unittest.TestCase):
    def test_observed_pairs_correct_bias_without_changing_width(self):
        prediction,rails=scene()
        result=stabilize_rail_prediction(rails,prediction)
        supported=-result.center[:,1]<=38
        np.testing.assert_allclose(result.center[supported,0],0,atol=1e-6)
        np.testing.assert_array_equal(result.center[:,1:],prediction.center[:,1:])
        self.assertTrue(result.details['envelope_stabilization']['lateral_applied'])
        points=np.array([[1.04,-12,.5],[1.06,-12,.5]],np.float32)
        detected=detect_obstacle_points(np.vstack((rails,points)),prediction,ObstacleFilterConfig(min_cluster_points=1))
        np.testing.assert_array_equal(detected.points,points[:1])

    def test_curvature_is_preserved(self):
        prediction,rails=scene(curve=.0005)
        result=stabilize_rail_prediction(rails,prediction)
        supported=-result.center[:,1]<=38
        np.testing.assert_allclose(result.center[supported,0],.0005*result.center[supported,1]**2,atol=.012)

    def test_lateral_correction_fades_outside_observed_support(self):
        prediction,rails=scene(length=100)
        result=stabilize_rail_prediction(rails,prediction)
        far=-result.center[:,1]>=51
        np.testing.assert_array_equal(result.center[far,0],prediction.center[far,0])
        self.assertLess(np.max(abs(np.diff(result.center[:,0]))),.01)
        legacy=stabilize_rail_prediction(rails,prediction,lateral_fade_m=None)
        np.testing.assert_allclose(legacy.center[:,0],0,atol=1e-6)

    def test_one_side_or_inconsistent_pair_does_not_shift(self):
        prediction,rails=scene()
        for bad in (rails[rails[:,0]>0],rails*np.array([2.,1.,1.])):
            result=stabilize_rail_prediction(np.empty((0,3)),replace(prediction,rail_points=bad))
            np.testing.assert_array_equal(result.center,prediction.center)

    def test_duplicates_do_not_invent_rail_support(self):
        prediction,rails=scene()
        # One unique sample on each rail per two-metre bin.
        sparse=rails[::20]
        result=stabilize_rail_prediction(np.empty((0,3)),replace(prediction,rail_points=np.repeat(sparse,20,axis=0)))
        np.testing.assert_array_equal(result.center,prediction.center)

    def test_height_correction_requires_continuous_bilateral_support(self):
        prediction,rails=scene(bias=0.,height=-.3,length=100)
        prediction=replace(prediction,rail_points=np.empty((0,3),np.float32))
        near=rails[-rails[:,1]<20]
        result=stabilize_rail_prediction(near,prediction)
        middle=(-result.center[:,1]>=5)&(-result.center[:,1]<=17)
        np.testing.assert_allclose(result.center[middle,2],0,atol=1e-5)
        np.testing.assert_array_equal(result.center[-result.center[:,1]>30,2],prediction.center[-prediction.center[:,1]>30,2])
        self.assertTrue(result.details['envelope_stabilization']['height_applied'])

    def test_short_box_cannot_set_track_height(self):
        prediction,_=scene(bias=0.)
        prediction=replace(prediction,rail_points=np.empty((0,3),np.float32))
        d=np.arange(10.,12.,.025)
        box=np.concatenate([np.column_stack((np.full_like(d,s*.76),-d,np.full_like(d,.3))) for s in (-1,1)]).astype(np.float32)
        result=stabilize_rail_prediction(box,prediction)
        np.testing.assert_array_equal(result.center,prediction.center)
        self.assertTrue(detect_obstacle_points(box,prediction).detected)

    def test_gaps_do_not_bridge_separate_objects_for_height(self):
        prediction,_=scene(bias=0.)
        prediction=replace(prediction,rail_points=np.empty((0,3),np.float32))
        d=np.r_[np.arange(10.,12.,.025),np.arange(20.,22.,.025),np.arange(30.,32.,.025)]
        points=np.concatenate([np.column_stack((np.full_like(d,s*.76),-d,np.full_like(d,.3))) for s in (-1,1)]).astype(np.float32)
        result=stabilize_rail_prediction(points,prediction)
        np.testing.assert_array_equal(result.center,prediction.center)

    def test_height_fix_removes_rails_but_retains_person_returns(self):
        prediction,rails=scene(bias=0.,height=-.3)
        prediction=replace(prediction,rail_points=np.empty((0,3),np.float32))
        obj=np.array([[i*.025,-12.,.5] for i in range(5)],np.float32)
        cloud=np.vstack((rails,obj))
        result=detect_obstacle_points(cloud,prediction)
        np.testing.assert_array_equal(result.points,obj)
        self.assertGreater(len(detect_obstacle_points(cloud,prediction,ObstacleFilterConfig(stabilize_envelope=False)).points),len(obj))

    def test_no_rail_provider_and_nonmonotonic_curves_are_unchanged(self):
        prediction,rails=scene()
        without=replace(prediction,rail_points=None)
        self.assertIs(stabilize_rail_prediction(rails,without),without)
        folded=prediction.center.copy();folded[10,1]=folded[9,1]
        invalid=replace(prediction,center=folded)
        self.assertIs(stabilize_rail_prediction(rails,invalid),invalid)

    def test_inputs_immutable_and_prepared_prediction_idempotent(self):
        prediction,rails=scene()
        center=prediction.center.copy();before=rails.copy()
        prediction=replace(prediction,continuation=center[-10:].copy(),details={'source':'test'})
        result=stabilize_rail_prediction(rails,prediction)
        np.testing.assert_array_equal(rails,before);np.testing.assert_array_equal(prediction.center,center)
        self.assertEqual(prediction.details,{'source':'test'})
        np.testing.assert_allclose(result.continuation[:,0],result.center[-10:,0],atol=1e-6)
        self.assertIs(stabilize_rail_prediction(rails,result),result)

    def test_unique_support_returns_all_records_in_original_order(self):
        pts=np.array([[i*.03,-10.,.5] for i in range(4)],np.float32)
        repeated=np.vstack((pts,pts))
        self.assertFalse(filter_obstacle_candidates(repeated).any())
        self.assertTrue(filter_obstacle_candidates(repeated,ObstacleFilterConfig(count_unique_xyz=False)).all())
        five=np.vstack((repeated,[[.12,-10.,.5]]))
        self.assertTrue(filter_obstacle_candidates(five).all())

    def test_unique_support_canonicalizes_signed_zero_without_rounding(self):
        points=np.array([[i*.03,-10.,0.] for i in range(4)],np.float64)
        negative_zero=points.copy();negative_zero[:,2]=-0.
        self.assertFalse(filter_obstacle_candidates(np.vstack((points,negative_zero))).any())
        fifth=points[-1:].copy();fifth[0,0]+=1e-12
        self.assertTrue(filter_obstacle_candidates(np.vstack((points,fifth))).all())

    def test_fast_unique_support_matches_float_sort_reference(self):
        from railpath.obstacle_filter import _filter_obstacle_candidates
        config=ObstacleFilterConfig()
        for seed in range(20):
            rng=np.random.default_rng(seed)
            points=rng.normal(size=(80,3))*[.5,2.,.1]+[0,-30,.3]
            points=np.vstack((points,points[rng.integers(0,80,100)]))
            unique,inverse=np.unique(points,axis=0,return_inverse=True)
            expected=_filter_obstacle_candidates(unique,config)[inverse]
            np.testing.assert_array_equal(filter_obstacle_candidates(points,config),expected)

    def test_compact_near_noise_removed_but_far_points_preserved(self):
        from railpath.obstacle_filter import filter_compact_near_components
        points=np.array([[i*.003,-3.,.5] for i in range(5)],np.float32)
        far=points.copy();far[:,1]=-50
        cloud=np.vstack((points,far));before=cloud.copy()
        np.testing.assert_array_equal(filter_compact_near_components(cloud),[False]*5+[True]*5)
        np.testing.assert_array_equal(cloud,before)
        self.assertTrue(filter_compact_near_components(points,ObstacleFilterConfig(min_cluster_points=1)).all())
        self.assertTrue(filter_compact_near_components(points,ObstacleFilterConfig(min_near_component_extent_m=0)).all())

    def test_compact_filter_preserves_thin_extended_objects_and_dense_surfaces(self):
        from railpath.obstacle_filter import filter_compact_near_components
        wire=np.column_stack((np.zeros(100),np.full(100,-3.),np.linspace(.3,1.5,100)))
        self.assertTrue(filter_compact_near_components(wire).all())
        dense=np.random.default_rng(2).uniform([-.5,-4.,.3],[.5,-3.,1.3],size=(20000,3))
        self.assertTrue(filter_compact_near_components(dense).all())

    def test_compact_filter_does_not_split_at_the_near_distance_boundary(self):
        from railpath.obstacle_filter import filter_compact_near_components
        points=np.column_stack((np.zeros(8),np.linspace(-9.99,-10.09,8),np.full(8,.3)))
        self.assertTrue(filter_compact_near_components(points).all())

    def test_compact_filter_keeps_float32_extent_boundary(self):
        from railpath.obstacle_filter import filter_compact_near_components
        points=np.array([[i*.01,-3.,.5] for i in range(5)],np.float32)
        self.assertTrue(filter_compact_near_components(points).all())

    def test_boolean_options_reject_accidental_strings(self):
        for kwargs in ({'count_unique_xyz':'false'},{'stabilize_envelope':1}):
            with self.assertRaises(ValueError):ObstacleFilterConfig(**kwargs)

    def test_app_and_node_share_corrected_geometry(self):
        from obstacle_app.learned_detection import adapt_prediction
        prediction,rails=scene()
        raw=prediction.center[::4]
        cloud=np.vstack((rails,np.array([[i*.02,-12.,.5] for i in range(5)],np.float32)))
        node=detect_obstacles(cloud,raw,rails,max_forward_m=150.)
        app=detect_obstacle_points(cloud,adapt_prediction(dict(centerline=raw,rails=rails,horizon=50.)))
        np.testing.assert_allclose(node.points,app.points)
        np.testing.assert_allclose(node.prediction.center,app.prediction.center)

    def test_structural_verifier_is_shared_by_app_and_node(self):
        from obstacle_app.learned_detection import adapt_prediction
        from obstacle_app.scan_detection import ScanDetector
        prediction,rails=scene()
        raw=prediction.center[::4]
        cloud=np.vstack((rails,np.array([[i*.02,-12.,.5] for i in range(5)],np.float32)))
        config=ScanDetector(None).filter_config
        self.assertTrue(config.verify_wall_geometry)
        self.assertTrue(config.reject_trackbed_strips)
        node=detect_obstacles(cloud,raw,rails,config=config,max_forward_m=150.)
        app=detect_obstacle_points(cloud,adapt_prediction(dict(centerline=raw,rails=rails,horizon=50.)),config)
        np.testing.assert_allclose(node.points,app.points)
        np.testing.assert_allclose(node.prediction.center,app.prediction.center)


if __name__=='__main__':unittest.main()
