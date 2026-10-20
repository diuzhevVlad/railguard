"""Public shared-core contracts for optional structural verification."""
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pytest
from railpath.obstacle_filter import ObstacleFilterConfig, RailPrediction, detect_obstacle_points
from railpath.structural_verification import _component_labels, _replace_center, _reject_ground_components


def fixture():
    depth=np.arange(0.,101.,.1)
    curve=np.column_stack([np.zeros(len(depth)),-depth,np.zeros(len(depth))]).astype(np.float32)
    prediction=RailPrediction(curve,depth,rail_points=np.empty((0,3),np.float32),max_forward_m=100.)
    # Four far-side points are supported by one low near-side return. The
    # near-side return is excluded by the 20 cm rule, but far-side points keep
    # their original full-scan clustering context.
    xyz=np.array([[0,-9.98,.15],[.01,-10.02,.15],[.02,-10.06,.15],[.03,-10.10,.15],[.04,-10.14,.15]],np.float32)
    return xyz,prediction


def test_noop_retains_full_scan_boundary_support_and_inputs():
    xyz,prediction=fixture();before=xyz.copy();center=prediction.center.copy()
    base=ObstacleFilterConfig(stabilize_envelope=False)
    expected=detect_obstacle_points(xyz,prediction,base)
    assert expected.point_count==4
    for cfg in [replace(base,verify_wall_geometry=True),replace(base,reject_trackbed_strips=True),replace(base,verify_wall_geometry=True,reject_trackbed_strips=True)]:
        actual=detect_obstacle_points(xyz,prediction,cfg)
        np.testing.assert_array_equal(actual.points,expected.points)
        assert actual.distance_m==expected.distance_m
    np.testing.assert_array_equal(xyz,before);np.testing.assert_array_equal(prediction.center,center)


def test_changed_wall_uses_full_context_and_cannot_add_points():
    xyz,prediction=fixture();cfg=ObstacleFilterConfig(stabilize_envelope=False,verify_wall_geometry=True)
    center=prediction.center.copy();center[:,0]+=.001
    changed=_replace_center(prediction,center,{})
    with patch('railpath.structural_verification.verification_geometry',return_value=(changed,np.ones(4,bool))):
        actual=detect_obstacle_points(xyz,prediction,cfg)
    assert actual.point_count==4
    np.testing.assert_array_equal(actual.points,xyz[1:])
    assert actual.prediction is changed


def test_floor_rejects_only_whole_thin_strips():
    _,prediction=fixture()
    strip=np.column_stack([np.linspace(-.2,.2,8),np.full(8,-55.),np.full(8,.05)]).astype(np.float32)
    box=strip.copy();box[:,1]-=2;box[-1,2]=.3
    points=np.r_[strip,box]
    keep=_reject_ground_components(points,prediction.center)
    assert not keep[:8].any();assert keep[8:].all()


def test_dense_component_budget_preserves_proposals():
    p=np.zeros((100,3),np.float32);p[:,0]=np.arange(100)*.00001
    assert _component_labels(p,pair_budget=10) is None


def test_continuation_follows_corrected_center():
    _,prediction=fixture();prediction=replace(prediction,continuation=prediction.center[500::10].copy())
    center=prediction.center.copy();center[:,0]+=.2
    actual=_replace_center(prediction,center,{})
    np.testing.assert_allclose(actual.continuation[:,0],.2)
    np.testing.assert_array_equal(prediction.continuation[:,0],0)
    assert np.all(np.diff(actual.distance_along_m)>0)


@pytest.mark.parametrize('name',['verify_wall_geometry','reject_trackbed_strips'])
def test_invalid_flags(name):
    with pytest.raises(ValueError):ObstacleFilterConfig(**{name:1})


def test_large_contact_cloud_skips_optional_rejection():
    points=np.zeros((20_001,3),np.float32)
    assert _component_labels(points) is None
