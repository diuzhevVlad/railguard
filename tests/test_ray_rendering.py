"""Geometry and record-preservation checks for opaque ray composition."""
from types import SimpleNamespace
from pathlib import Path
import unittest
import struct
from unittest.mock import patch

import numpy as np

from obstacle_app.cloud import append_xyz, filter_points, valid_xyz
from obstacle_app.ray_rendering import MeshRayRenderer, SensorRayModel
from obstacle_app.mesh import load_collada_mesh


def cloud(points, rings=None):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    dtype = np.dtype({"names": ["x", "y", "z", "intensity", "ring", "tag"],
                      "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<u2"],
                      "offsets": [0, 4, 8, 12, 16, 18], "itemsize": 20})
    records = np.zeros(len(points), dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        records[name] = points[:, axis]
    records["intensity"] = 17
    records["ring"] = 0 if rings is None else rings
    records["tag"] = np.arange(len(points)) + 100
    fields = [SimpleNamespace(name=name, offset=offset, datatype=kind, count=1)
              for name, offset, kind in (("x", 0, 7), ("y", 4, 7), ("z", 8, 7),
                                         ("intensity", 12, 7), ("ring", 16, 4), ("tag", 18, 4))]
    return SimpleNamespace(width=len(points), height=1, point_step=20,
                           row_step=len(points) * 20, fields=fields, is_bigendian=False,
                           is_dense=False, data=np.frombuffer(records.tobytes(), dtype=np.uint8),
                           header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=2)))


def box(low=(-.5, -11, -.5), high=(.5, -10, .5)):
    x, y, z = low
    X, Y, Z = high
    vertices = np.array([[x,y,z], [X,y,z], [X,Y,z], [x,Y,z],
                         [x,y,Z], [X,y,Z], [X,Y,Z], [x,Y,Z]], dtype=np.float32)
    faces = np.array([[0,2,1], [0,3,2], [4,5,6], [4,6,7], [0,1,5], [0,5,4],
                      [1,2,6], [1,6,5], [2,3,7], [2,7,6], [3,0,4], [3,4,7]])
    return vertices, faces


def sensor():
    return SensorRayModel(np.deg2rad(np.arange(-5, 5.01, .5)), np.arange(21),
                          np.ones(21, dtype=int), np.zeros(21), angular_step=np.deg2rad(.5))


class RayRenderingTests(unittest.TestCase):
    def test_vectorized_records_match_scalar_format_in_both_byte_orders(self):
        points = np.array([[.123456,-4.5,1.25],[-2.5,-8.125,3.75]],dtype=np.float32)
        rings = np.array([3,128],dtype=np.int32)
        for endian in ('<','>'):
            msg = cloud([[1,2,3]])
            original = struct.pack(endian+'ffffHH',1,2,3,17,0,100)
            msg.data = np.frombuffer(original,dtype=np.uint8)
            msg.is_bigendian = endian == '>'
            append_xyz(msg,points,rings)
            expected = original + b''.join(struct.pack(endian+'ffffHH',
                float(p[0]),float(p[1]),float(p[2]),17,int(ring),0)
                for p,ring in zip(points,rings))
            self.assertEqual(msg.data.tobytes(),expected)

    def test_scan_wall_hides_obstacle_between_recorded_rays(self):
        model = sensor()
        az, el = np.meshgrid(np.deg2rad(np.arange(-8, 8.01, .5)), model.elevations)
        dirs = np.column_stack((np.sin(az.ravel())*np.cos(el.ravel()),
            -np.cos(az.ravel())*np.cos(el.ravel()), np.sin(el.ravel())))
        wall = dirs * (-5/dirs[:,1,None])
        from obstacle_app.scan_mesh import ScanSurface
        surface = ScanSurface.from_points(wall, model)
        self.assertGreater(surface.triangle_count, 100)
        # These rays lie between recorded samples, where cell depth is insufficient.
        ray = np.array([[np.sin(np.deg2rad(.25)), -np.cos(np.deg2rad(.25)), 0]], dtype=np.float32)
        self.assertFalse(surface.visible(ray, np.array([10.]))[0])
        self.assertTrue(surface.visible(ray, np.array([4.]))[0])
        self.assertTrue(surface.visible(ray, np.array([5.1]))[0])
        # Sensor-local surface checks do not depend on the world origin.
        result = MeshRayRenderer(*box(),model).render(cloud(wall),np.eye(4))
        self.assertEqual(len(result.points),0)
        self.assertTrue(result.keep.all())
        # Isolate surface occlusion from the preexisting cell-depth filter:
        # a missing angular depth lookup must not expose rays through this wall.
        with patch.object(model, 'background_depth', return_value=np.full(
                len(model.elevations)*model.columns, np.inf, dtype=np.float32)):
            result = MeshRayRenderer(*box(),model).render(cloud(wall),np.eye(4))
        self.assertEqual(len(result.points),0)
        self.assertGreater(result.tunnel_hidden_count,50)
        self.assertTrue(result.keep.all())

    def test_scan_mesh_does_not_bridge_opening_or_depth_jump(self):
        from obstacle_app.scan_mesh import ScanSurface
        model = sensor()
        az, el = np.meshgrid(np.deg2rad(np.arange(-8,8.01,.5)),model.elevations)
        dirs = np.column_stack((np.sin(az.ravel())*np.cos(el.ravel()),
            -np.cos(az.ravel())*np.cos(el.ravel()),np.sin(el.ravel())))
        wall = dirs * (-5/dirs[:,1,None])
        opening = wall[abs(az.ravel()) > np.deg2rad(1)]
        surface = ScanSurface.from_points(opening,model)
        self.assertTrue(surface.visible(np.array([[0,-1,0]],dtype=np.float32),np.array([10.]))[0])
        wall[az.ravel()>0] *= 4
        surface = ScanSurface.from_points(wall,model)
        angle = np.deg2rad(.25)
        self.assertTrue(surface.visible(np.array([[np.sin(angle),-np.cos(angle),0]],dtype=np.float32),np.array([10.]))[0])

    def test_multiple_objects_share_first_hit_scene(self):
        near = box(low=(-1, -6, -1), high=(1, -5, 1))
        far = box(low=(-1, -11, -1), high=(1, -10, 1))
        renderer = MeshRayRenderer.from_meshes([near, far], sensor())
        result = renderer.render(cloud([[0, -20, 0]]), np.eye(4))
        self.assertGreater(len(result.points), 50)
        np.testing.assert_allclose(result.points[:, 1], -5, atol=2e-5)
        self.assertEqual(result.removed_count, 1)
        pose = np.eye(4)
        pose[1, 3] = -8
        result = renderer.render(cloud([[0, 0, 0]]), pose)
        self.assertGreater(len(result.points), 0)
        self.assertTrue(np.all(result.points[:, 1] <= 0))
        np.testing.assert_allclose(result.points[:, 1], -2, atol=2e-5)

    def test_first_hits_only_and_points_on_previously_empty_rays(self):
        renderer = MeshRayRenderer(*box(), sensor())
        result = renderer.render(cloud([[0, 0, 0]]), np.eye(4))
        self.assertGreater(len(result.points), 50)
        np.testing.assert_allclose(result.points[:, 1], -10, atol=2e-5)
        self.assertEqual(result.removed_count, 0)
        self.assertEqual(len(result.rings), len(result.points))

    def test_shadow_removal_nearer_foreground_and_untouched_bytes(self):
        msg = cloud([[0, -20, 0], [0, -5, 0], [3, -20, 0], [0, 0, 0], [np.nan, 0, 0]])
        original = msg.data.copy().reshape(-1, 20)
        result = MeshRayRenderer(*box(), sensor()).render(msg, np.eye(4))
        np.testing.assert_array_equal(result.keep, [False, True, True, True, True])
        self.assertFalse(np.any(np.linalg.norm(result.points[:, [0, 2]], axis=1) < 1e-5))
        filter_points(msg, result.keep)
        append_xyz(msg, result.points, result.rings)
        np.testing.assert_array_equal(msg.data[:80].reshape(-1, 20), original[1:])
        self.assertEqual(msg.width, 4 + len(result.points))

    def test_true_gap_is_not_filled_by_a_bounding_box_shadow(self):
        a, fa = box(low=(-1.5, -11, -.5), high=(-.7, -10, .5))
        b, fb = box(low=(.7, -11, -.5), high=(1.5, -10, .5))
        renderer = MeshRayRenderer(np.r_[a, b], np.r_[fa, fb + len(a)], sensor())
        result = renderer.render(cloud([[0, -20, 0], [2, -20, 0], [-2, -20, 0]]), np.eye(4))
        np.testing.assert_array_equal(result.keep, [True, False, False])
        self.assertFalse(np.any(abs(result.points[:, 0]) < .69))

    def test_density_changes_with_range_and_pose_is_world_fixed(self):
        renderer = MeshRayRenderer(*box(), sensor())
        empty = cloud([[0, 0, 0]])
        near = renderer.render(empty, np.eye(4))
        pose = np.eye(4)
        pose[1, 3] = 30
        far = renderer.render(empty, pose)
        self.assertGreater(len(near.points), 5 * len(far.points))
        np.testing.assert_allclose(far.points[:, 1] + 30, -10, atol=2e-5)

    def test_range_limit_does_not_restore_occluded_background(self):
        model = sensor()
        model.max_range_m = 5
        result = MeshRayRenderer(*box(), model).render(cloud([[0, -20, 0]]), np.eye(4))
        self.assertEqual(len(result.points), 0)
        self.assertEqual(result.removed_count, 1)

    def test_ring_angles_inferred_only_for_a_complete_channel_set(self):
        elevations = np.deg2rad(np.linspace(-25, 15, 128))
        azimuths = np.deg2rad(np.arange(-.8, .8, .1))
        az, el = np.meshgrid(azimuths, elevations)
        points = 30 * np.column_stack((np.sin(az.ravel()) * np.cos(el.ravel()),
                                       -np.cos(az.ravel()) * np.cos(el.ravel()), np.sin(el.ravel())))
        inferred = SensorRayModel.from_cloud(cloud(points, np.repeat(np.arange(128), len(azimuths))))
        self.assertEqual(inferred.source, "observed_ring_angles_and_steps_approximation")
        np.testing.assert_allclose(inferred.elevations, elevations, atol=1e-6)
        self.assertEqual(SensorRayModel.from_cloud(cloud([[0, -10, 0]])).source,
                         "nominal_pandar128_approximation")

    def test_filtering_organized_cloud_preserves_fields_and_discards_row_padding(self):
        msg = cloud([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        records = msg.data.copy().reshape(4, 20)
        padded = records[:2].tobytes() + b"padding!" + records[2:].tobytes() + b"padding!"
        msg.width, msg.height, msg.row_step = 2, 2, 48
        msg.data = np.frombuffer(padded, dtype=np.uint8)
        filter_points(msg, np.array([True, False, False, True]))
        self.assertEqual((msg.height, msg.width, msg.row_step), (1, 2, 40))
        np.testing.assert_array_equal(msg.data.reshape(2, 20), records[[0, 3]])

    def test_uneven_channel_cells_use_each_side_midpoint(self):
        model = SensorRayModel(np.array([0, .01, .1]), np.arange(3),
                               np.ones(3, dtype=int), np.zeros(3))
        point = np.array([[0, -5 * np.cos(.04), 5 * np.sin(.04)]])
        depth = model.background_depth(point, np.array([5]))
        self.assertEqual(depth[model.columns], 5)

    def test_near_contact_keeps_scan_and_other_objects_active(self):
        msg = cloud([[0,-20,0],[0,-5,0],[0,5,0]])
        near = box(low=(-1,-1,-1),high=(1,1,1))
        result = MeshRayRenderer(*near,sensor()).render(msg,np.eye(4))
        self.assertEqual(result.near_contact_obstacles,1)
        self.assertEqual(len(result.points),0)
        self.assertTrue(result.keep.all())
        renderer = MeshRayRenderer.from_meshes([near,box()],sensor())
        result = renderer.render(msg,np.eye(4))
        self.assertEqual(result.near_contact_obstacles,1)
        self.assertGreater(len(result.points),0)
        np.testing.assert_array_equal(result.keep,[False,True,True])
        np.testing.assert_allclose(result.points[:,1],-10,atol=2e-5)
        # Reuse the active scene through a stationary interval.
        active = renderer._active_renderer
        renderer.render(msg,np.eye(4))
        self.assertIs(renderer._active_renderer,active)
        # A non-contact object returns to the active scene on approach.
        pose = np.eye(4);pose[1,3]=4
        result = renderer.render(msg,pose)
        self.assertEqual(result.near_contact_obstacles,0)
        self.assertGreater(len(result.points),0)
        self.assertLess(float(-result.points[:,1].min()),6)

    def test_near_contact_uses_actual_surface_not_empty_bounding_box(self):
        a,fa=box(low=(-2,-1,-1),high=(-1,1,1))
        b,fb=box(low=(1,-1,-1),high=(2,1,1))
        result=MeshRayRenderer(np.r_[a,b],np.r_[fa,fb+len(a)],sensor()).render(
            cloud([[0,-20,0]]),np.eye(4))
        self.assertEqual(result.near_contact_obstacles,0)
        self.assertGreater(len(result.points),0)
        self.assertTrue(result.keep.all())
        # Sensor just outside a surface is also covered by the clearance rule.
        result=MeshRayRenderer(*box(low=(-1,-1.1,-1),high=(1,-.1,1)),sensor()).render(
            cloud([[0,-20,0]]),np.eye(4))
        self.assertEqual(result.near_contact_obstacles,1)
        self.assertTrue(result.keep.all())

    def test_every_model_disappears_from_scan_after_train_passes(self):
        root = Path(__file__).resolve().parents[1] / "models"
        for model in ("person_standing/meshes/person.dae", "ITL-RailGuard_alexander/meshes/person.dae",
                      "alien_reference/meshes/alien.dae"):
            with self.subTest(model=model):
                vertices, faces = load_collada_mesh(root / model)
                vertices[:, 1] -= 10
                vertices[:, 2] -= vertices[:, 2].min() + 1.1
                renderer = MeshRayRenderer(vertices, faces, sensor())
                msg = cloud([[0, 20, 0], [0, -20, 0]])
                before = renderer.render(msg, np.eye(4))
                self.assertGreater(len(before.points), 0)
                self.assertTrue(np.all(before.points[:, 1] <= 0))
                pose = np.eye(4)
                pose[1, 3] = -20
                # Include rotation: 'behind' is always checked in LiDAR coordinates.
                yaw = .3
                pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
                after = renderer.render(msg, pose)
                self.assertEqual(len(after.points), 0)
                self.assertEqual(after.removed_count, 0)
                self.assertTrue(after.keep.all())


if __name__ == "__main__":
    unittest.main()
