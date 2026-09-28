"""Conservative sensor-local surfaces from neighboring angular scan samples."""
from __future__ import annotations

import numpy as np


class ScanSurface:
    tolerance_m = .3
    maximum_edge_m = 1.5

    def __init__(self, scene, o3d, triangle_count):
        self.scene, self.o3d, self.triangle_count = scene, o3d, triangle_count

    @classmethod
    def from_points(cls, points, sensor):
        from .ray_rendering import angles
        import open3d as o3d

        points = np.asarray(points, dtype=np.float32)
        ranges = np.linalg.norm(points, axis=1)
        valid = np.isfinite(points).all(axis=1) & (ranges >= sensor.min_range_m) & (points[:, 1] <= 0)
        points, ranges = points[valid], ranges[valid]
        if len(points) < 3:
            return cls(None, o3d, 0)
        az, el = angles(points)
        upper = np.clip(np.searchsorted(sensor.elevations, el), 0, len(sensor.elevations)-1)
        lower = np.maximum(upper-1, 0)
        channel = np.where(abs(el-sensor.elevations[lower]) < abs(el-sensor.elevations[upper]), lower, upper)
        # Do not use points outside the sensor elevation grid to build surfaces.
        near = abs(el-sensor.elevations[channel]) <= np.deg2rad(.3)
        points, ranges, az, channel = points[near], ranges[near], az[near], channel[near]
        column = np.rint(az / sensor.angular_step).astype(np.int64) % sensor.columns
        keys = channel*sensor.columns+column
        # Nearest return wins when several points occupy the same angular cell.
        order = np.argsort(ranges, kind='stable')
        _, first = np.unique(keys[order], return_index=True)
        chosen = order[first]
        vertices = points[chosen]
        if len(vertices) < 3:
            return cls(None, o3d, 0)
        grid = np.full((len(sensor.elevations), sensor.columns), -1, dtype=np.int32)
        grid.ravel()[keys[chosen]] = np.arange(len(chosen))
        # A 0.2 degree grid accommodates observed alternating firing columns.
        step = max(1, int(round(np.deg2rad(.2)/sensor.angular_step)))
        faces = []
        occupied = keys[chosen]
        eligible = occupied // sensor.columns < len(sensor.elevations)-1
        vertex_ids = np.flatnonzero(eligible)
        row = occupied[eligible] // sensor.columns * sensor.columns
        column = occupied[eligible] % sensor.columns
        flat = grid.ravel()
        for shift in sorted({1, step} if np.any(sensor.strides > 1) else {1}):
            # Look up neighbors only for occupied cells, instead of scanning
            # every slot in the full 128 x 3600 grid on each triangle pass.
            first = np.column_stack((vertex_ids, flat[row+(column+shift)%sensor.columns],
                                      flat[row+sensor.columns+column]))
            second = np.column_stack((vertex_ids, flat[row+sensor.columns+column],
                                       flat[row+sensor.columns+(column-shift)%sensor.columns]))
            # Preserve the original row/column triangle ordering, including seam.
            second = second[np.argsort(row+(column-shift)%sensor.columns, kind="stable")]
            for ids in (first, second):
                ids = ids[(ids >= 0).all(axis=1)]
                if not len(ids):
                    continue
                tri = vertices[ids]
                lengths = np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2)
                r = np.linalg.norm(tri,axis=2)
                _, elevations = angles(tri.reshape(-1,3))
                elevation_span = np.ptp(elevations.reshape(-1,3),axis=1)
                # Reject discontinuities, long bridges, sparse vertical gaps and
                # degenerate triangles. Openings remain open when samples disagree.
                area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
                supported = ((lengths.max(axis=1) <= cls.maximum_edge_m)
                    & (np.ptp(r,axis=1) <= .5 + .02*r.min(axis=1))
                    & (elevation_span <= np.deg2rad(.75)) & (area > 1e-8))
                faces.append(ids[supported])
        triangles = np.concatenate(faces) if faces else np.empty((0,3),dtype=np.int32)
        scene = None
        if len(triangles):
            scene = o3d.t.geometry.RaycastingScene(nthreads=4)
            scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(triangles.astype(np.uint32)))
        return cls(scene, o3d, len(triangles))

    def visible(self, directions, obstacle_depth):
        if self.scene is None or not len(directions):
            return np.ones(len(directions),dtype=bool)
        rays = np.column_stack((np.zeros_like(directions),directions)).astype(np.float32)
        depth = self.scene.cast_rays(self.o3d.core.Tensor(rays),nthreads=4)['t_hit'].numpy()
        return ~(np.isfinite(depth) & (depth+self.tolerance_m < obstacle_depth))

    @classmethod
    def metadata(cls):
        return {'method':'per_scan_angular_triangles', 'coordinates':'current_lidar_frame',
                'odometry_required_for_mesh':False, 'occlusion_tolerance_m':cls.tolerance_m,
                'maximum_edge_m':cls.maximum_edge_m, 'maximum_range_jump_m':'0.5 + 0.02 * nearest_range',
                'maximum_vertical_gap_deg':.75, 'maximum_horizontal_gap_deg':.2,
                'holes':'fall_back_to_recorded_angular_depth', 'background_replacement':False, 'angular_crop_padding_deg':2}
