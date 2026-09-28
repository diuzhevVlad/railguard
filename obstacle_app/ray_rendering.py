"""Opaque first-hit mesh rendering and composition with recorded LiDAR returns.

The grid approximates a Pandar128. Observed channel angles/steps are used when
ring metadata is sufficient; otherwise a documented nominal grid is used.
This does not model beam divergence, material dropout, echoes, or scan motion.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .cloud import xyz_view


def coordinates(msg):
    cloud = xyz_view(msg)
    return np.column_stack([cloud[name].ravel() for name in ("x", "y", "z")])


def angles(points):
    return (np.arctan2(points[:, 0], -points[:, 1]),
            np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1)))


@dataclass
class SensorRayModel:
    elevations: np.ndarray  # ascending, radians
    rings: np.ndarray
    strides: np.ndarray
    phases: np.ndarray
    angular_step: float = np.deg2rad(0.1)
    min_range_m: float = 0.3
    max_range_m: float = 200.0
    source: str = "nominal_pandar128_approximation"

    def __post_init__(self):
        self.columns = int(round(2 * np.pi / self.angular_step))
        channel, column = np.meshgrid(np.arange(len(self.elevations)),
                                     np.arange(self.columns), indexing="ij")
        enabled = column % self.strides[:, None] == 0
        self.channels = channel[enabled]
        self.keys = (channel * self.columns + column)[enabled]
        az = column[enabled] * self.angular_step + self.phases[self.channels]
        el = self.elevations[self.channels]
        self.directions = np.column_stack((np.sin(az) * np.cos(el),
                                          -np.cos(az) * np.cos(el), np.sin(el))).astype(np.float32)
        # Augmentation is restricted to the train's forward half-space (-Y).
        forward = self.directions[:, 1] <= 0
        self.directions = self.directions[forward]
        self.channels = self.channels[forward]
        self.keys = self.keys[forward]

    @classmethod
    def from_cloud(cls, msg):
        points = coordinates(msg)
        ranges = np.linalg.norm(points, axis=1)
        valid = np.isfinite(points).all(axis=1) & (ranges > 3)
        ring = next((f for f in msg.fields if f.name == "ring" and f.count == 1), None)
        formats = {2: "u1", 4: "u2", 6: "u4"}
        estimates = []
        if ring is not None and ring.datatype in formats:
            endian = ">" if msg.is_bigendian else "<"
            ids = np.ndarray((msg.height, msg.width), dtype=endian + formats[ring.datatype],
                             buffer=msg.data, offset=ring.offset,
                             strides=(msg.row_step, msg.point_step)).ravel()
            azimuth, elevation = angles(points[valid])
            ids = ids[valid]
            for channel in np.unique(ids):
                mask = ids == channel
                if mask.sum() < 16:
                    continue
                az = azimuth[mask]
                delta = np.diff(np.sort(az))
                increments = delta[(delta > np.deg2rad(.05)) & (delta < np.deg2rad(.25))]
                stride = 2 if len(increments) > 100 and np.median(increments) > np.deg2rad(.15) else 1
                frequency = 3600 / stride
                phase = np.angle(np.mean(np.exp(1j * az * frequency))) / frequency
                estimates.append((float(np.median(elevation[mask])), int(channel), stride, phase))
        # A sparse/partial ring field is not a complete firing pattern.
        if len(estimates) == 128:
            estimates.sort()
            values = np.asarray(estimates)
            return cls(values[:, 0], values[:, 1].astype(np.int32),
                       values[:, 2].astype(np.int32), values[:, 3],
                       source="observed_ring_angles_and_steps_approximation")
        elevations = np.deg2rad(np.r_[np.linspace(-25, -6.5, 37),
                                      np.arange(-6, 2.001, .125), np.linspace(2.5, 15, 26)])
        return cls(elevations, np.arange(127, -1, -1), np.ones(128, dtype=int), np.zeros(128))

    def background_depth(self, points, ranges):
        """Approximate background depth by nearest firing cell, not a solid bbox."""
        az, el = angles(points)
        upper = np.clip(np.searchsorted(self.elevations, el), 0, len(self.elevations) - 1)
        lower = np.maximum(0, upper - 1)
        channels = np.where(abs(el - self.elevations[lower]) < abs(el - self.elevations[upper]),
                            lower, upper)
        steps = self.angular_step * self.strides[channels]
        column = (np.rint((az - self.phases[channels]) / steps).astype(np.int64)
                  * self.strides[channels]) % self.columns
        gaps = np.diff(self.elevations)
        if len(gaps):
            lower_half = np.r_[gaps[0], gaps] * .5
            upper_half = np.r_[gaps, gaps[-1]] * .5
        else:
            lower_half = upper_half = np.array([self.angular_step * .5])
        near = ((el >= self.elevations[channels] - lower_half[channels] - 1e-7)
                & (el <= self.elevations[channels] + upper_half[channels] + 1e-7))
        image = np.full(len(self.elevations) * self.columns, np.inf, dtype=np.float32)
        np.minimum.at(image, channels[near] * self.columns + column[near], ranges[near])
        return image

    def metadata(self):
        return {"source": self.source, "approximate": True,
                "channels": len(self.elevations), "azimuth_base_step_deg": .1,
                "elevations_deg": np.rad2deg(self.elevations).tolist(),
                "ring_ids": self.rings.tolist(), "horizontal_strides": self.strides.tolist(),
                "azimuth_phases_deg": np.rad2deg(self.phases).tolist(),
                "min_range_m": self.min_range_m, "max_range_m": self.max_range_m,
                "returns": "opaque_first_hit", "motion_during_scan": False,
                "obstacle_forward_axis": "-Y", "obstacle_horizontal_fov_deg": 180,
                "material_dropout": False}


@dataclass(frozen=True)
class RenderedScan:
    points: np.ndarray
    rings: np.ndarray
    keep: np.ndarray
    tunnel_hidden_count: int = 0
    tunnel_triangle_count: int = 0
    near_contact_obstacles: int = 0

    @property
    def removed_count(self):
        return int((~self.keep).sum())


class MeshRayRenderer:
    """Cache world-space geometry; transform rays for each sensor pose."""

    def __init__(self, world_vertices, faces, sensor):
        import open3d as o3d

        self.meshes = [(world_vertices, faces)]
        self.contact_bounds = np.array([[world_vertices.min(axis=0), world_vertices.max(axis=0)]])
        self._active_renderer = None
        self._active_key = None
        self.o3d = o3d
        self.sensor = sensor
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        self.scene.add_triangles(o3d.core.Tensor(world_vertices.astype(np.float32)),
                                 o3d.core.Tensor(faces.astype(np.uint32)))
        self._contact_scenes = [self.scene]
        low, high = world_vertices.min(axis=0), world_vertices.max(axis=0)
        self.center = (low + high) * .5
        self.radius = float(np.linalg.norm(high - low) * .5) + 1e-4
        self.corners = np.array(np.meshgrid(*zip(low, high))).reshape(3, -1).T

    @classmethod
    def from_meshes(cls, meshes, sensor):
        meshes = list(meshes)
        vertices, faces, offset, bounds = [], [], 0, []
        for points, triangles in meshes:
            vertices.append(points)
            faces.append(triangles + offset)
            offset += len(points)
            low, high = points.min(axis=0), points.max(axis=0)
            bounds.append(((low + high) * .5, float(np.linalg.norm(high - low) * .5) + 1e-4))
        renderer = cls(np.concatenate(vertices), np.concatenate(faces), sensor)
        renderer.bounds = bounds
        renderer.meshes = meshes
        renderer._contact_scenes = [renderer.scene] if len(meshes) == 1 else [None] * len(meshes)
        renderer.contact_bounds = np.asarray([[v.min(axis=0), v.max(axis=0)] for v, _ in meshes])
        return renderer

    def _inside_cone(self, directions, center, angular_padding=0.0):
        distance = np.linalg.norm(center)
        if distance <= self.radius:
            return np.ones(len(directions), dtype=bool)
        half_angle = min(np.pi, np.arcsin(self.radius / distance) + angular_padding)
        cosine = np.cos(half_angle)
        return directions @ (center / distance) >= cosine - 1e-6

    def _cast(self, directions, pose):
        if not len(directions):
            return np.empty(0, dtype=np.float32)
        rays = np.empty((len(directions), 6), dtype=np.float32)
        rays[:, :3] = pose[:3, 3]
        rays[:, 3:] = directions @ pose[:3, :3].T
        return self.scene.cast_rays(self.o3d.core.Tensor(rays), nthreads=4)["t_hit"].numpy()

    def render(self, msg, pose, background_count=None):
        keep = np.ones(msg.width * msg.height, dtype=bool)
        origin = pose[:3, 3]
        separation = np.maximum(np.maximum(self.contact_bounds[:, 0] - origin,
                                            origin - self.contact_bounds[:, 1]), 0)
        candidates = np.flatnonzero(np.linalg.norm(separation, axis=1) < self.sensor.min_range_m)
        contact = np.zeros(len(self.meshes), dtype=bool)
        for index in candidates:
            scene = self._contact_scenes[index]
            if scene is None:
                vertices, faces = self.meshes[index]
                scene = self.o3d.t.geometry.RaycastingScene(nthreads=4)
                scene.add_triangles(self.o3d.core.Tensor(vertices.astype(np.float32)),
                                    self.o3d.core.Tensor(faces.astype(np.uint32)))
                self._contact_scenes[index] = scene
            # Bounds are only a broad check: use the actual closed mesh so an
            # empty area inside an alien's long tail bounds is not suppressed.
            signed_distance = scene.compute_signed_distance(
                self.o3d.core.Tensor(origin[None].astype(np.float32)), nthreads=4, nsamples=3).numpy()[0]
            contact[index] = signed_distance < self.sensor.min_range_m
        if contact.any():
            active = tuple(np.flatnonzero(~contact).tolist())
            if not active:
                return RenderedScan(np.empty((0, 3), dtype=np.float32),
                                    np.empty(0, dtype=np.int32), keep,
                                    near_contact_obstacles=int(contact.sum()))
            # Keep a single active scene, rather than an unbounded cache of
            # combinations. Safe objects retain mutual occlusion and shadows.
            if active != self._active_key:
                self._active_renderer = MeshRayRenderer.from_meshes(
                    [self.meshes[index] for index in active], self.sensor)
                self._active_key = active
            result = self._active_renderer.render(msg, pose, background_count)
            return replace(result, near_contact_obstacles=int(contact.sum()))
        local_corners = (self.corners - pose[:3, 3]) @ pose[:3, :3]
        if np.min(local_corners[:, 1]) >= 0:
            return RenderedScan(np.empty((0, 3), dtype=np.float32),
                                np.empty(0, dtype=np.int32), keep)
        points = coordinates(msg)
        ranges = np.linalg.norm(points, axis=1)
        valid = np.isfinite(points).all(axis=1) & (ranges >= self.sensor.min_range_m)
        local_bounds = [((world_center - pose[:3, 3]) @ pose[:3, :3], radius)
                        for world_center, radius in getattr(self, "bounds", [(self.center, self.radius)])]
        def inside(directions, padding=0.0):
            selected = np.zeros(len(directions), dtype=bool)
            for local, radius in local_bounds:
                distance = np.linalg.norm(local)
                if distance <= radius:
                    return np.ones(len(directions), dtype=bool)
                angle = min(np.pi, np.arcsin(radius / distance) + padding)
                # Three-column dot products are faster without BLAS thread launch.
                selected |= np.einsum("ij,j->i", directions, local / distance) >= np.cos(angle) - 1e-6
            return selected

        indices = np.flatnonzero(valid)
        recorded_directions = points[valid] / ranges[valid, None]
        candidates = inside(recorded_directions) & (points[valid, 1] <= 0)
        recorded_hits = self._cast(recorded_directions[candidates], pose)
        hidden = ((recorded_hits + 1e-3 < ranges[indices[candidates]]) & np.isfinite(recorded_hits))

        emitted = np.flatnonzero(inside(self.sensor.directions))
        directions = self.sensor.directions[emitted]
        hits = self._cast(directions, pose)
        spacing = np.diff(self.sensor.elevations)
        margin = (float(spacing.max()) if len(spacing) else np.pi) + 2 * self.sensor.angular_step
        foreground = inside(recorded_directions, margin)
        foreground_indices = indices[foreground]
        depth = self.sensor.background_depth(points[foreground_indices], ranges[foreground_indices])
        accepted = (np.isfinite(hits) & (hits >= self.sensor.min_range_m)
                    & (hits <= self.sensor.max_range_m)
                    & (hits + 1e-3 < depth[self.sensor.keys[emitted]]))

        # Build a surface only for real obstacle hits needing an occlusion check.
        # Every vertex of a triangle intersecting a ray before distance R must
        # lie within R + maximum_edge_m of the sensor. This exact bound avoids
        # rebuilding the entire tunnel when a nearby object's cone grows large.
        from .scan_mesh import ScanSurface
        surface = None
        if hidden.any() or accepted.any():
            maximum_hit = max(float(recorded_hits[hidden].max()) if hidden.any() else 0.,
                              float(hits[accepted].max()) if accepted.any() else 0.)
            mesh_valid = valid & (ranges <= maximum_hit + ScanSurface.maximum_edge_m + 1e-4)
            if background_count is not None:
                mesh_valid[background_count:] = False
            mesh_points = points[mesh_valid]
            mesh_directions = mesh_points / ranges[mesh_valid, None]
            mesh_points = mesh_points[inside(mesh_directions, np.deg2rad(2))]
            surface = ScanSurface.from_points(mesh_points, self.sensor)
            checked = np.flatnonzero(hidden)
            hidden[checked] &= surface.visible(recorded_directions[candidates][checked], recorded_hits[checked])
            checked = np.flatnonzero(accepted)
            tunnel_visible = surface.visible(directions[checked], hits[checked])
            tunnel_hidden_count = int((~tunnel_visible).sum())
            accepted[checked] &= tunnel_visible
        else:
            tunnel_hidden_count = 0
        keep[indices[candidates][hidden]] = False
        return RenderedScan((directions[accepted] * hits[accepted, None]).astype(np.float32),
                            self.sensor.rings[self.sensor.channels[emitted[accepted]]], keep,
                            tunnel_hidden_count, surface.triangle_count if surface is not None else 0)
