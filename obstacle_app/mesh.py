"""Deterministic area-weighted sampling of a COLLADA obstacle mesh."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def load_collada_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the triangle geometry used by our generated COLLADA assets."""
    root = ET.parse(path).getroot()
    mesh = root.find(".//c:mesh", NS)
    if mesh is None:
        raise ValueError("COLLADA mesh is missing")
    vertices = mesh.find("c:vertices/c:input[@semantic='POSITION']", NS)
    triangle_groups = mesh.findall("c:triangles", NS)
    if vertices is None or not triangle_groups:
        raise ValueError("COLLADA positions or triangles are missing")
    position_id = vertices.attrib["source"].lstrip("#")
    source = next((s for s in mesh.findall("c:source", NS) if s.attrib.get("id") == position_id), None)
    if source is None:
        raise ValueError("COLLADA position source is missing")
    array = source.find("c:float_array", NS)
    if array is None or array.text is None:
        raise ValueError("COLLADA position array is empty")
    positions = np.fromstring(array.text, sep=" ", dtype=np.float64).reshape(-1, 3)
    face_groups = []
    for triangles in triangle_groups:
        inputs = triangles.findall("c:input", NS)
        vertex_input = next((i for i in inputs if i.attrib.get("semantic") == "VERTEX"), None)
        indices_node = triangles.find("c:p", NS)
        if vertex_input is None or indices_node is None or indices_node.text is None:
            raise ValueError("COLLADA triangle indices are missing")
        stride = max(int(i.attrib["offset"]) for i in inputs) + 1
        indices = np.fromstring(indices_node.text, sep=" ", dtype=np.int64)
        if len(indices) % (3 * stride):
            raise ValueError("Malformed COLLADA triangle index array")
        faces = indices.reshape(-1, 3, stride)[:, :, int(vertex_input.attrib["offset"])]
        if len(faces) != int(triangles.attrib["count"]):
            raise ValueError("COLLADA triangle count disagrees with indices")
        face_groups.append(faces)
    faces = np.concatenate(face_groups)
    if not np.isfinite(positions).all() or faces.min() < 0 or faces.max() >= len(positions):
        raise ValueError("Invalid COLLADA vertices or triangle indices")
    return positions.astype(np.float32), faces.astype(np.int32)


def sample_collada_surface(path: Path, count: int, seed: int = 2026) -> np.ndarray:
    positions, faces = load_collada_mesh(path)
    positions = positions.astype(np.float64)
    tri = positions[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("COLLADA mesh has no sampleable surface")
    rng = np.random.default_rng(seed)
    chosen = tri[rng.choice(len(tri), size=count, p=area / area.sum())]
    u = rng.random((count, 1))
    v = rng.random((count, 1))
    fold = (u + v) > 1
    u[fold] = 1 - u[fold]
    v[fold] = 1 - v[fold]
    return (chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0])
            + v * (chosen[:, 2] - chosen[:, 0])).astype(np.float32)
