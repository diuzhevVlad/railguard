"""Cached map previews and explicit world-space obstacle geometry."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from .cloud import valid_xyz, normalize_axes
from .mesh import load_collada_mesh

MAX_MAP_POINTS = 500_000


def source_fingerprint(bag):
    files = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                   for p in bag.iterdir() if p.suffix in (".db3", ".mcap", ".yaml"))
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_map(directory, bag, topic, bag_id=None):
    manifest = json.loads((directory/"manifest.json").read_text())
    if (manifest["topic"] != topic or manifest["source_fingerprint"] != source_fingerprint(bag)
            or (topic in ("/iv_points", "iv_points") and manifest.get("axes") != "iv-minus-y-v1")
            or (bag_id is not None and manifest["bag_id"] != bag_id)
            or manifest["map_id"] != directory.name
            or manifest["poses_sha256"] != file_digest(directory/"poses.npz")):
        raise ValueError("Map is stale or belongs to another recording/topic. Prepare the map again.")
    return manifest


def world_mesh(item):
    """The preview and export call this same transform, including anchor rules."""
    position = np.asarray(item.position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite([item.yaw_deg, item.scale]).all() or item.scale <= 0:
        raise ValueError("World position/yaw must be finite and scale must be positive")
    vertices, faces = load_collada_mesh(item.model_path)
    anchor = np.zeros(3) if item.model_id == "wire_hanging" else np.r_[
        (vertices[:, :2].min(axis=0)+vertices[:, :2].max(axis=0))/2, vertices[:, 2].min()]
    yaw = np.deg2rad(item.yaw_deg)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return (((vertices-anchor)*item.scale) @ rotation.T+position).astype(np.float32), faces


def reduce_cloud(points, limit=MAX_MAP_POINTS):
    """Spatially spread deterministic subset; temporary pool is at most 2× cap."""
    if not len(points):
        return points
    cells = np.floor(points/.1).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    if len(indices) > limit:
        xyz = cells[indices].astype(np.uint64)
        priority = (xyz[:, 0]*np.uint64(73856093)) ^ (xyz[:, 1]*np.uint64(19349663)) ^ (xyz[:, 2]*np.uint64(83492791))
        # Mix bits so selection doesn't favor coordinates close to zero.
        priority ^= priority >> np.uint64(30)
        priority *= np.uint64(0xbf58476d1ce4e5b9)
        indices = indices[np.argpartition(priority, limit-1)[:limit]]
    return points[indices].astype(np.float32)


@dataclass
class MapJob:
    id: str
    bag_id: str
    topic: str
    fingerprint: str
    initial_speed_m_s: float = 0.0
    state: str = "queued"
    progress: float = 0.0
    message: str = "Preparing map"
    map_id: str | None = None


class WorldMapManager:
    def __init__(self, output_root, detection_manager):
        self.root = output_root/".world_maps"
        self.detection = detection_manager
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, bag_id, bag, topic, speed):
        fingerprint = source_fingerprint(bag)
        with self.lock:
            for job in self.jobs.values():
                if (job.bag_id, job.topic, job.fingerprint, job.initial_speed_m_s) == (bag_id, topic, fingerprint, speed) and job.state != "failed":
                    if job.state == "done":
                        try:
                            self.resolve(job.map_id, bag_id, bag, topic)
                        except ValueError:
                            job.state = "failed"
                            continue
                    return asdict(job)
            job = MapJob(uuid.uuid4().hex, bag_id, topic, fingerprint, speed)
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, bag, speed), daemon=True).start()
        return asdict(job)

    def resolve(self, map_id, bag_id, bag, topic):
        if len(map_id) != 32 or any(c not in "0123456789abcdef" for c in map_id):
            raise ValueError("Invalid map identifier")
        directory = self.root/map_id
        try:
            validate_map(directory, bag, topic, bag_id)
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError("Map is unavailable, stale, or does not match this bag/topic") from exc
        return directory

    def _run(self, job, bag, speed):
        working = None
        try:
            job.state = "running"
            prediction = self.detection.start(job.bag_id, bag, job.topic, speed)
            while prediction.state in ("queued", "running"):
                job.progress = .65*prediction.progress
                job.message = prediction.message
                time.sleep(.1)
            if prediction.state != "done":
                raise ValueError(prediction.message)
            digest = file_digest(prediction.map_path)
            version = "world-v3-iv-axes" if job.topic in ("/iv_points", "iv_points") else "world-v2"
            map_id = hashlib.sha256(json.dumps([version, job.bag_id, job.topic, job.fingerprint, digest, job.initial_speed_m_s]).encode()).hexdigest()[:32]
            directory = self.root/map_id
            if directory.exists():
                try:
                    validate_map(directory, bag, job.topic, job.bag_id)
                except (ValueError, OSError, KeyError):
                    shutil.rmtree(directory)
            if not (directory/"manifest.json").is_file():
                self.root.mkdir(parents=True, exist_ok=True)
                working = self.root/(".working-"+job.id)
                working.mkdir()
                shutil.copyfile(prediction.map_path, working/"poses.npz")
                with np.load(working/"poses.npz", allow_pickle=False) as data:
                    poses, stamps, center = data["poses"], data["timestamps_ns"], data["center"]
                if poses.shape != (len(stamps), 4, 4) or not np.isfinite(poses).all() or not len(stamps):
                    raise ValueError("Invalid cached map poses")
                pool = np.empty((0, 3), dtype=np.float32)
                with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
                    connections = [c for c in reader.connections if c.topic == job.topic]
                    count = 0
                    for _, stamp, raw in reader.messages(connections=connections):
                        if count >= len(stamps) or stamp != stamps[count]:
                            raise ValueError("Cached map poses do not match the source scan timestamps")
                        msg = normalize_axes(reader.deserialize(raw, connections[0].msgtype), job.topic)
                        xyz = valid_xyz(msg, stride=max(1, (msg.width*msg.height)//5000))[:5000]
                        world = (xyz @ poses[count, :3, :3].T+poses[count, :3, 3]).astype(np.float32)
                        pool = np.concatenate((pool, world))
                        if len(pool) > 2*MAX_MAP_POINTS-5000:
                            pool = reduce_cloud(pool)
                        count += 1
                        job.progress = .65+.34*count/len(stamps)
                        job.message = f"Merging map preview: {count}/{len(stamps)} scans"
                    if count != len(stamps):
                        raise ValueError("Incomplete pose sequence")
                pool = reduce_cloud(pool)
                pool.astype("<f4").tofile(working/"points.bin")
                (working/"geometry.json").write_text(json.dumps(dict(centerline=center.tolist(), trajectory=poses[:, :3, 3].tolist(), point_count=len(pool))))
                if source_fingerprint(bag) != job.fingerprint:
                    raise ValueError("Source changed while preparing map")
                (working/"manifest.json").write_text(json.dumps(dict(map_id=map_id, bag_id=job.bag_id,
                    axes="iv-minus-y-v1" if job.topic in ("/iv_points", "iv_points") else "source",
                    topic=job.topic, source_fingerprint=job.fingerprint, poses_sha256=digest, initial_speed_m_s=job.initial_speed_m_s)))
                working.rename(directory)
                working = None
            validate_map(directory, bag, job.topic, job.bag_id)
            job.map_id, job.state, job.progress, job.message = map_id, "done", 1., "Merged map ready"
        except Exception as exc:
            job.state, job.message = "failed", str(exc)
        finally:
            if working is not None:
                shutil.rmtree(working, ignore_errors=True)


def preview_materials(path):
    """Diffuse colors in the same triangle order as load_collada_mesh."""
    import xml.etree.ElementTree as ET
    from .mesh import NS
    root = ET.parse(path).getroot()
    effects = {}
    for effect in root.findall(".//c:effect", NS):
        color = effect.find(".//c:diffuse/c:color", NS)
        effects[effect.attrib["id"]] = ([float(x) for x in color.text.split()[:3]]
            if color is not None else [.6, .6, .6])
    materials = {m.attrib["id"]: effects.get(m.find("c:instance_effect", NS).attrib["url"].lstrip("#"), [.6]*3)
                 for m in root.findall(".//c:material", NS)}
    bindings = {m.attrib["symbol"]: m.attrib["target"].lstrip("#")
                for m in root.findall(".//c:instance_material", NS)}
    colors = []
    for group in root.findall(".//c:mesh/c:triangles", NS):
        symbol = group.attrib.get("material", "")
        colors.extend([materials.get(bindings.get(symbol, symbol), [.6]*3)]*int(group.attrib["count"]))
    vertices, _ = load_collada_mesh(path)
    size = np.maximum(np.ptp(vertices, axis=0), 1e-6)
    return colors, ((vertices-vertices.min(axis=0))/size).tolist()
