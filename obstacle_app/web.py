"""Local HTTP interface for browsing, playing, and exporting obstacle bags."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field as dataclass_field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from .cloud import preview_xyz
from .core import ObstacleSpec, WorldObstacleSpec, Progress, Scenario, SimulationBackend
from .world_map import WorldMapManager, world_mesh, preview_materials, source_fingerprint
from .detection import INITIAL_SPEED_MPS, build_merged_rail_map, RailPrediction
from .scan_detection import ScanDetector, display_points, rejected_payload
from .learned_detection import LearnedDetectionManager, prediction_payload
from .sampling_backend import RayCastingBackend


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("OBSTACLE_DATA_DIR", ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("OBSTACLE_OUTPUT_DIR", ROOT / "outputs"))
MODEL_PATH = Path(os.environ.get("OBSTACLE_MODEL_PATH", ROOT / "models/person_standing/meshes/person.dae"))
MODEL_FILES = {
    "niva_car": ("Niva car", "niva_car/meshes/niva.dae"),
    "person_standing": ("Standing person", "person_standing/meshes/person.dae"),
    "person_photo": ("Person from photo", "person_photo/meshes/person.dae"),
    "alien_reference": ("Alien", "alien_reference/meshes/alien.dae"),
    "trump_stylized": ("Stylized Donald Trump", "trump_stylized/meshes/trump.dae"),
    "box_square": ("Square box", "box_square/meshes/box.dae"),
    "box_long_thin": ("Long thin box", "box_long_thin/meshes/box.dae"),
    "wire_hanging": ("Hanging wire", "wire_hanging/meshes/wire.dae"),
}
STATIC = Path(__file__).resolve().parent / "static"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def resolve_bag(bag_id: str, data_root: Path, output_root: Path) -> Path:
    parts = bag_id.split("/")
    if len(parts) != 2 or parts[0] not in ("source", "output") or not NAME.fullmatch(parts[1]):
        raise ValueError("Choose a bag from the list")
    root = data_root if parts[0] == "source" else output_root
    path = root / parts[1]
    if path.is_symlink() or path.resolve().parent != root.resolve() or not (path / "metadata.yaml").is_file():
        raise ValueError("Bag not found")
    return path


def list_bags(root: Path, scope: str) -> list[dict[str, str]]:
    if not root.exists():
        return []
    return [{"id": f"{scope}/{p.name}", "name": p.name, "scope": scope}
            for p in sorted(root.iterdir()) if p.is_dir() and not p.is_symlink()
            and not p.name.startswith(".") and (p / "metadata.yaml").is_file()]


def source_initial_speed(bag: Path) -> float:
    name = bag.name
    scenario_path = bag / "obstacle_scenario.json"
    if scenario_path.is_file():
        name = json.loads(scenario_path.read_text()).get("source_bag", name)
    return INITIAL_SPEED_MPS.get(name, 0.0)


class ObstacleRequest(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    model_id: str
    encounter_seconds: float = Field(ge=0, allow_inf_nan=False)
    lateral_offset_m: float = Field(default=0, ge=-1, le=1, allow_inf_nan=False)
    height_above_track_m: float = Field(default=0, ge=0, le=4, allow_inf_nan=False)


class WorldObstacleRequest(BaseModel):
    model_id: str
    position: tuple[float, float, float]
    yaw_deg: float = Field(default=0, allow_inf_nan=False)
    scale: float = Field(default=1, gt=0, allow_inf_nan=False)


class JobRequest(BaseModel):
    bag_id: str
    topic: str
    obstacles: list[ObstacleRequest] = Field(default_factory=list, max_length=50)
    world_obstacle: WorldObstacleRequest | None = None
    world_obstacles: list[WorldObstacleRequest] = Field(default_factory=list, max_length=50)
    map_id: str | None = None
    initial_speed_m_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def has_obstacle(self):
        if self.world_obstacle is not None and self.world_obstacles:
            raise ValueError("Use either a single world obstacle or a list")
        if not self.obstacles and self.world_obstacle is None and not self.world_obstacles:
            raise ValueError("Add at least one obstacle")
        return self


class DetectionRequest(BaseModel):
    bag_id: str
    topic: str
    initial_speed_m_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    train_width_m: float = Field(default=2.1, gt=0, le=20, allow_inf_nan=False)
    train_height_m: float = Field(default=3., gt=0, le=20, allow_inf_nan=False)
    max_forward_m: float = Field(default=150., ge=5, le=200, allow_inf_nan=False)


@dataclass
class DetectionJob:
    id: str
    bag_id: str
    topic: str
    initial_speed_m_s: float
    map_path: Path
    state: str = "queued"
    progress: float = 0.0
    message: str = "Queued"


def detection_job_info(job: DetectionJob) -> dict:
    return {key: getattr(job, key) for key in
            ("id", "bag_id", "topic", "initial_speed_m_s", "state", "progress", "message")}


class DetectionManager:
    def __init__(self, output_root: Path, builder=build_merged_rail_map):
        self.root = output_root / ".detection_maps"
        self.builder = builder
        self.jobs: dict[str, DetectionJob] = {}
        self.detectors: dict[Path, ScanDetector] = {}
        self.lock = threading.Lock()

    def start(self, bag_id: str, bag: Path, topic: str, speed: float) -> DetectionJob:
        source_files = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                              for p in bag.glob("*.db3"))
        identity = json.dumps(["v2", bag_id, topic, speed, source_files])
        key = hashlib.sha256(identity.encode()).hexdigest()[:24]
        path = self.root / f"{key}.npz"
        with self.lock:
            for job in self.jobs.values():
                if job.map_path == path and job.state in ("queued", "running", "done"):
                    return job
            job = DetectionJob(uuid.uuid4().hex, bag_id, topic, speed, path)
            self.jobs[job.id] = job
            if path.is_file():
                job.state, job.progress, job.message = "done", 1.0, "Rail predictions ready"
                return job
        threading.Thread(target=self._run, args=(job, bag), daemon=True).start()
        return job

    def _run(self, job: DetectionJob, bag: Path) -> None:
        temporary_map = job.map_path.with_name(f"{job.id}.npz")
        def progress(value: float, message: str) -> None:
            with self.lock:
                job.progress = value
                job.message = message
        try:
            with self.lock:
                job.state = "running"
            if not job.map_path.is_file():
                self.builder(bag, job.topic, job.initial_speed_m_s, temporary_map,
                             progress)
                temporary_map.replace(job.map_path)
            with self.lock:
                job.state, job.progress, job.message = "done", 1.0, "Rail predictions ready"
        except Exception as exc:
            temporary_map.unlink(missing_ok=True)
            with self.lock:
                job.state = "failed"
                job.message = str(exc)

    def get(self, job_id: str) -> DetectionJob | None:
        with self.lock:
            return self.jobs.get(job_id)

    def detector(self, job: DetectionJob) -> ScanDetector:
        with self.lock:
            detector = self.detectors.get(job.map_path)
            if detector is None:
                detector = ScanDetector.from_map(job.map_path)
                self.detectors[job.map_path] = detector
            return detector


@dataclass
class Job:
    id: str
    state: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    output_bag_id: str | None = None
    warnings: list[str] = dataclass_field(default_factory=list)


class JobManager:
    def __init__(self, backend: SimulationBackend, output_root: Path):
        self.backend = backend
        self.output_root = output_root
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def start(self, scenario: Scenario) -> Job:
        with self.lock:
            if any(job.state in ("queued", "running") for job in self.jobs.values()):
                raise ValueError("An export is already running")
            job = Job(id=uuid.uuid4().hex)
            self.jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job, scenario), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, scenario: Scenario) -> None:
        name = f"obstacles_{time.strftime('%Y%m%d_%H%M%S')}_{job.id[:6]}"
        working_root = self.output_root / f".working-{job.id}"
        partial = working_root / name
        final = self.output_root / name
        def progress(value: float, message: str) -> None:
            with self.lock:
                job.progress = max(0.0, min(1.0, value))
                job.message = message
        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
            working_root.mkdir()
            with self.lock:
                job.state = "running"
            result = self.backend.run(scenario, partial, progress)
            if result != partial or not (partial / "metadata.yaml").is_file():
                raise ValueError("Simulation backend did not produce a ROS 2 bag")
            scenario_metadata = partial / "obstacle_scenario.json"
            warnings = json.loads(scenario_metadata.read_text()).get("warnings", []) if scenario_metadata.is_file() else []
            partial.rename(final)
            working_root.rmdir()
            with self.lock:
                job.state = "done"
                job.output_bag_id = f"output/{name}"
                job.warnings = warnings
                job.progress = 1.0
        except Exception as exc:
            if working_root.exists():
                shutil.rmtree(working_root)
            with self.lock:
                job.state = "failed"
                job.message = str(exc)

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)


def create_app(
    backend: SimulationBackend | None = None,
    data_root: Path = DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    model_path: Path = MODEL_PATH,
    detection_map_builder=build_merged_rail_map,
    scan_detector_factory=None,
) -> FastAPI:
    manager = JobManager(backend or RayCastingBackend(), output_root)
    detection_manager = DetectionManager(output_root, detection_map_builder)
    map_manager = WorldMapManager(output_root, detection_manager)
    learned_manager = LearnedDetectionManager(scan_detector_factory)
    models_root = model_path.parent.parent.parent

    app = FastAPI(title="Obstacle Bag Studio")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/app.js")
    def javascript() -> FileResponse:
        return FileResponse(STATIC / "app.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})

    @app.get("/style.css")
    def stylesheet() -> FileResponse:
        return FileResponse(STATIC / "style.css", media_type="text/css", headers={"Cache-Control": "no-store"})

    @app.get("/mesh-controls.js")
    def mesh_controls_javascript():
        return FileResponse(STATIC/"mesh-controls.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})

    @app.get("/map-editor.js")
    def map_javascript():
        return FileResponse(STATIC/"map-editor.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})

    def world_spec(item):
        if item.model_id not in MODEL_FILES:
            raise ValueError("Choose an obstacle from the list")
        path = models_root/MODEL_FILES[item.model_id][1]
        if not path.is_file() or not np.isfinite(item.position).all():
            raise ValueError("Mesh unavailable or invalid position")
        return WorldObstacleSpec(item.model_id, path, item.position, item.yaw_deg, item.scale)

    @app.post("/api/world-preview")
    def preview_world(item: WorldObstacleRequest):
        try:
            vertices, triangles = world_mesh(world_spec(item))
            colors, local = preview_materials(world_spec(item).model_path)
            return {"vertices": vertices.tolist(), "triangles": triangles.tolist(),
                    "colors": colors, "local_vertices": local, "cardboard": item.model_id.startswith("box_")}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/maps", status_code=202)
    def prepare_map(request: DetectionRequest):
        try:
            bag = resolve_bag(request.bag_id, data_root, output_root)
            with AnyReader([bag], default_typestore=TYPESTORE) as reader:
                matches = [c for c in reader.connections if c.topic == request.topic and c.msgtype == "sensor_msgs/msg/PointCloud2"]
                if len(matches) != 1:
                    raise ValueError("Select one PointCloud2 topic")
            return map_manager.start(request.bag_id, bag, request.topic, request.initial_speed_m_s if request.initial_speed_m_s is not None else source_initial_speed(bag))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/maps/jobs/{job_id}")
    def map_job(job_id: str):
        job = map_manager.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Map job not found")
        return asdict(job)

    @app.get("/api/maps/{map_id}/{kind}")
    def map_data(map_id: str, kind: Literal["points", "geometry"], bag_id: str, topic: str):
        try:
            bag = resolve_bag(bag_id, data_root, output_root)
            directory = map_manager.resolve(map_id, bag_id, bag, topic)
            return FileResponse(directory/("points.bin" if kind == "points" else "geometry.json"),
                                media_type="application/octet-stream" if kind == "points" else "application/json")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/bags")
    def bags() -> list[dict[str, str]]:
        return list_bags(data_root, "source") + list_bags(output_root, "output")

    @app.get("/model-thumbnails/{model_id}.png")
    def model_thumbnail(model_id: str):
        if model_id not in MODEL_FILES:
            raise HTTPException(404, "Unknown model")
        return FileResponse(STATIC/"model-thumbnails"/(model_id+".png"), media_type="image/png")

    @app.get("/api/models")
    def models() -> list[dict[str, str]]:
        return [{"id": model_id, "name": label, "thumbnail": f"/model-thumbnails/{model_id}.png"} for model_id, (label, relative) in MODEL_FILES.items()
                if (models_root / relative).is_file()]

    @app.get("/api/info")
    def info(bag_id: str) -> dict:
        try:
            path = resolve_bag(bag_id, data_root, output_root)
            with AnyReader([path], default_typestore=TYPESTORE) as reader:
                return {"bag_id": bag_id, "duration_seconds": reader.duration / 1e9, "initial_speed_m_s": source_initial_speed(path),
                        "topics": [{"name": c.topic, "type": c.msgtype, "messages": c.msgcount}
                                   for c in reader.connections]}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs", status_code=202)
    def start_job(request: JobRequest) -> dict:
        try:
            bag = resolve_bag(request.bag_id, data_root, output_root)
            if request.world_obstacle is not None or request.world_obstacles:
                if request.obstacles or request.map_id is None:
                    raise ValueError("Map placement requires a map ID and cannot include timed obstacles")
                directory = map_manager.resolve(request.map_id, request.bag_id, bag, request.topic)
                manifest = json.loads((directory/"manifest.json").read_text())
                if request.initial_speed_m_s is not None and request.initial_speed_m_s != manifest.get("initial_speed_m_s", source_initial_speed(bag)):
                    raise ValueError("Initial speed changed; prepare the merged map again")
                items = tuple(world_spec(item) for item in (request.world_obstacles or [request.world_obstacle]))
                with AnyReader([bag], default_typestore=TYPESTORE) as reader:
                    duration = reader.duration/1e9
                return asdict(manager.start(Scenario(bag, request.topic, 0, duration,
                    world_obstacles=items, world_map_directory=directory,
                    initial_speed_m_s=request.initial_speed_m_s if request.initial_speed_m_s is not None else source_initial_speed(bag))))
            if request.map_id is not None or not request.obstacles:
                raise ValueError("Add an obstacle and choose one placement mode")
            objects = []
            for item in request.obstacles:
                if item.model_id not in MODEL_FILES:
                    raise ValueError("Choose an obstacle from the list")
                model = models_root / MODEL_FILES[item.model_id][1]
                if not model.is_file():
                    raise ValueError("Selected obstacle mesh is unavailable")
                objects.append(ObstacleSpec(item.id, model, item.model_id,
                    item.encounter_seconds, item.lateral_offset_m, item.height_above_track_m))
            if len({item.id for item in objects}) != len(objects):
                raise ValueError("Each obstacle must have a unique ID")
            with AnyReader([bag], default_typestore=TYPESTORE) as reader:
                if not any(c.topic == request.topic and c.msgtype == "sensor_msgs/msg/PointCloud2"
                           for c in reader.connections):
                    raise ValueError("Select a PointCloud2 topic")
                bag_duration = reader.duration / 1e9
                if bag_duration <= 0:
                    raise ValueError("Bag has no duration")
            if bag_duration <= 4:
                raise ValueError("Recording must be longer than 4 seconds to add obstacles")
            if any(not 2 <= item.encounter_seconds <= bag_duration - 2 for item in objects):
                raise ValueError(f"Obstacle time must be between 2 and {bag_duration - 2:.2f} seconds "
                                 "(leave 2 seconds at each end of the recording)")
            scenario = Scenario(bag, request.topic, 0, bag_duration,
                obstacles=tuple(objects), initial_speed_m_s=request.initial_speed_m_s if request.initial_speed_m_s is not None else source_initial_speed(bag))
            job = manager.start(scenario)
            return asdict(job)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return asdict(job)

    @app.post("/api/detections", status_code=202)
    def start_detection(request: DetectionRequest) -> dict:
        try:
            bag = resolve_bag(request.bag_id, data_root, output_root)
            with AnyReader([bag], default_typestore=TYPESTORE) as reader:
                if not any(c.topic == request.topic and c.msgtype == "sensor_msgs/msg/PointCloud2"
                           for c in reader.connections):
                    raise ValueError("Select a PointCloud2 topic")
            return asdict(learned_manager.start(request.bag_id, bag, request.topic, request.train_width_m, request.train_height_m, request.max_forward_m))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/detections/{job_id}")
    def get_detection(job_id: str) -> dict:
        job = learned_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Detection job not found")
        return asdict(job)

    @app.get("/api/detections/{job_id}/frame")
    def get_detection_frame(job_id: str, timestamp_ns: int) -> dict:
        job = learned_manager.get(job_id)
        if job is None or job.state != "done":
            raise HTTPException(status_code=404, detail="Rail predictions are not ready")
        bag = resolve_bag(job.bag_id, data_root, output_root)
        if source_fingerprint(bag) != job.fingerprint:
            raise HTTPException(400,"Recording changed; restart detection for this bag")
        detector = learned_manager.detector(job)
        result = detector.cached(timestamp_ns)
        if result is None:
            bag = resolve_bag(job.bag_id, data_root, output_root)
            with AnyReader([bag], default_typestore=TYPESTORE) as reader:
                matches = [c for c in reader.connections if c.topic == job.topic]
                frame = next(reader.messages(connections=matches, start=timestamp_ns,
                                             stop=timestamp_ns + 1), None)
                if frame is None or frame[1] != timestamp_ns:
                    raise HTTPException(status_code=404, detail="Frame not found in selected bag")
                try:
                    result = detector.detect(timestamp_ns, reader.deserialize(frame[2], frame[0].msgtype))
                except Exception as exc:
                    raise HTTPException(400, f"RailGuard inference failed: {exc}") from exc
        prediction, detection = result
        return {"timestamp_ns": str(timestamp_ns), "candidate_points": detection.point_count,
                "detected": detection.detected, "distance_m": detection.distance_m,
                **prediction_payload(prediction),
                "detected_xyz": display_points(detection.points).tolist(), **rejected_payload(detection)}

    @app.websocket("/ws/playback")
    async def playback(ws: WebSocket) -> None:
        await ws.accept()
        bag_id = ws.query_params.get("bag_id", "")
        topic = ws.query_params.get("topic", "")
        detection_id = ws.query_params.get("detection_id", "")

        def open_detection(job_id: str) -> ScanDetector:
            job = learned_manager.get(job_id)
            if job is None or job.state != "done" or job.bag_id != bag_id or job.topic != topic:
                raise ValueError("Prepare rail predictions for this bag and topic first")
            if source_fingerprint(resolve_bag(bag_id,data_root,output_root)) != job.fingerprint:
                raise ValueError("Recording changed; restart detection for this bag")
            return learned_manager.detector(job)

        detector = None
        try:
            start_s = float(ws.query_params.get("start", "0"))
            if not math.isfinite(start_s) or start_s < 0:
                raise ValueError("Invalid playback start time")
            path = resolve_bag(bag_id, data_root, output_root)
            if detection_id:
                detector = open_detection(detection_id)
        except ValueError as exc:
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close()
            return

        paused = ws.query_params.get("paused") == "1"
        steps = 0
        refresh_overlay = False
        changed = asyncio.Event()

        async def controls() -> None:
            nonlocal paused, steps, detector, refresh_overlay
            try:
                while True:
                    command = await ws.receive_json()
                    action = command.get("action")
                    if action == "pause":
                        paused = True
                    elif action == "resume":
                        paused = False
                    elif action == "step":
                        paused = True
                        steps += 1
                    elif action == "show_detection":
                        try:
                            detector = open_detection(str(command.get("detection_id", "")))
                            refresh_overlay = True
                        except ValueError as exc:
                            await ws.send_json({"type": "overlay_error", "message": str(exc)})
                    elif action == "hide_detection":
                        detector = None
                        refresh_overlay = True
                    changed.set()
            except (WebSocketDisconnect, RuntimeError):
                changed.set()

        receiver = asyncio.create_task(controls())
        try:
            with AnyReader([path], default_typestore=TYPESTORE) as reader:
                matches = [c for c in reader.connections if c.topic == topic]
                if len(matches) != 1 or matches[0].msgtype != "sensor_msgs/msg/PointCloud2":
                    raise ValueError("Select one PointCloud2 topic")
                begin = reader.start_time + round(start_s * 1e9)
                if begin >= reader.end_time:
                    raise ValueError("Start is after the end of the bag")
                scenario_path = path / "obstacle_scenario.json"
                provenance = json.loads(scenario_path.read_text()) if scenario_path.is_file() else {}
                synthetic_count = provenance.get("sampled_points", 0)
                modified_timestamps = set(provenance.get("modified_timestamps_ns", []))
                frame_counts = provenance.get("frame_counts", {})

                async def send_frame(conn, bag_ns: int, raw: bytes, frame: int) -> None:
                    msg = reader.deserialize(raw, conn.msgtype)
                    counts = frame_counts.get(str(bag_ns), {})
                    tail = counts.get("synthetic_points", counts.get(
                        "added", synthetic_count if bag_ns in modified_timestamps else 0))
                    xyz = preview_xyz(msg, tail_points=tail)
                    prediction = None
                    detection = None
                    detected_points = np.empty((0, 3), dtype=np.float32)
                    if detector is not None:
                        selected_detector = detector
                        try:
                            prediction, detection = await asyncio.to_thread(selected_detector.detect, bag_ns, msg)
                            if detector is not selected_detector:
                                prediction, detection = None, None
                        except Exception as exc:
                            prediction = RailPrediction(np.empty((0,3),np.float32),np.empty(0,np.float32),
                                details={"status":"error","message":str(exc)})
                            await ws.send_json({"type":"overlay_error","message":f"RailGuard inference failed: {exc}"})
                        detected_points = display_points(detection.points) if detection is not None else detected_points
                        if len(detected_points):
                            xyz = np.concatenate((xyz, detected_points))
                    await ws.send_json({"type": "frame", "frame": frame,
                                        "timestamp_ns": str(bag_ns),
                                        "time_seconds": (bag_ns - reader.start_time) / 1e9,
                                        "points": len(xyz), "frame_id": msg.header.frame_id,
                                        "obstacle_points": tail,
                                        "removed_points": frame_counts.get(str(bag_ns), {}).get("removed", 0),
                                        "detected_points": len(detected_points),
                                        "candidate_points": detection.point_count if detection else 0,
                                        **rejected_payload(detection),
                                        "detected": detection.detected if detection else False,
                                        "distance_m": detection.distance_m if detection else None,
                                        **(prediction_payload(prediction) if prediction else {"rail_center": None,"rail_continuation": [],"rail_prediction": None})})
                    await ws.send_bytes(xyz.astype("<f4", copy=False).tobytes())

                previous = None
                previous_wall = time.monotonic()
                frame = 0
                last_frame = None
                for conn, bag_ns, raw in reader.messages(connections=matches, start=begin):
                    while paused and steps == 0:
                        if refresh_overlay:
                            refresh_overlay = False
                            if last_frame is not None:
                                await send_frame(*last_frame)
                            continue
                        changed.clear()
                        if not refresh_overlay and paused and steps == 0:
                            await changed.wait()
                        if receiver.done():
                            return
                    if steps:
                        steps -= 1
                    if previous is not None and not paused:
                        await asyncio.sleep(max(0., min(0.5, (bag_ns - previous) / 1e9) - (time.monotonic()-previous_wall)))
                    previous = bag_ns
                    previous_wall = time.monotonic()
                    frame += 1
                    last_frame = (conn, bag_ns, raw, frame)
                    await send_frame(*last_frame)
                    refresh_overlay = False
                await ws.send_json({"type": "end", "frames": frame})
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as exc:
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            receiver.cancel()
            try:
                await ws.close()
            except (RuntimeError, WebSocketDisconnect):
                pass

    return app


app = create_app()
