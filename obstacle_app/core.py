"""Scenario and backend boundary shared by the web app and future simulators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


@dataclass(frozen=True)
class ObstacleSpec:
    id: str
    model_path: Path
    model_id: str
    encounter_seconds: float
    lateral_offset_m: float = 0.0  # Positive is right relative to train travel.
    height_above_track_m: float = 0.0


@dataclass(frozen=True)
class WorldObstacleSpec:
    model_id: str
    model_path: Path
    position: tuple[float, float, float]
    yaw_deg: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True)
class Scenario:
    source_bag: Path
    topic: str
    start_seconds: float
    duration_seconds: float
    model_path: Path | None = None  # Legacy single-obstacle callers.
    obstacles: tuple[ObstacleSpec, ...] = ()
    model_id: str = "person_standing"
    distance_m: float = 100.0  # Rail arc length ahead of the first exported scan.
    rail_end_policy: str = "skip"
    ground_offset_m: float = 0.0
    ground_z_m: float | None = None
    every_n_frames: int = 1
    initial_speed_m_s: float = 0.0
    world_obstacle: WorldObstacleSpec | None = None
    world_map_directory: Path | None = None
    world_obstacles: tuple[WorldObstacleSpec, ...] = ()
    compression: str | None = None  # Optional lossless ROS 2 message compression.


Progress = Callable[[float, str], None]


class SimulationBackend(Protocol):
    def run(self, scenario: Scenario, output_directory: Path, progress: Progress) -> Path:
        """Produce a replayable ROS 2 bag at output_directory."""
