"""Shared obstacle-detection configuration for the web app and ROS node."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping

import numpy as np

from .centerline_filter import CenterlineFilterConfig
from .obstacle_filter import ObstacleFilterConfig
from .occupancy_filter import BayesianFilterConfig


_BAYESIAN_PARAMETERS = {
    "occupancy_grid_resolution_m": "grid_resolution_m",
    "occupancy_prior_probability": "prior_probability",
    "occupancy_point_hit_probability": "point_hit_probability",
    "occupancy_miss_probability": "miss_probability",
    "occupancy_detection_probability": "detection_probability",
    "occupancy_prediction_decay": "prediction_decay",
    "occupancy_prediction_diffusion": "prediction_diffusion",
    "occupancy_neighbor_hit_weight": "neighbor_hit_weight",
    "occupancy_max_scan_hit_probability": "max_scan_hit_probability",
    "occupancy_immediate_close_distance_m": "immediate_close_distance_m",
    "occupancy_immediate_near_distance_m": "immediate_near_distance_m",
    "occupancy_immediate_near_probability": "immediate_near_probability",
    "occupancy_immediate_far_distance_m": "immediate_far_distance_m",
    "occupancy_max_point_evidence": "max_point_evidence",
    "occupancy_max_log_odds": "max_log_odds",
    "occupancy_max_time_gap_s": "max_time_gap_s",
    "odometry_scan_stride": "odometry_scan_stride",
}

_CENTERLINE_PARAMETERS = {
    "centerline_filter_algorithm": "algorithm",
    "centerline_filter_initial_variance_m2": "initial_variance_m2",
    "centerline_filter_process_variance_m2": "process_variance_m2",
    "centerline_filter_measurement_variance_m2": "measurement_variance_m2",
    "centerline_filter_measurement_variance_growth_m2_per_m":
        "measurement_variance_growth_m2_per_m",
    "centerline_filter_max_measurement_variance_m2":
        "max_measurement_variance_m2",
    "centerline_filter_use_model_uncertainty": "use_model_uncertainty",
    "centerline_filter_max_correspondence_distance_m":
        "max_correspondence_distance_m",
    "centerline_filter_association_fade_start_ratio":
        "association_fade_start_ratio",
    "centerline_nurbs_control_points": "nurbs_control_points",
    "centerline_nurbs_measurement_spacing_m": "nurbs_measurement_spacing_m",
    "centerline_nurbs_max_forward_m": "nurbs_max_forward_m",
    "centerline_nurbs_behind_spans": "nurbs_behind_spans",
    "centerline_nurbs_anchor_fraction": "nurbs_anchor_fraction",
    "centerline_nurbs_max_distance_m": "nurbs_max_distance_m",
    "centerline_nurbs_gate": "nurbs_gate",
    "centerline_nurbs_reinit_accept_fraction": "nurbs_reinit_accept_fraction",
    "centerline_nurbs_reinit_patience": "nurbs_reinit_patience",
    "centerline_nurbs_path_measurement_enabled": "nurbs_path_measurement_enabled",
    "centerline_nurbs_path_lateral_sigma_m": "nurbs_path_lateral_sigma_m",
    "centerline_nurbs_white_m": "nurbs_white_m",
    "centerline_nurbs_frame_translation_m": "nurbs_frame_translation_m",
    "centerline_nurbs_smooth_shape_m": "nurbs_smooth_shape_m",
    "centerline_nurbs_correlation_length_m": "nurbs_correlation_length_m",
    "centerline_nurbs_process_per_sqrt_second_m": "nurbs_process_per_sqrt_second_m",
    "centerline_nurbs_birth_m": "nurbs_birth_m",
    "centerline_nurbs_robust_threshold_m": "nurbs_robust_threshold_m",
    "centerline_nurbs_output_strength": "nurbs_output_strength",
}


@dataclass(frozen=True)
class ObstacleDetectionConfig:
    """All settings that affect shared obstacle detection."""

    obstacle: ObstacleFilterConfig = field(default_factory=ObstacleFilterConfig)
    bayesian: BayesianFilterConfig = field(default_factory=BayesianFilterConfig)
    centerline: CenterlineFilterConfig = field(default_factory=CenterlineFilterConfig)
    max_forward_m: float = 150.0
    bayesian_filter_enabled: bool = False
    centerline_filter_enabled: bool = False
    odometry_enabled: bool = False

    def __post_init__(self):
        if (not np.isfinite(self.max_forward_m)
                or self.max_forward_m < self.obstacle.min_forward_m):
            raise ValueError("max_forward_m must be finite and >= min_forward_m")
        for name in ("bayesian_filter_enabled", "centerline_filter_enabled",
                     "odometry_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> "ObstacleDetectionConfig":
        """Build from the flat names used by the ROS parameter YAML."""
        obstacle_values = {}
        for item in fields(ObstacleFilterConfig):
            if item.name in parameters:
                value = parameters[item.name]
                if item.name in ("distance_knots_m", "radius_knots_m"):
                    value = tuple(value)
                obstacle_values[item.name] = value
        bayesian_values = {
            field_name: parameters[parameter_name]
            for parameter_name, field_name in _BAYESIAN_PARAMETERS.items()
            if parameter_name in parameters
        }
        centerline_values = {
            field_name: parameters[parameter_name]
            for parameter_name, field_name in _CENTERLINE_PARAMETERS.items()
            if parameter_name in parameters
        }
        defaults = cls()
        return cls(
            obstacle=ObstacleFilterConfig(**obstacle_values),
            bayesian=BayesianFilterConfig(**bayesian_values),
            centerline=CenterlineFilterConfig(**centerline_values),
            max_forward_m=parameters.get("max_forward_m", defaults.max_forward_m),
            bayesian_filter_enabled=parameters.get(
                "bayesian_filter_enabled", defaults.bayesian_filter_enabled),
            centerline_filter_enabled=parameters.get(
                "centerline_filter_enabled", defaults.centerline_filter_enabled),
            odometry_enabled=parameters.get("odometry_enabled", defaults.odometry_enabled),
        )

    def to_parameters(self) -> dict[str, object]:
        """Return flat ROS-compatible values, also suitable for JSON."""
        values = {
            item.name: getattr(self.obstacle, item.name)
            for item in fields(ObstacleFilterConfig)
        }
        values.update({
            parameter_name: getattr(self.bayesian, field_name)
            for parameter_name, field_name in _BAYESIAN_PARAMETERS.items()
        })
        values.update({
            parameter_name: getattr(self.centerline, field_name)
            for parameter_name, field_name in _CENTERLINE_PARAMETERS.items()
        })
        values.update(
            max_forward_m=self.max_forward_m,
            bayesian_filter_enabled=self.bayesian_filter_enabled,
            centerline_filter_enabled=self.centerline_filter_enabled,
            odometry_enabled=self.odometry_enabled,
        )
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in values.items()
        }


def load_obstacle_detection_config(path: str | Path) -> ObstacleDetectionConfig:
    """Load obstacle settings from a ROS parameter YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load obstacle detection config") from exc
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"Invalid obstacle detection config: {config_path}")
    node = document.get("railguard_node", document)
    parameters = node.get("ros__parameters", node) if isinstance(node, dict) else None
    if not isinstance(parameters, dict):
        raise ValueError(f"Missing railguard_node.ros__parameters in {config_path}")
    return ObstacleDetectionConfig.from_parameters(parameters)
