"""Compatibility imports; filtering is implemented once in railpath."""
from railpath.obstacle_filter import (
    ObstacleFilterConfig, ObstacleDetection, RailPrediction,
    filter_obstacle_candidates, envelope_candidates, detect_obstacle_points, detect_obstacles,
)
