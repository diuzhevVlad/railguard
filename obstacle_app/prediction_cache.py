"""Validated on-disk cache for per-frame learned rail predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from railpath.obstacle_filter import RailPrediction

from .world_map import source_fingerprint


CACHE_SCHEMA = "rail-prediction-v1"


def _weights_fingerprint(weights: Path) -> str:
    files = []
    for relative in ("path/best.pt", "refine/last.pt", "refine/calib.json"):
        path = weights / relative
        stat = path.stat()
        files.append((relative, stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()


def cache_path(cache_directory: Path, bag: Path, topic: str,
               weights: Path) -> tuple[Path, str, str]:
    bag_key = source_fingerprint(bag)
    weights_key = _weights_fingerprint(weights)
    key = hashlib.sha256(json.dumps(
        [CACHE_SCHEMA, bag_key, topic, weights_key]).encode()).hexdigest()[:20]
    return cache_directory / f"{bag.name}-{key}.npz", bag_key, weights_key


def load_prediction_cache(cache_directory: Path, bag: Path, topic: str,
                          weights: Path) -> dict[int, RailPrediction] | None:
    path, bag_key, weights_key = cache_path(cache_directory, bag, topic, weights)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if (saved["schema"].item() != CACHE_SCHEMA
                or saved["bag_fingerprint"].item() != bag_key
                or saved["weights_fingerprint"].item() != weights_key
                or saved["topic"].item() != topic):
            return None
        timestamps = saved["timestamps_ns"]
        centers = saved["centers"]
        center_offsets = saved["center_offsets"]
        distances = saved["distances"]
        rails = saved["rail_points"]
        rail_offsets = saved["rail_offsets"]
        max_forward = saved["max_forward_m"]
        if not (len(center_offsets) == len(timestamps) + 1
                and len(rail_offsets) == len(timestamps) + 1
                and len(max_forward) == len(timestamps)
                and center_offsets[-1] == len(centers) == len(distances)
                and rail_offsets[-1] == len(rails)):
            raise ValueError(f"Invalid RailPrediction cache: {path}")
        predictions = {}
        for index, timestamp_ns in enumerate(timestamps):
            center_slice = slice(center_offsets[index], center_offsets[index + 1])
            rail_slice = slice(rail_offsets[index], rail_offsets[index + 1])
            limit = float(max_forward[index])
            predictions[int(timestamp_ns)] = RailPrediction(
                centers[center_slice].copy(), distances[center_slice].copy(),
                rail_points=rails[rail_slice].copy(),
                max_forward_m=None if np.isnan(limit) else limit,
            )
    return predictions


def save_prediction_cache(cache_directory: Path, bag: Path, topic: str,
                          weights: Path,
                          predictions: Iterable[tuple[int, RailPrediction]]) -> Path:
    path, bag_key, weights_key = cache_path(cache_directory, bag, topic, weights)
    rows = list(predictions)
    timestamps = np.asarray([timestamp for timestamp, _ in rows], dtype=np.int64)
    center_arrays = [np.asarray(item.center, dtype=np.float32) for _, item in rows]
    distance_arrays = [np.asarray(item.distance_along_m, dtype=np.float32)
                       for _, item in rows]
    rail_arrays = [np.asarray(item.rail_points, dtype=np.float32)
                   if item.rail_points is not None else np.empty((0, 3), np.float32)
                   for _, item in rows]
    if any(center.shape != (len(distance), 3)
           for center, distance in zip(center_arrays, distance_arrays)):
        raise ValueError("Cannot cache an invalid RailPrediction")
    if any(rails.ndim != 2 or rails.shape[1] != 3 for rails in rail_arrays):
        raise ValueError("Cannot cache invalid rail points")
    center_offsets = np.r_[0, np.cumsum([len(array) for array in center_arrays])]
    rail_offsets = np.r_[0, np.cumsum([len(array) for array in rail_arrays])]
    centers = np.concatenate(center_arrays) if center_arrays else np.empty((0, 3), np.float32)
    distances = np.concatenate(distance_arrays) if distance_arrays else np.empty(0, np.float32)
    rails = np.concatenate(rail_arrays) if rail_arrays else np.empty((0, 3), np.float32)
    max_forward = np.asarray([
        np.nan if item.max_forward_m is None else item.max_forward_m for _, item in rows
    ], dtype=np.float32)
    cache_directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, schema=np.asarray(CACHE_SCHEMA), bag_fingerprint=np.asarray(bag_key),
        weights_fingerprint=np.asarray(weights_key), topic=np.asarray(topic),
        timestamps_ns=timestamps, centers=centers,
        center_offsets=center_offsets.astype(np.int64), distances=distances,
        rail_points=rails, rail_offsets=rail_offsets.astype(np.int64),
        max_forward_m=max_forward,
    )
    temporary.replace(path)
    return path
