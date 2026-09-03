"""GeoJSON trajectory loading and pose sampling."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from sampling.camera import extrinsic
from sampling.config import AXES, POSITION_AXES


def load_trajectory(
    path: Path, trajectory_id: int
) -> tuple[dict[str, Any], np.ndarray]:
    """Load one trajectory feature by its zero-based feature index."""
    with path.open("r", encoding="utf-8") as file:
        geojson = json.load(file)
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("Trajectory file must be a GeoJSON FeatureCollection")

    features = geojson.get("features", [])
    if trajectory_id < 0 or trajectory_id >= len(features):
        raise ValueError(
            f"trajectory_id must be between 0 and {len(features) - 1}"
        )
    feature = features[trajectory_id]
    geometry = feature.get("geometry")
    if not geometry:
        raise ValueError(f"Trajectory {trajectory_id} has an empty geometry")

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "MultiLineString":
        if not isinstance(coordinates, list) or len(coordinates) != 1:
            raise ValueError("Trajectory must be a single-part MultiLineString")
        coordinates = coordinates[0]
    elif geometry_type != "LineString":
        raise ValueError("Trajectory must be a LineString or MultiLineString")

    try:
        points = np.asarray(coordinates, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Trajectory has invalid coordinates") from error
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        raise ValueError("Trajectory requires at least two XY points")
    points = points[:, :2]
    if not np.isfinite(points).all():
        raise ValueError("Trajectory coordinates must be finite")

    keep = np.concatenate(([True], np.any(np.diff(points, axis=0) != 0.0, axis=1)))
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Trajectory has zero length")
    return feature, points


def _oscillation(
    settings: dict[str, Any] | None,
    axes: tuple[str, ...],
    phases: dict[str, float],
    timestamp: float,
) -> dict[str, float]:
    if settings is None:
        return {axis: 0.0 for axis in axes}
    return {
        axis: settings["amplitudes"][axis]
        * math.sin(
            2.0 * math.pi * settings["frequencies"][axis] * timestamp
            + phases[axis]
        )
        for axis in axes
    }


def sample_trajectory(
    points: np.ndarray,
    motion: dict[str, Any],
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Sample camera poses along one trajectory."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")

    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    distance_step = motion["speed_mps"] / motion["camera_fps"]
    frame_count = math.floor(cumulative[-1] / distance_step) + 1
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)

    random = np.random.default_rng(motion["random_seed"])
    phases = {
        "orientation": dict(zip(AXES, random.uniform(0.0, 2.0 * math.pi, 3))),
        "position": dict(
            zip(POSITION_AXES, random.uniform(0.0, 2.0 * math.pi, 3))
        ),
    }

    frames = []
    for frame_index in tqdm(
        range(frame_count), desc="Sampling", unit="position", dynamic_ncols=True
    ):
        distance = frame_index * distance_step
        segment_index = min(
            int(np.searchsorted(cumulative, distance, side="right") - 1),
            len(segment_lengths) - 1,
        )
        distance_in_segment = distance - cumulative[segment_index]
        fraction = distance_in_segment / segment_lengths[segment_index]
        xy = points[segment_index] + fraction * deltas[segment_index]
        tangent = deltas[segment_index] / segment_lengths[segment_index]
        heading_deg = math.degrees(math.atan2(tangent[0], tangent[1]))
        timestamp = frame_index / motion["camera_fps"]

        position_noise = _oscillation(
            motion["position_oscillation"],
            POSITION_AXES,
            phases["position"],
            timestamp,
        )
        path_right = np.array([tangent[1], -tangent[0]])
        perturbed_xy = (
            xy
            + tangent * position_noise["forward"]
            + path_right * position_noise["right"]
        )
        position = np.array(
            [
                perturbed_xy[0],
                perturbed_xy[1],
                motion["altitude_m"] + position_noise["up"],
            ],
            dtype=np.float64,
        )

        orientation_noise = _oscillation(
            motion["orientation_oscillation"],
            AXES,
            phases["orientation"],
            timestamp,
        )
        orientation = {
            axis: motion["orientation_deg"][axis] + orientation_noise[axis]
            for axis in AXES
        }
        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": timestamp,
                "distance_m": distance,
                "position_world": position.tolist(),
                "heading_deg": heading_deg,
                "orientation_deg": orientation,
                "extrinsic_world_to_camera": extrinsic(
                    position, heading_deg, orientation
                ).tolist(),
            }
        )
    return frames
