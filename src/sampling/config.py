"""Load and validate rendering configuration files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from schema_validation import validate_schema


AXES = ("yaw", "pitch", "roll")
POSITION_AXES = ("forward", "right", "up")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_number(value: Any, name: str) -> float:
    value = _number(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _axis_values(
    block: dict[str, Any], axes: tuple[str, ...], prefix: str
) -> dict[str, float]:
    return {
        axis: _number(block.get(axis, 0.0), f"{prefix}.{axis}")
        for axis in axes
    }


def load_camera_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Load a camera config and calculate its intrinsic matrix."""
    config = load_yaml(path)
    validate_schema(config, "camera-config-v1.schema.json", "camera config")

    try:
        image = config["image"]
        sensor = config["sensor"]
        lens = config["lens"]
        width = _positive_number(image["width_px"], "image.width_px")
        height = _positive_number(image["height_px"], "image.height_px")
        sensor_width = _positive_number(sensor["width_mm"], "sensor.width_mm")
        sensor_height = _positive_number(
            sensor["height_mm"], "sensor.height_mm"
        )
        focal_x = _positive_number(
            lens["focal_length_x_mm"], "lens.focal_length_x_mm"
        )
        focal_y = _positive_number(
            lens["focal_length_y_mm"], "lens.focal_length_y_mm"
        )
        skew = _number(lens.get("skew_px", 0.0), "lens.skew_px")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing camera config field: {error}") from error

    cx = lens.get("principal_point_x_px")
    cy = lens.get("principal_point_y_px")
    cx = width / 2.0 if cx is None else _number(cx, "principal_point_x_px")
    cy = height / 2.0 if cy is None else _number(cy, "principal_point_y_px")
    if not 0.0 <= cx < width or not 0.0 <= cy < height:
        raise ValueError("The principal point must lie inside the image")

    intrinsic = np.array(
        [
            [focal_x * width / sensor_width, skew, cx],
            [0.0, focal_y * height / sensor_height, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera = {
        "image_width_px": int(width),
        "image_height_px": int(height),
    }
    return config, camera, intrinsic


def load_motion_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and normalize a motion config."""
    config = load_yaml(path)
    validate_schema(config, "motion-config-v1.schema.json", "motion config")

    random_seed = config.get("random_seed", 0)
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")

    motion = {
        "speed_mps": _positive_number(config.get("speed_mps"), "speed_mps"),
        "camera_fps": _positive_number(
            config.get("camera_fps"), "camera_fps"
        ),
        "altitude_m": _positive_number(
            config.get("altitude_m"), "altitude_m"
        ),
        "orientation_deg": _axis_values(
            config.get("orientation_deg") or {}, AXES, "orientation_deg"
        ),
        "random_seed": random_seed,
    }

    oscillations = (
        ("orientation_oscillation", "amplitude_deg", AXES),
        ("position_oscillation", "amplitude_m", POSITION_AXES),
    )
    for name, amplitude_name, axes in oscillations:
        block = config.get(name)
        if block is None:
            motion[name] = None
            continue
        if not isinstance(block, dict):
            raise ValueError(f"{name} must be a mapping or null")
        amplitudes = _axis_values(
            block.get(amplitude_name) or {}, axes, f"{name}.{amplitude_name}"
        )
        frequencies = _axis_values(
            block.get("frequency_hz") or {}, axes, f"{name}.frequency_hz"
        )
        if any(value < 0.0 for value in frequencies.values()):
            raise ValueError(f"{name} frequencies cannot be negative")
        motion[name] = {
            "amplitudes": amplitudes,
            "frequencies": frequencies,
        }
    return config, motion
