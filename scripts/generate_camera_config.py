#!/usr/bin/env python
"""Generate a FakeFlight camera config from common lens specifications."""

from __future__ import annotations

import math
from pathlib import Path

import fire
import yaml
from logkit.core import add_handlers, get_logger
from logkit.handlers import DefaultConsoleHandler


FULL_FRAME_WIDTH_MM = 36.0
FULL_FRAME_HEIGHT_MM = 24.0

logger = get_logger("fake_flight", __name__, level="INFO", propagate=False)


def _positive_number(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return value


def _image_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _actual_focal_length(
    sensor_width_mm: float,
    sensor_height_mm: float,
    diagonal_fov_deg: float | None,
    full_frame_focal_length_mm: float | None,
) -> float:
    if (diagonal_fov_deg is None) == (full_frame_focal_length_mm is None):
        raise ValueError(
            "Pass exactly one of diagonal_fov_deg or "
            "full_frame_focal_length_mm"
        )

    sensor_diagonal = math.hypot(sensor_width_mm, sensor_height_mm)
    if diagonal_fov_deg is not None:
        diagonal_fov_deg = _positive_number(
            diagonal_fov_deg, "diagonal_fov_deg"
        )
        if diagonal_fov_deg >= 180.0:
            raise ValueError("diagonal_fov_deg must be less than 180")
        return sensor_diagonal / (
            2.0 * math.tan(math.radians(diagonal_fov_deg) / 2.0)
        )

    full_frame_focal_length_mm = _positive_number(
        full_frame_focal_length_mm, "full_frame_focal_length_mm"
    )
    full_frame_diagonal = math.hypot(
        FULL_FRAME_WIDTH_MM, FULL_FRAME_HEIGHT_MM
    )
    return (
        full_frame_focal_length_mm
        * sensor_diagonal
        / full_frame_diagonal
    )


def generate_camera_config(
    output_file: str,
    width_px: int,
    height_px: int,
    sensor_width_mm: float,
    sensor_height_mm: float,
    diagonal_fov_deg: float | None = None,
    full_frame_focal_length_mm: float | None = None,
) -> None:
    """Generate a camera YAML config using the active sensor dimensions.

    Args:
        output_file: Destination YAML path.
        width_px: Rendered image width in pixels.
        height_px: Rendered image height in pixels.
        sensor_width_mm: Active sensor width in millimetres.
        sensor_height_mm: Active sensor height in millimetres.
        diagonal_fov_deg: Desired diagonal field of view in degrees.
        full_frame_focal_length_mm: Desired 36 x 24 mm equivalent focal
            length. Mutually exclusive with diagonal_fov_deg.
    """
    width_px = _image_dimension(width_px, "width_px")
    height_px = _image_dimension(height_px, "height_px")
    sensor_width_mm = _positive_number(sensor_width_mm, "sensor_width_mm")
    sensor_height_mm = _positive_number(
        sensor_height_mm, "sensor_height_mm"
    )
    focal_length_mm = _actual_focal_length(
        sensor_width_mm,
        sensor_height_mm,
        diagonal_fov_deg,
        full_frame_focal_length_mm,
    )

    config = {
        "version": 1,
        "image": {
            "width_px": width_px,
            "height_px": height_px,
        },
        "sensor": {
            "width_mm": sensor_width_mm,
            "height_mm": sensor_height_mm,
        },
        "lens": {
            "focal_length_x_mm": focal_length_mm,
            "focal_length_y_mm": focal_length_mm,
            "skew_px": 0.0,
            "principal_point_x_px": None,
            "principal_point_y_px": None,
        },
    }

    output_path = Path(output_file)
    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    logger.info(
        "Generated camera config: %s (actual focal length %.6f mm)",
        output_path,
        focal_length_mm,
    )


if __name__ == "__main__":
    add_handlers(logger, __file__, [DefaultConsoleHandler()])
    fire.Fire(generate_camera_config)
