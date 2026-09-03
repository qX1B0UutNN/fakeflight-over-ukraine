"""Camera pose calculations."""

from __future__ import annotations

import math

import numpy as np


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def extrinsic(
    position: np.ndarray,
    heading_deg: float,
    orientation: dict[str, float],
) -> np.ndarray:
    """Build a world-to-camera extrinsic matrix."""
    heading = math.radians(heading_deg + orientation["yaw"])
    forward = np.array([math.sin(heading), math.cos(heading), 0.0])
    right = np.array([math.cos(heading), -math.sin(heading), 0.0])
    camera_to_world = np.column_stack(
        (right, -forward, np.array([0.0, 0.0, -1.0]))
    )
    camera_to_world = (
        camera_to_world
        @ _rotation_x(math.radians(orientation["pitch"]))
        @ _rotation_z(math.radians(orientation["roll"]))
    )
    world_to_camera = camera_to_world.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = world_to_camera
    transform[:3, 3] = -world_to_camera @ position
    return transform
