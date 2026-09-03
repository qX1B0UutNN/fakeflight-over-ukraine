"""Render camera frames from an orthophoto."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def _image_corners(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float64,
    )


def _ground_footprint(
    frame: dict[str, Any], intrinsic: np.ndarray, width: int, height: int
) -> np.ndarray:
    image_corners = _image_corners(width, height)
    camera_rays = np.column_stack(
        (image_corners, np.ones(len(image_corners), dtype=np.float64))
    ) @ np.linalg.inv(intrinsic).T

    extrinsic = np.asarray(frame["extrinsic_world_to_camera"], dtype=np.float64)
    camera_to_world = extrinsic[:3, :3].T
    origin = np.asarray(frame["position_world"], dtype=np.float64)
    world_rays = camera_rays @ camera_to_world.T
    distances = -origin[2] / world_rays[:, 2]
    if np.any(distances <= 0.0) or not np.isfinite(distances).all():
        raise ValueError(
            f"Frame {frame['frame_index']} does not see the ground plane"
        )
    return origin[:2] + world_rays[:, :2] * distances[:, None]


def _render_frame(
    source: rasterio.DatasetReader,
    frame: dict[str, Any],
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    footprint = _ground_footprint(frame, intrinsic, width, height)
    inverse_transform = ~source.transform
    source_points = np.array(
        [inverse_transform * tuple(point) for point in footprint],
        dtype=np.float64,
    )

    col_start = math.floor(source_points[:, 0].min())
    row_start = math.floor(source_points[:, 1].min())
    col_stop = math.ceil(source_points[:, 0].max()) + 1
    row_stop = math.ceil(source_points[:, 1].max()) + 1
    if (
        col_start < 0
        or row_start < 0
        or col_stop > source.width
        or row_stop > source.height
    ):
        raise ValueError(
            f"Frame {frame['frame_index']} footprint is outside the orthophoto"
        )

    window = Window(
        col_start, row_start, col_stop - col_start, row_stop - row_start
    )
    raster = source.read((1, 2, 3), window=window)
    raster = np.moveaxis(raster, 0, -1)
    source_points -= np.array([col_start, row_start])
    transform = cv2.getPerspectiveTransform(
        source_points.astype(np.float32),
        _image_corners(width, height).astype(np.float32),
    )
    return cv2.warpPerspective(raster, transform, (width, height))


def render_frames(
    ortho_path: Path,
    frames: list[dict[str, Any]],
    intrinsic: np.ndarray,
    width: int,
    height: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Render frames and return enriched frame metadata and raster CRS."""
    output_dir.mkdir(parents=True)
    metadata_frames = []
    with rasterio.open(ortho_path) as source:
        for frame in tqdm(
            frames, desc="Rendering", unit="frame", dynamic_ncols=True
        ):
            image = _render_frame(source, frame, intrinsic, width, height)
            image_name = f"frame-{frame['frame_index']:06d}.png"
            image_path = output_dir / image_name
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(image_path), bgr_image):
                raise OSError(f"Could not write frame: {image_path}")
            metadata_frames.append(
                {**frame, "image_path": f"frames/{image_name}"}
            )
        crs = str(source.crs)
    return metadata_frames, crs
