#!/usr/bin/env python
"""Sample one trajectory and render its camera frames."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import fire
from logkittt.core import add_handlers, get_logger
from logkittt.handlers import DefaultConsoleHandler


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rendering.frames import render_frames  # noqa: E402
from rendering.metadata import (  # noqa: E402
    append_registry,
    build_metadata,
    choose_run_name,
    registry_row,
    write_json_atomic,
)
from sampling.config import load_camera_config, load_motion_config  # noqa: E402
from sampling.trajectory import load_trajectory, sample_trajectory  # noqa: E402


logger = get_logger("fake_flight", __name__, level="INFO", propagate=False)


def _location_name(trajectory_path: Path) -> str:
    if trajectory_path.parent.name != "geoms":
        raise ValueError("trajectory_file must be inside a location geoms folder")
    return trajectory_path.parent.parent.name


def _save_video(
    run_dir: Path,
    metadata_frames: list[dict],
    fps: float,
    width: int,
    height: int,
) -> None:
    video_path = run_dir / "render.mp4"
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), codec, fps, (width, height))
    try:
        for frame in metadata_frames:
            image = cv2.imread(str(run_dir / frame["image_path"]))
            writer.write(image)
    finally:
        writer.release()
    logger.info("Wrote video to %s", video_path)


def render(
    camera_config: str,
    motion_config: str,
    trajectory_file: str,
    trajectory_id: int,
    source_ortho: str,
    render_name: str | None = None,
    renders_dir: str = "dataset/renders",
    max_frames: int | None = None,
    save_video: bool = False,
) -> None:
    """Render one trajectory into a self-contained rendering run.

    Args:
        camera_config: Camera YAML path.
        motion_config: Motion YAML path.
        trajectory_file: Trajectory GeoJSON path.
        trajectory_id: Zero-based GeoJSON feature index to render.
        source_ortho: Source GeoTIFF path.
        render_name: Optional output run name override.
        renders_dir: Root folder for all rendering runs.
        max_frames: Optional frame limit, useful for test renders.
        save_video: Whether to merge rendered frames into render.mp4.
    """
    camera_path = Path(camera_config)
    motion_path = Path(motion_config)
    trajectory_path = Path(trajectory_file)
    ortho_path = Path(source_ortho)
    output_root = Path(renders_dir)
    trajectory_id = int(trajectory_id)
    max_frames = None if max_frames is None else int(max_frames)

    # Stage 1: resolve the run identity before creating output.
    location = _location_name(trajectory_path)
    run_name = choose_run_name(
        output_root, location, trajectory_id, ortho_path.stem, render_name
    )
    run_dir = output_root / run_name
    rendered_at = datetime.now(timezone.utc).isoformat()
    logger.info("Rendering run: %s", run_name)

    # Stage 2: load the complete, reproducible input definition.
    camera_values, camera, intrinsic = load_camera_config(camera_path)
    motion_values, motion = load_motion_config(motion_path)
    _, points = load_trajectory(trajectory_path, trajectory_id)

    # Stage 3: sample only the selected trajectory.
    frames = sample_trajectory(points, motion, max_frames=max_frames)

    # Stage 4: render sampled poses.
    metadata_frames, crs = render_frames(
        ortho_path,
        frames,
        intrinsic,
        camera["image_width_px"],
        camera["image_height_px"],
        run_dir / "frames",
    )

    if save_video:
        _save_video(
            run_dir,
            metadata_frames,
            motion["camera_fps"],
            camera["image_width_px"],
            camera["image_height_px"],
        )

    # Stage 5: make the run self-describing.
    metadata = build_metadata(
        run_name,
        rendered_at,
        location,
        ortho_path,
        crs,
        camera_path,
        camera_values,
        camera,
        intrinsic,
        motion_path,
        motion_values,
        trajectory_path,
        trajectory_id,
        metadata_frames,
    )
    write_json_atomic(run_dir / "meta.json", metadata)

    # Stage 6: register the completed run.
    append_registry(
        output_root / "renders.csv",
        registry_row(metadata),
    )
    logger.info("Wrote %d frames and metadata to %s", len(frames), run_dir)


if __name__ == "__main__":
    add_handlers(logger, __file__, [DefaultConsoleHandler()])
    fire.Fire(render)
