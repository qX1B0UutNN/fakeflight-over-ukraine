#!/usr/bin/env python3

from pathlib import Path

import cv2
import fire
from logkit.core import add_handlers, get_logger
from logkit.handlers import DefaultConsoleHandler
from PIL import Image


logger = get_logger("fake_flight", __name__, level="INFO", propagate=False)


def _scaled_size(
    width: int,
    height: int,
    max_width: int | None,
    max_height: int | None,
    scale: float | None,
) -> tuple[int, int]:
    if scale is not None:
        if scale <= 0:
            raise ValueError("scale must be greater than 0")
        ratio = scale
    else:
        ratios = []
        if max_width is not None:
            ratios.append(max_width / width)
        if max_height is not None:
            ratios.append(max_height / height)
        ratio = min(ratios, default=1.0)

    ratio = min(ratio, 1.0)
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def mp4_to_gif(
    input_path: str,
    output_path: str,
    fps: float = 10.0,
    max_width: int | None = 480,
    max_height: int | None = None,
    scale: float | None = None,
    start_second: float = 0.0,
    duration_seconds: float | None = None,
    speed_up: float = 1.0,
) -> None:
    """Convert part of an MP4 video to a proportionally resized GIF.

    Args:
        input_path: Input .mp4 video file.
        output_path: Output .gif file.
        fps: Output GIF frames per second.
        max_width: Maximum output width while preserving aspect ratio.
        max_height: Maximum output height while preserving aspect ratio.
        scale: Optional proportional scale, overriding maximum dimensions.
        start_second: Video position in seconds at which to start.
        duration_seconds: Number of seconds to use, or all remaining video.
        speed_up: GIF playback speed multiplier.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if input_path.suffix.lower() != ".mp4":
        raise ValueError(f"Expected a .mp4 input file, got: {input_path}")
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    if max_width is not None and max_width <= 0:
        raise ValueError("max_width must be greater than 0")
    if max_height is not None and max_height <= 0:
        raise ValueError("max_height must be greater than 0")
    if start_second < 0:
        raise ValueError("start_second must be 0 or greater")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than 0")
    if speed_up <= 0:
        raise ValueError("speed_up must be greater than 0")

    video = cv2.VideoCapture(str(input_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    source_fps = video.get(cv2.CAP_PROP_FPS) or fps
    output_fps = min(fps, source_fps)
    source_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError(f"Could not read video dimensions: {input_path}")

    output_size = _scaled_size(
        source_width,
        source_height,
        max_width=max_width,
        max_height=max_height,
        scale=scale,
    )
    frame_duration_ms = max(1, round(1000 / output_fps / speed_up))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    start_frame = round(start_second * source_fps)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    end_time = (
        None if duration_seconds is None else start_second + duration_seconds
    )
    next_sample_time = start_second
    sample_interval = 1.0 / output_fps

    try:
        while True:
            frame_time = frame_index / source_fps
            if end_time is not None and frame_time >= end_time:
                break

            ok, frame = video.read()
            if not ok:
                break

            if frame_time + 1e-9 >= next_sample_time:
                if (frame.shape[1], frame.shape[0]) != output_size:
                    frame = cv2.resize(
                        frame,
                        output_size,
                        interpolation=cv2.INTER_AREA,
                    )

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb_frame))
                next_sample_time += sample_interval

            frame_index += 1
    finally:
        video.release()

    if not frames:
        raise RuntimeError(f"No frames were read from: {input_path}")

    first_frame, *remaining_frames = frames
    first_frame.save(
        output_path,
        save_all=True,
        append_images=remaining_frames,
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )

    logger.info("Input: %s", input_path)
    logger.info("Source FPS: %.2f", source_fps)
    logger.info("Output FPS: %.2f", output_fps)
    logger.info("Playback speed: %.2fx", speed_up)
    logger.info("Source size: %dx%d", source_width, source_height)
    logger.info("Output size: %dx%d", output_size[0], output_size[1])
    logger.info("GIF frames: %d", len(frames))
    logger.info("Saved GIF: %s", output_path)


if __name__ == "__main__":
    add_handlers(logger, __file__, [DefaultConsoleHandler()])
    fire.Fire(mp4_to_gif)
