#!/usr/bin/env python
"""Validate a FakeFlight camera or motion configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import fire
from logkit.core import add_handlers, get_logger
from logkit.handlers import DefaultConsoleHandler


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sampling.config import (  # noqa: E402
    load_camera_config,
    load_motion_config,
    load_yaml,
)


logger = get_logger("fake_flight", __name__, level="INFO", propagate=False)


def _detect_config_type(path: Path) -> str:
    config = load_yaml(path)
    camera_keys = {"image", "sensor", "lens"}
    motion_keys = {"speed_mps", "camera_fps", "altitude_m"}
    if camera_keys & config.keys() and not motion_keys & config.keys():
        return "camera"
    if motion_keys & config.keys() and not camera_keys & config.keys():
        return "motion"
    raise ValueError(
        "Could not identify config type; pass --config_type=camera or motion"
    )


def validate_config(config_file: str, config_type: str | None = None) -> None:
    """Validate one camera or motion YAML configuration.

    Args:
        config_file: YAML configuration path.
        config_type: Optional explicit type: camera or motion.
    """
    path = Path(config_file)
    config_type = config_type or _detect_config_type(path)
    if config_type == "camera":
        load_camera_config(path)
    elif config_type == "motion":
        load_motion_config(path)
    else:
        raise ValueError("config_type must be camera or motion")
    logger.info("Valid %s config: %s", config_type, path)


if __name__ == "__main__":
    add_handlers(logger, __file__, [DefaultConsoleHandler()])
    fire.Fire(validate_config)
