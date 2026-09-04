"""Rendering metadata and run registry helpers."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from coolname import generate

from schema_validation import validate_schema


REGISTRY_FIELDS = (
    "run_name",
    "rendered_at",
    "location",
    "trajectory_id",
    "source_ortho",
    "camera_config",
    "motion_config",
    "frame_count",
)


def choose_run_name(
    renders_dir: Path,
    location: str,
    trajectory_id: int,
    ortho_name: str,
    requested_name: str | None,
) -> str:
    """Return an unused explicit or generated rendering run name."""
    registry_names = registered_names(renders_dir / "renders.csv")
    if requested_name:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", requested_name):
            raise ValueError(
                "render_name may contain only letters, numbers, hyphens, "
                "and underscores"
            )
        if (
            requested_name in registry_names
            or (renders_dir / requested_name).exists()
        ):
            raise FileExistsError(
                f"Render name already exists: {requested_name}"
            )
        return requested_name

    prefix = f"{location}-{trajectory_id}-{ortho_name}"
    while True:
        name = f"{prefix}-{generate(2)[-1]}"
        if name not in registry_names and not (renders_dir / name).exists():
            return name


def build_metadata(
    run_name: str,
    rendered_at: str,
    location: str,
    ortho_path: Path,
    crs: str,
    camera_path: Path,
    camera_values: dict[str, Any],
    camera: dict[str, Any],
    intrinsic: np.ndarray,
    motion_path: Path,
    motion_values: dict[str, Any],
    trajectory_path: Path,
    trajectory_id: int,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build complete, reproducible metadata for one rendering run."""
    metadata = {
        "schema_version": 2,
        "run_name": run_name,
        "rendered_at": rendered_at,
        "location": location,
        "source_ortho": {
            "name": ortho_path.name,
            "path": str(ortho_path),
            "crs": crs,
        },
        "camera_config": {
            "name": camera_path.name,
            "path": str(camera_path),
            "values": camera_values,
        },
        "motion_config": {
            "name": motion_path.name,
            "path": str(motion_path),
            "values": motion_values,
        },
        "intrinsics": {
            "matrix": intrinsic.tolist(),
            **camera,
        },
        "trajectory": {
            "id": trajectory_id,
            "file": str(trajectory_path),
        },
        "frames": frames,
    }
    validate_schema(metadata, "meta-v2.schema.json", "render metadata")
    return metadata


def registry_row(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build a registry row from completed run metadata."""
    return {
        "run_name": metadata["run_name"],
        "rendered_at": metadata["rendered_at"],
        "location": metadata["location"],
        "trajectory_id": metadata["trajectory"]["id"],
        "source_ortho": metadata["source_ortho"]["path"],
        "camera_config": metadata["camera_config"]["path"],
        "motion_config": metadata["motion_config"]["path"],
        "frame_count": len(metadata["frames"]),
    }


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Atomically write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(data, file, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def registered_names(path: Path) -> set[str]:
    """Return all names recorded in the render registry."""
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as file:
        return {row["run_name"] for row in csv.DictReader(file)}


def append_registry(path: Path, row: dict[str, Any]) -> None:
    """Append one completed rendering run to the CSV registry."""
    if row["run_name"] in registered_names(path):
        raise FileExistsError(f"Run is already registered: {row['run_name']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REGISTRY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
