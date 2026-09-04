"""Validate FakeFlight files against their versioned JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import FormatChecker
from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for
from referencing import Registry, Resource


SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def _load_schemas() -> tuple[dict[str, dict[str, Any]], Registry]:
    schemas = {}
    resources = []
    for path in SCHEMA_DIR.glob("*.schema.json"):
        with path.open("r", encoding="utf-8") as file:
            schema = json.load(file)
        schemas[path.name] = schema
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return schemas, Registry().with_resources(resources)


def validate_schema(
    data: Any, schema_name: str, description: str
) -> None:
    """Validate data and raise a path-aware ValueError on failure."""
    schemas, registry = _load_schemas()
    try:
        schema = schemas[schema_name]
    except KeyError as error:
        raise ValueError(f"Unknown schema: {schema_name}") from error

    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(
        schema,
        registry=registry,
        format_checker=FormatChecker(),
    )
    error = best_match(validator.iter_errors(data))
    if error is None:
        return

    path = "$"
    for part in error.absolute_path:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    raise ValueError(f"Invalid {description} at {path}: {error.message}")
