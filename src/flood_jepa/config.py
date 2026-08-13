"""Small configuration loader with environment-variable expansion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    def expand(value):
        if isinstance(value, str):
            return os.path.expanduser(os.path.expandvars(value))
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    return expand(config)


def require_sections(config: dict[str, Any], *names: str) -> None:
    missing = [name for name in names if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``section.key=value`` CLI overrides with YAML-typed values."""
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must have the form section.key=value: {override!r}")
        path, raw_value = override.split("=", 1)
        keys = path.split(".")
        if not keys or any(not key for key in keys):
            raise ValueError(f"Invalid override path: {path!r}")
        destination: Any = config
        for key in keys[:-1]:
            if isinstance(destination, dict):
                if key not in destination:
                    raise KeyError(f"Unknown configuration path: {path!r}")
                destination = destination[key]
            elif isinstance(destination, list) and key.isdigit():
                index = int(key)
                if index >= len(destination):
                    raise KeyError(f"Unknown configuration path: {path!r}")
                destination = destination[index]
            else:
                raise KeyError(f"Unknown configuration path: {path!r}")
        final_key = keys[-1]
        value = yaml.safe_load(raw_value)
        if isinstance(destination, dict):
            if final_key not in destination:
                raise KeyError(f"Unknown configuration key: {path!r}")
            destination[final_key] = value
        elif isinstance(destination, list) and final_key.isdigit():
            index = int(final_key)
            if index >= len(destination):
                raise KeyError(f"Unknown configuration key: {path!r}")
            destination[index] = value
        else:
            raise KeyError(f"Unknown configuration key: {path!r}")
    return config
