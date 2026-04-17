from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _expand_env_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _expand_env_values(data)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep merge configuration dictionaries."""

    merged: dict[str, Any] = {}
    for config in configs:
        merged = _merge_two(merged, config)
    return merged


def _merge_two(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_two(result[key], value)
        else:
            result[key] = value
    return result

