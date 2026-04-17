from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar"
)
DEFAULT_CLASS_NAMES = ("class_0_non_agri", "class_1_agri")
DEFAULT_LABEL_NAMES = ("non-agri", "agri")
DEFAULT_SEED = 7331


class AssetNotFoundError(FileNotFoundError):
    """Raised when a required dataset or model asset is missing."""


def project_path(*parts: str) -> Path:
    """Return an absolute path inside the repository."""

    return PROJECT_ROOT.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def require_path(path: str | Path, description: str) -> Path:
    """Validate that an asset exists and raise a clear error otherwise."""

    resolved = Path(path)
    if not resolved.exists():
        raise AssetNotFoundError(f"{description} not found at: {resolved}")
    return resolved


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python and NumPy for reproducible non-framework operations."""

    random.seed(seed)
    np.random.seed(seed)


def write_json(data: Any, output_path: str | Path) -> Path:
    """Write JSON with deterministic formatting."""

    destination = Path(output_path)
    ensure_dir(destination.parent)
    destination.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def read_env_path(env_name: str, default: str | Path) -> Path:
    """Resolve a path override from the environment."""

    return Path(os.getenv(env_name, str(default)))

