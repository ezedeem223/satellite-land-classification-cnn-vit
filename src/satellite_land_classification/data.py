from __future__ import annotations

import tarfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from .utils import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_DATASET_URL,
    AssetNotFoundError,
    ensure_dir,
    require_path,
)


def validate_dataset_dir(
    dataset_dir: str | Path,
    class_names: tuple[str, str] | list[str] = DEFAULT_CLASS_NAMES,
) -> Path:
    """Ensure the dataset root and class folders exist."""

    root = require_path(dataset_dir, "Dataset directory")
    missing = [name for name in class_names if not root.joinpath(name).exists()]
    if missing:
        raise AssetNotFoundError(
            "Dataset directory is missing expected class folders: " + ", ".join(missing)
        )
    return root


def download_file(url: str, destination: str | Path, overwrite: bool = False) -> Path:
    """Download a file using the standard library."""

    output_path = Path(destination)
    ensure_dir(output_path.parent)
    if output_path.exists() and not overwrite:
        return output_path
    urllib.request.urlretrieve(url, output_path)
    return output_path


def extract_tar_archive(archive_path: str | Path, extract_dir: str | Path) -> Path:
    """Extract a tar archive to a directory."""

    archive = require_path(archive_path, "Dataset archive")
    destination = ensure_dir(extract_dir)
    with tarfile.open(archive, "r:*") as tar_ref:
        tar_ref.extractall(path=destination)
    return destination


def prepare_dataset(
    root_dir: str | Path,
    dataset_url: str = DEFAULT_DATASET_URL,
    archive_name: str = "images-dataSAT.tar",
    extracted_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Download and extract the public dataset archive to the requested location."""

    root = ensure_dir(root_dir)
    archive_path = root / archive_name
    download_file(dataset_url, archive_path, overwrite=overwrite)
    extract_tar_archive(archive_path, root)
    if extracted_dir is None:
        extracted_dir = root / "images_dataSAT"
    return validate_dataset_dir(extracted_dir)


def iter_image_files(dataset_dir: str | Path) -> Iterator[Path]:
    """Yield every image path in the binary dataset."""

    root = validate_dataset_dir(dataset_dir)
    for class_name in DEFAULT_CLASS_NAMES:
        yield from sorted(root.joinpath(class_name).glob("*.jpg"))


def build_image_index(dataset_dir: str | Path) -> list[tuple[Path, int]]:
    """Build a list of (path, label) pairs aligned with the dataset folder naming."""

    root = validate_dataset_dir(dataset_dir)
    index: list[tuple[Path, int]] = []
    for label, class_name in enumerate(DEFAULT_CLASS_NAMES):
        for path in sorted(root.joinpath(class_name).glob("*.jpg")):
            index.append((path, label))
    return index


def load_image_as_rgb(image_path: str | Path, target_size: tuple[int, int]) -> Image.Image:
    """Load and resize an RGB image using Pillow."""

    with Image.open(require_path(image_path, "Input image")) as image:
        return image.convert("RGB").resize(target_size)
