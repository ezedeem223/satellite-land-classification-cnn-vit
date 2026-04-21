from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .data import load_image_as_rgb


def custom_keras_data_generator(
    image_paths: Iterable[str | Path],
    labels: Iterable[int],
    batch_size: int,
    target_size: tuple[int, int] = (64, 64),
):
    """Yield batches of resized image tensors and labels for the preserved loading workflow."""

    paths = list(image_paths)
    label_list = list(labels)
    while True:
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch_labels = label_list[start : start + batch_size]
            images = [
                np.asarray(load_image_as_rgb(path, target_size), dtype=np.float32) / 255.0
                for path in batch_paths
            ]
            yield np.stack(images, axis=0), np.asarray(batch_labels, dtype=np.int64)


def keras_datagen_kwargs(validation_split: float = 0.2) -> dict[str, float | bool | str]:
    """Return the augmentation settings used in the preserved Keras experiments."""

    return {
        "rescale": 1.0 / 255.0,
        "rotation_range": 40,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.2,
        "zoom_range": 0.2,
        "horizontal_flip": True,
        "fill_mode": "nearest",
        "validation_split": validation_split,
    }


def pytorch_normalization() -> tuple[list[float], list[float]]:
    """Return the ImageNet normalization used in the preserved PyTorch experiments."""

    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
