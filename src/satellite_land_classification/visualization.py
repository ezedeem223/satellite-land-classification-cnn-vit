from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .utils import ensure_dir


def plot_training_history(
    history: dict[str, list[float]],
    output_dir: str | Path,
    prefix: str,
) -> list[Path]:
    """Save training accuracy and loss curves if the history contains them."""

    destination = ensure_dir(output_dir)
    saved_paths: list[Path] = []
    curve_specs = [
        (("accuracy", "val_accuracy"), "Accuracy", f"{prefix}_accuracy.png"),
        (("loss", "val_loss"), "Loss", f"{prefix}_loss.png"),
        (("train_accuracy", "val_accuracy"), "Accuracy", f"{prefix}_accuracy.png"),
        (("train_loss", "val_loss"), "Loss", f"{prefix}_loss.png"),
    ]
    seen = set()
    for keys, ylabel, filename in curve_specs:
        if keys in seen:
            continue
        seen.add(keys)
        train_key, val_key = keys
        if train_key not in history or val_key not in history:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history[train_key], label=train_key)
        ax.plot(history[val_key], label=val_key)
        ax.set_title(f"{prefix.replace('_', ' ').title()} {ylabel}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        path = destination / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)
    return saved_paths
