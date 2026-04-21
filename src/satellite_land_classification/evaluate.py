from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .utils import ensure_dir, write_json


def compute_binary_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    label_names: list[str] | tuple[str, str],
) -> dict[str, Any]:
    """Compute the binary classification metrics used throughout the project."""

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)
    if y_prob_arr.ndim == 2:
        positive_prob = y_prob_arr[:, 1]
    else:
        positive_prob = y_prob_arr
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr)),
        "recall": float(recall_score(y_true_arr, y_pred_arr)),
        "f1_score": float(f1_score(y_true_arr, y_pred_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, positive_prob)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=list(label_names),
            digits=4,
        ),
    }


def save_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_names: list[str] | tuple[str, str],
    output_path: str | Path,
    title: str,
) -> Path:
    """Render and save a confusion matrix plot."""

    destination = Path(output_path)
    ensure_dir(destination.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=list(label_names),
    )
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def save_roc_curve(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render and save a ROC curve plot."""

    destination = Path(output_path)
    ensure_dir(destination.parent)
    probabilities = np.asarray(y_prob)
    if probabilities.ndim == 2:
        positive_prob = probabilities[:, 1]
    else:
        positive_prob = probabilities
    fpr, tpr, _ = roc_curve(y_true, positive_prob)
    auc = roc_auc_score(y_true, positive_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def write_metrics_json(
    metrics_by_model: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write evaluation metrics to JSON."""

    return write_json(metrics_by_model, output_path)


def write_classification_report(
    metrics_by_model: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write plain-text classification reports for all evaluated models."""

    destination = Path(output_path)
    ensure_dir(destination.parent)
    sections = []
    for model_name, metrics in metrics_by_model.items():
        sections.append(model_name)
        sections.append(metrics["classification_report"].rstrip())
        sections.append("")
    destination.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return destination


def write_comparison_csv(
    metrics_by_model: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write a compact comparison table for evaluated models."""

    destination = Path(output_path)
    ensure_dir(destination.parent)
    fieldnames = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in metrics_by_model.items():
            writer.writerow(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "roc_auc": metrics["roc_auc"],
                }
            )
    return destination
