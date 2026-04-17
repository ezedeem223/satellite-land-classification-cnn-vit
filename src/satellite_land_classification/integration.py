from __future__ import annotations

from pathlib import Path

from .evaluate import write_classification_report, write_comparison_csv, write_metrics_json


def persist_evaluation_bundle(
    metrics_by_model: dict,
    output_dir: str | Path,
    metrics_json: str | Path,
    comparison_csv: str | Path,
    classification_report_txt: str | Path,
) -> dict[str, str]:
    """Persist the core evaluation bundle to disk."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return {
        "metrics_json": str(write_metrics_json(metrics_by_model, metrics_json)),
        "comparison_csv": str(write_comparison_csv(metrics_by_model, comparison_csv)),
        "classification_report_txt": str(
            write_classification_report(metrics_by_model, classification_report_txt)
        ),
    }

