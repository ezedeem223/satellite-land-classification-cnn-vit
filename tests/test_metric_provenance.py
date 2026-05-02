"""Tests that verify metric provenance: results artifacts are parseable,
contain expected model rows, and metrics are consistent with documented values.

Does not require the dataset or model checkpoints.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESEARCH_PACK_DIR = ROOT / "docs" / "research_pack"

EXPECTED_CSV_MODELS = {
    "keras_cnn",
    "pytorch_cnn",
    "keras_cnn_vit",
    "pytorch_cnn_vit",
}

EXPECTED_CSV_COLUMNS = {
    "model",
    "family",
    "framework",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "source_notebook",
}

KNOWN_METRICS = {
    "keras_cnn": {
        "accuracy": 0.9925,
        "precision": 1.0,
        "recall": 0.985,
        "f1_score": 0.9924,
        "roc_auc": 1.0,
    },
    "pytorch_cnn": {
        "accuracy": 0.9988,
        "precision": 0.9983,
        "recall": 0.9993,
        "f1_score": 0.9988,
        "roc_auc": 1.0,
    },
    "keras_cnn_vit": {
        "accuracy": 0.9958,
        "precision": 0.999,
        "recall": 0.9927,
        "f1_score": 0.9958,
        "roc_auc": 0.9998,
    },
    "pytorch_cnn_vit": {
        "accuracy": 0.999,
        "precision": 0.999,
        "recall": 0.999,
        "f1_score": 0.999,
        "roc_auc": 1.0,
    },
}


def test_metrics_json_is_parseable():
    path = RESULTS_DIR / "metrics.json"
    assert path.exists(), "results/metrics.json must exist"
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict), "results/metrics.json top-level must be a dict"


def test_metrics_json_has_historical_notebook_metrics():
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert "historical_notebook_metrics" in data, (
        "results/metrics.json must contain 'historical_notebook_metrics' key"
    )


def test_metrics_json_module2_cnn_comparison_present():
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    hnm = data.get("historical_notebook_metrics", {})
    assert "module2_cnn_comparison" in hnm, (
        "results/metrics.json must contain 'module2_cnn_comparison'"
    )
    m2 = hnm["module2_cnn_comparison"]
    assert "keras_cnn" in m2, "module2_cnn_comparison must contain 'keras_cnn'"
    assert "pytorch_cnn" in m2, "module2_cnn_comparison must contain 'pytorch_cnn'"


def test_metrics_json_module4_hybrid_comparison_present():
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    hnm = data.get("historical_notebook_metrics", {})
    assert "module4_hybrid_comparison" in hnm, (
        "results/metrics.json must contain 'module4_hybrid_comparison'"
    )
    m4 = hnm["module4_hybrid_comparison"]
    assert "keras_cnn_vit" in m4, (
        "module4_hybrid_comparison must contain 'keras_cnn_vit'"
    )
    assert "pytorch_cnn_vit" in m4, (
        "module4_hybrid_comparison must contain 'pytorch_cnn_vit'"
    )


def test_metrics_json_values_match_known_metrics():
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    hnm = data.get("historical_notebook_metrics", {})
    m2 = hnm.get("module2_cnn_comparison", {})
    m4 = hnm.get("module4_hybrid_comparison", {})
    json_metrics = {
        "keras_cnn": m2.get("keras_cnn", {}),
        "pytorch_cnn": m2.get("pytorch_cnn", {}),
        "keras_cnn_vit": m4.get("keras_cnn_vit", {}),
        "pytorch_cnn_vit": m4.get("pytorch_cnn_vit", {}),
    }
    for model, expected in KNOWN_METRICS.items():
        actual = json_metrics.get(model, {})
        for metric, expected_value in expected.items():
            actual_value = actual.get(metric)
            assert actual_value is not None, (
                f"results/metrics.json: missing '{metric}' for '{model}'"
            )
            assert abs(float(actual_value) - expected_value) < 1e-6, (
                f"results/metrics.json: '{model}.{metric}' = {actual_value}, "
                f"expected {expected_value}"
            )


def test_model_comparison_csv_is_parseable():
    path = RESULTS_DIR / "model_comparison.csv"
    assert path.exists(), "results/model_comparison.csv must exist"
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) > 0, "results/model_comparison.csv must not be empty"


def test_model_comparison_csv_has_expected_columns():
    path = RESULTS_DIR / "model_comparison.csv"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
    missing = EXPECTED_CSV_COLUMNS - fieldnames
    assert not missing, (
        f"results/model_comparison.csv missing columns: {sorted(missing)}"
    )


def test_model_comparison_csv_has_four_model_rows():
    path = RESULTS_DIR / "model_comparison.csv"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 4, (
        f"results/model_comparison.csv must have exactly 4 model rows, "
        f"found {len(rows)}"
    )


def test_model_comparison_csv_has_expected_model_names():
    path = RESULTS_DIR / "model_comparison.csv"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        found_models = {row["model"].strip() for row in reader}
    missing = EXPECTED_CSV_MODELS - found_models
    assert not missing, (
        f"results/model_comparison.csv missing expected model rows: {sorted(missing)}"
    )


def test_model_comparison_csv_values_match_known_metrics():
    path = RESULTS_DIR / "model_comparison.csv"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = {row["model"].strip(): row for row in reader}
    for model, expected in KNOWN_METRICS.items():
        if model not in rows:
            continue
        row = rows[model]
        for metric, expected_value in expected.items():
            csv_key = metric
            if csv_key not in row:
                continue
            actual_value = float(row[csv_key])
            assert abs(actual_value - expected_value) < 1e-6, (
                f"model_comparison.csv: '{model}.{metric}' = {actual_value}, "
                f"expected {expected_value}"
            )


def test_csv_model_names_referenced_in_provenance_matrix():
    csv_path = RESULTS_DIR / "model_comparison.csv"
    matrix_path = RESEARCH_PACK_DIR / "METRIC_PROVENANCE_MATRIX.md"
    if not csv_path.exists() or not matrix_path.exists():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        csv_models = [row["model"].strip() for row in reader]
    matrix_text = matrix_path.read_text(encoding="utf-8").lower()
    missing = [m for m in csv_models if m.lower() not in matrix_text]
    assert not missing, (
        f"Models from model_comparison.csv not referenced in "
        f"METRIC_PROVENANCE_MATRIX.md: {missing}"
    )
