"""Validation tool for the satellite research evidence pack.

Checks that all required research pack files exist, results artifacts are present,
forbidden overclaim phrases are absent, and required limitation phrases are present.

Does not require the dataset or model checkpoints.

Usage:
    python tools/evidence/validate_research_pack.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RESEARCH_PACK_DIR = ROOT / "docs" / "research_pack"
RESULTS_DIR = ROOT / "results"

REQUIRED_PACK_FILES = [
    "README.md",
    "PROJECT_BRIEF_KAUST.md",
    "MODEL_COMPARISON_BRIEF.md",
    "METRIC_PROVENANCE_MATRIX.md",
    "EXPERIMENT_LIMITATION_MATRIX.md",
    "CNN_VS_CNN_VIT_EXPLAINER.md",
    "DATASET_AND_TASK_CARD.md",
    "EXPLAINABILITY_PROTOCOL.md",
    "REPRODUCIBILITY_CHECKLIST.md",
]

REQUIRED_RESULTS_FILES = [
    "metrics.json",
    "model_comparison.csv",
]

EXPECTED_CSV_MODELS = {
    "keras_cnn",
    "pytorch_cnn",
    "keras_cnn_vit",
    "pytorch_cnn_vit",
}

FORBIDDEN_PHRASES = [
    "state-of-the-art",
    "production deployment",
    "operational agricultural monitoring",
    "validated real-world deployment",
    "real-time monitoring platform",
]

REQUIRED_PHRASES = [
    "dataset is not bundled",
    "checkpoints are not bundled",
    "binary",
    "CNN-ViT hybrid",
    "preserved",
]


def check_pack_files_exist() -> list[str]:
    failures: list[str] = []
    for filename in REQUIRED_PACK_FILES:
        path = RESEARCH_PACK_DIR / filename
        if not path.exists():
            failures.append(f"MISSING research pack file: docs/research_pack/{filename}")
    return failures


def check_results_files_exist() -> list[str]:
    failures: list[str] = []
    for filename in REQUIRED_RESULTS_FILES:
        path = RESULTS_DIR / filename
        if not path.exists():
            failures.append(f"MISSING results artifact: results/{filename}")
    return failures


def check_metrics_json_parseable() -> list[str]:
    failures: list[str] = []
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            failures.append("results/metrics.json: top-level value is not a dict")
    except json.JSONDecodeError as exc:
        failures.append(f"results/metrics.json: JSON parse error: {exc}")
    return failures


def check_csv_models() -> list[str]:
    failures: list[str] = []
    path = RESULTS_DIR / "model_comparison.csv"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "model" not in reader.fieldnames:
                failures.append(
                    "results/model_comparison.csv: missing 'model' column"
                )
                return failures
            found_models = {row["model"].strip() for row in reader}
        missing = EXPECTED_CSV_MODELS - found_models
        for model in sorted(missing):
            failures.append(
                f"results/model_comparison.csv: expected model row not found: '{model}'"
            )
    except Exception as exc:
        failures.append(f"results/model_comparison.csv: read error: {exc}")
    return failures


def check_csv_models_in_provenance_matrix() -> list[str]:
    failures: list[str] = []
    csv_path = RESULTS_DIR / "model_comparison.csv"
    matrix_path = RESEARCH_PACK_DIR / "METRIC_PROVENANCE_MATRIX.md"
    if not csv_path.exists() or not matrix_path.exists():
        return []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "model" not in reader.fieldnames:
                return []
            csv_models = [row["model"].strip() for row in reader]
        matrix_text = matrix_path.read_text(encoding="utf-8").lower()
        for model in csv_models:
            if model.lower() not in matrix_text:
                failures.append(
                    f"METRIC_PROVENANCE_MATRIX.md: model '{model}' from "
                    f"model_comparison.csv not referenced in provenance matrix"
                )
    except Exception as exc:
        failures.append(f"CSV/provenance matrix cross-check error: {exc}")
    return failures


def _collect_pack_text() -> str:
    parts: list[str] = []
    for filename in REQUIRED_PACK_FILES:
        path = RESEARCH_PACK_DIR / filename
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def check_forbidden_phrases() -> list[str]:
    failures: list[str] = []
    pack_text_lower = _collect_pack_text().lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in pack_text_lower:
            failures.append(
                f"FORBIDDEN PHRASE found in research pack docs: '{phrase}'"
            )
    return failures


def check_required_phrases() -> list[str]:
    failures: list[str] = []
    pack_text_lower = _collect_pack_text().lower()
    for phrase in REQUIRED_PHRASES:
        if phrase.lower() not in pack_text_lower:
            failures.append(
                f"REQUIRED PHRASE absent from research pack docs: '{phrase}'"
            )
    return failures


def run_all_checks() -> int:
    all_failures: list[str] = []

    checks = [
        ("Research pack files exist", check_pack_files_exist),
        ("Results artifacts exist", check_results_files_exist),
        ("metrics.json is valid JSON", check_metrics_json_parseable),
        ("model_comparison.csv has expected model rows", check_csv_models),
        (
            "CSV model names referenced in provenance matrix",
            check_csv_models_in_provenance_matrix,
        ),
        ("Forbidden phrases absent from pack docs", check_forbidden_phrases),
        ("Required limitation phrases present in pack docs", check_required_phrases),
    ]

    for check_name, check_fn in checks:
        failures = check_fn()
        if failures:
            print(f"  FAIL  {check_name}")
            for failure in failures:
                print(f"        - {failure}")
            all_failures.extend(failures)
        else:
            print(f"  PASS  {check_name}")

    print()
    if all_failures:
        print(f"Result: {len(all_failures)} check(s) failed.")
        return 1
    print("Result: all checks passed.")
    return 0


def main() -> None:
    print("Satellite Research Evidence Pack — Validation Tool")
    print("=" * 52)
    print()
    exit_code = run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
