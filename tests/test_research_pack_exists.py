"""Tests that verify all required research evidence pack files exist and contain
no forbidden overclaim phrases.

Does not require the dataset or model checkpoints.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_PACK_DIR = ROOT / "docs" / "research_pack"

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

FORBIDDEN_PHRASES = [
    "state-of-the-art",
    "production deployment",
    "operational agricultural monitoring",
    "validated real-world deployment",
    "real-time monitoring platform",
]

REQUIRED_PACK_PHRASES = [
    "dataset is not bundled",
    "checkpoints are not bundled",
    "binary",
    "CNN-ViT hybrid",
    "preserved",
]


def test_research_pack_directory_exists():
    assert RESEARCH_PACK_DIR.exists(), (
        f"docs/research_pack/ directory does not exist at {RESEARCH_PACK_DIR}"
    )
    assert RESEARCH_PACK_DIR.is_dir()


def test_all_required_pack_files_exist():
    missing = []
    for filename in REQUIRED_PACK_FILES:
        path = RESEARCH_PACK_DIR / filename
        if not path.exists():
            missing.append(filename)
    assert not missing, (
        f"Missing research pack files in docs/research_pack/: {missing}"
    )


def test_pack_files_are_non_empty():
    for filename in REQUIRED_PACK_FILES:
        path = RESEARCH_PACK_DIR / filename
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            assert len(content) > 0, f"Research pack file is empty: {filename}"


def test_forbidden_phrases_absent_from_pack():
    collected_text = ""
    for filename in REQUIRED_PACK_FILES:
        path = RESEARCH_PACK_DIR / filename
        if path.exists():
            collected_text += path.read_text(encoding="utf-8").lower() + "\n"

    found = [p for p in FORBIDDEN_PHRASES if p.lower() in collected_text]
    assert not found, (
        f"Forbidden overclaim phrases found in research pack docs: {found}"
    )


def test_required_phrases_present_in_pack():
    collected_text = ""
    for filename in REQUIRED_PACK_FILES:
        path = RESEARCH_PACK_DIR / filename
        if path.exists():
            collected_text += path.read_text(encoding="utf-8").lower() + "\n"

    missing = [p for p in REQUIRED_PACK_PHRASES if p.lower() not in collected_text]
    assert not missing, (
        f"Required limitation phrases absent from research pack docs: {missing}"
    )


def test_explainability_protocol_states_no_maps_generated():
    path = RESEARCH_PACK_DIR / "EXPLAINABILITY_PROTOCOL.md"
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8").lower()
    assert "no explainability maps are generated in this pass" in content, (
        "EXPLAINABILITY_PROTOCOL.md must clearly state that no maps are generated "
        "in this pass unless real assets exist."
    )


def test_validation_tool_exists():
    tool_path = ROOT / "tools" / "evidence" / "validate_research_pack.py"
    assert tool_path.exists(), (
        "tools/evidence/validate_research_pack.py does not exist"
    )


def test_validation_tool_passes():
    tool_path = ROOT / "tools" / "evidence" / "validate_research_pack.py"
    if not tool_path.exists():
        return
    result = subprocess.run(
        [sys.executable, str(tool_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"validate_research_pack.py exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_results_metrics_json_exists():
    path = ROOT / "results" / "metrics.json"
    assert path.exists(), "results/metrics.json does not exist"


def test_results_model_comparison_csv_exists():
    path = ROOT / "results" / "model_comparison.csv"
    assert path.exists(), "results/model_comparison.csv does not exist"
