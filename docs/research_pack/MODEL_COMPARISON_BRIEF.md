# Model Comparison Brief

## Source Artifacts

All metrics in this brief are drawn exclusively from preserved historical evaluation
artifacts already present in this repository. No new metrics have been generated
in this evidence-pack pass.

| Artifact | Path |
|---|---|
| Structured metrics | `results/metrics.json` |
| Comparison table | `results/model_comparison.csv` |
| Classification reports | `results/classification_report.txt` |

---

## Important Cautions

1. **Preserved, not newly reproduced.** The scores below come from experiment
   notebooks run during earlier project phases. They have not been re-executed in
   this evidence-pack pass.

2. **Checkpoints not bundled.** Model weight files are not committed to this
   repository. The preserved comparison scores may reflect externally referenced
   pretrained checkpoints used in the evaluation notebooks, in addition to any
   locally trained snapshots.

3. **Dataset not bundled.** Fresh reproduction requires the `images_dataSAT`
   archive to be prepared locally via `python scripts/run_prepare_data.py`.

4. **Historical artifact confidence.** Metrics marked as "verified from
   `results/metrics.json`" and "verified from `results/model_comparison.csv`"
   match the values in those artifacts exactly. They are not production scores
   and do not imply dataset-independent generalization.

---

## Comparison Table

| Model | Family | Framework | Accuracy | Precision | Recall | F1 | ROC-AUC | Source Artifact |
|---|---|---|---:|---:|---:|---:|---:|---|
| Keras CNN | CNN | Keras | 0.9925 | 1.0000 | 0.9850 | 0.9924 | 1.0000 | `results/metrics.json` → `module2_cnn_comparison.keras_cnn`; `results/model_comparison.csv` row 1 |
| PyTorch CNN | CNN | PyTorch | 0.9988 | 0.9983 | 0.9993 | 0.9988 | 1.0000 | `results/metrics.json` → `module2_cnn_comparison.pytorch_cnn`; `results/model_comparison.csv` row 2 |
| Keras CNN-ViT Hybrid | CNN-ViT Hybrid | Keras | 0.9958 | 0.9990 | 0.9927 | 0.9958 | 0.9998 | `results/metrics.json` → `module4_hybrid_comparison.keras_cnn_vit`; `results/model_comparison.csv` row 3 |
| PyTorch CNN-ViT Hybrid | CNN-ViT Hybrid | PyTorch | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 1.0000 | `results/metrics.json` → `module4_hybrid_comparison.pytorch_cnn_vit`; `results/model_comparison.csv` row 4 |

---

## Model Family Descriptions

### CNN Baselines

The CNN baselines apply convolutional feature extraction directly to input image
patches. Convolutional layers learn local spatial patterns — edges, textures, and
regional intensity gradients — that are highly informative for distinguishing
agricultural from non-agricultural terrain at 64×64 resolution.

Both the Keras and PyTorch CNN implementations use equivalent architectural
principles, making their preserved scores directly comparable as a cross-framework
consistency check.

### CNN-ViT Hybrid Models

The CNN-ViT hybrid models are **not** pure Vision Transformers. The transformer
encoder operates on feature tokens produced by a CNN backbone, not directly on raw
image patches. This is a key architectural distinction:

- The CNN backbone extracts local spatial features (same role as in the CNN baseline)
- The transformer encoder applies multi-head self-attention across the resulting
  token sequence, capturing global contextual relationships between spatial regions
- A classifier head produces the final binary prediction

This design is specifically a CNN-backed transformer hybrid, and the preserved
results should not be interpreted as a benchmark for standalone ViT architectures.

---

## Interpretation

- **PyTorch CNN-ViT Hybrid** achieves the highest preserved accuracy (0.9990) and
  is the strongest single model in the preserved evaluation record.
- **PyTorch CNN baseline** outperforms the Keras CNN baseline (0.9988 vs 0.9925),
  consistent with the shorter Keras training snapshots showing lower validation
  accuracy in the early-epoch records.
- **Keras CNN-ViT Hybrid** achieves 0.9958 accuracy with a slightly lower ROC-AUC
  (0.9998) compared to the PyTorch variants, which both reach ROC-AUC of 1.0000.
- All four models show strong preserved performance on this balanced binary task.

---

## What These Results Do Not Prove

- Generalization to satellite imagery from different sensors, resolutions, or
  geographic regions not present in `images_dataSAT`
- Superiority over other remote-sensing architectures not tested here
- Production-readiness or deployment suitability
- Equivalent performance if retrained from scratch with different seeds or hardware
- Multi-class land-cover classification capability

---

## Cross-Framework Comparison Value

The parallel Keras and PyTorch implementations serve as a framework consistency
check. Comparable performance across both frameworks increases confidence that the
preserved results reflect genuine model behavior on this dataset rather than
framework-specific artifacts.
