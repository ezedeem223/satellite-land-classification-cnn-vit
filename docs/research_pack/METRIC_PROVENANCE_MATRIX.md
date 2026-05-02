# Metric Provenance Matrix

This matrix traces every metric cited in this repository back to its source artifact.
It is the authoritative traceability record for the evidence pack.

---

## Red Lines

- Do not call preserved metrics "newly reproduced" unless rerun commands were
  actually executed in the current environment with the current codebase.
- Do not call these production scores.
- Do not imply dataset-independent generalization.
- Do not imply broader multi-class land-cover classification capability.
- Do not present training snapshot metrics as final evaluation metrics.

---

## Evidence Confidence Key

| Level | Meaning |
|---|---|
| **verified from results/metrics.json** | Value confirmed present in `results/metrics.json` with matching key path |
| **verified from results/model_comparison.csv** | Value confirmed present in `results/model_comparison.csv` with matching row |
| **verified from results/classification_report.txt** | Value confirmed present in `results/classification_report.txt` |
| **README-only** | Value appears in `README.md` but not cross-checked against a structured artifact |
| **unverified / do not state** | Value has no confirmed source in the repository |

---

## Module 2 — CNN Comparison (Keras CNN)

Source notebook: `Lab_M2L3_Comparative_Analysis_of_Keras_and_PyTorch_Models (1).ipynb`

| Model | Family | Framework | Metric | Value | results/metrics.json key | CSV row | Classification report | Evidence confidence | Allowed wording | Forbidden wording |
|---|---|---|---|---|---|---|---|---|---|---|
| Keras CNN | CNN | Keras | accuracy | 0.9925 | `historical_notebook_metrics.module2_cnn_comparison.keras_cnn.accuracy` | row 1 | [Module 2] Keras CNN | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved accuracy of 0.9925 from historical evaluation" | "newly measured", "live accuracy", superlative benchmark claims |
| Keras CNN | CNN | Keras | precision | 1.0000 | `historical_notebook_metrics.module2_cnn_comparison.keras_cnn.precision` | row 1 | [Module 2] Keras CNN | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved precision of 1.0000 from historical evaluation" | "perfect precision in a live system", superlative benchmark claims |
| Keras CNN | CNN | Keras | recall | 0.9850 | `historical_notebook_metrics.module2_cnn_comparison.keras_cnn.recall` | row 1 | [Module 2] Keras CNN | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved recall of 0.9850 from historical evaluation" | "live recall", superlative benchmark claims |
| Keras CNN | CNN | Keras | f1_score | 0.9924 | `historical_notebook_metrics.module2_cnn_comparison.keras_cnn.f1_score` | row 1 | [Module 2] Keras CNN | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved F1 of 0.9924 from historical evaluation" | "live F1", superlative benchmark claims |
| Keras CNN | CNN | Keras | roc_auc | 1.0000 | `historical_notebook_metrics.module2_cnn_comparison.keras_cnn.roc_auc` | row 1 | [Module 2] Keras CNN | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved ROC-AUC of 1.0000 from historical evaluation" | "perfect AUC in a live system", superlative benchmark claims |
| Keras CNN | CNN | Keras | loss | 0.0247 | `historical_notebook_metrics.module2_cnn_comparison.keras_cnn.loss` | — | [Module 2] Keras CNN | verified from results/metrics.json | "preserved loss of 0.0247 from historical evaluation notebook" | "live loss", superlative benchmark claims |
| Keras CNN | CNN | Keras | source_cell | 47 | `historical_notebook_metrics.module2_cnn_comparison.keras_cnn.source_cell` | — | — | verified from results/metrics.json | "source cell 47 of the comparison notebook" | "independently verified cell" |

---

## Module 2 — CNN Comparison (PyTorch CNN)

Source notebook: `Lab_M2L3_Comparative_Analysis_of_Keras_and_PyTorch_Models (1).ipynb`

| Model | Family | Framework | Metric | Value | results/metrics.json key | CSV row | Classification report | Evidence confidence | Allowed wording | Forbidden wording |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch CNN | CNN | PyTorch | accuracy | 0.9988 | `historical_notebook_metrics.module2_cnn_comparison.pytorch_cnn.accuracy` | row 2 | [Module 2] PyTorch CNN | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved accuracy of 0.9988 from historical evaluation" | "newly measured", "live accuracy", superlative benchmark claims |
| PyTorch CNN | CNN | PyTorch | precision | 0.9983 | `historical_notebook_metrics.module2_cnn_comparison.pytorch_cnn.precision` | row 2 | [Module 2] PyTorch CNN | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved precision of 0.9983 from historical evaluation" | "live precision", superlative benchmark claims |
| PyTorch CNN | CNN | PyTorch | recall | 0.9993 | `historical_notebook_metrics.module2_cnn_comparison.pytorch_cnn.recall` | row 2 | [Module 2] PyTorch CNN | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved recall of 0.9993 from historical evaluation" | "live recall", superlative benchmark claims |
| PyTorch CNN | CNN | PyTorch | f1_score | 0.9988 | `historical_notebook_metrics.module2_cnn_comparison.pytorch_cnn.f1_score` | row 2 | [Module 2] PyTorch CNN | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved F1 of 0.9988 from historical evaluation" | "live F1", superlative benchmark claims |
| PyTorch CNN | CNN | PyTorch | roc_auc | 1.0000 | `historical_notebook_metrics.module2_cnn_comparison.pytorch_cnn.roc_auc` | row 2 | [Module 2] PyTorch CNN | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved ROC-AUC of 1.0000 from historical evaluation" | "live AUC", superlative benchmark claims |
| PyTorch CNN | CNN | PyTorch | loss | 0.0024 | `historical_notebook_metrics.module2_cnn_comparison.pytorch_cnn.loss` | — | [Module 2] PyTorch CNN | verified from results/metrics.json | "preserved loss of 0.0024 from historical evaluation notebook" | "live loss", superlative benchmark claims |
| PyTorch CNN | CNN | PyTorch | source_cell | 57 | `historical_notebook_metrics.module2_cnn_comparison.pytorch_cnn.source_cell` | — | — | verified from results/metrics.json | "source cell 57 of the comparison notebook" | "independently verified cell" |

---

## Module 4 — Hybrid Evaluation (Keras CNN-ViT)

Source notebook: `lab_M4L1_Land_Classification_CNN-ViT_Integration_Evaluation.ipynb`

| Model | Family | Framework | Metric | Value | results/metrics.json key | CSV row | Classification report | Evidence confidence | Allowed wording | Forbidden wording |
|---|---|---|---|---|---|---|---|---|---|---|
| Keras CNN-ViT Hybrid | CNN-ViT Hybrid | Keras | accuracy | 0.9958 | `historical_notebook_metrics.module4_hybrid_comparison.keras_cnn_vit.accuracy` | row 3 | [Module 4] Keras CNN-ViT Hybrid | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved accuracy of 0.9958 from historical hybrid evaluation" | "newly measured", "live accuracy", superlative benchmark claims |
| Keras CNN-ViT Hybrid | CNN-ViT Hybrid | Keras | precision | 0.9990 | `historical_notebook_metrics.module4_hybrid_comparison.keras_cnn_vit.precision` | row 3 | [Module 4] Keras CNN-ViT Hybrid | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved precision of 0.9990 from historical hybrid evaluation" | "live precision", superlative benchmark claims |
| Keras CNN-ViT Hybrid | CNN-ViT Hybrid | Keras | recall | 0.9927 | `historical_notebook_metrics.module4_hybrid_comparison.keras_cnn_vit.recall` | row 3 | [Module 4] Keras CNN-ViT Hybrid | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved recall of 0.9927 from historical hybrid evaluation" | "live recall", superlative benchmark claims |
| Keras CNN-ViT Hybrid | CNN-ViT Hybrid | Keras | f1_score | 0.9958 | `historical_notebook_metrics.module4_hybrid_comparison.keras_cnn_vit.f1_score` | row 3 | [Module 4] Keras CNN-ViT Hybrid | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved F1 of 0.9958 from historical hybrid evaluation" | "live F1", superlative benchmark claims |
| Keras CNN-ViT Hybrid | CNN-ViT Hybrid | Keras | roc_auc | 0.9998 | `historical_notebook_metrics.module4_hybrid_comparison.keras_cnn_vit.roc_auc` | row 3 | [Module 4] Keras CNN-ViT Hybrid | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved ROC-AUC of 0.9998 from historical hybrid evaluation" | "live AUC", superlative benchmark claims |
| Keras CNN-ViT Hybrid | CNN-ViT Hybrid | Keras | loss | 0.0530 | `historical_notebook_metrics.module4_hybrid_comparison.keras_cnn_vit.loss` | — | [Module 4] Keras CNN-ViT Hybrid | verified from results/metrics.json | "preserved loss of 0.0530 from historical hybrid evaluation notebook" | "live loss", superlative benchmark claims |
| Keras CNN-ViT Hybrid | CNN-ViT Hybrid | Keras | source_cell | 56 | `historical_notebook_metrics.module4_hybrid_comparison.keras_cnn_vit.source_cell` | — | — | verified from results/metrics.json | "source cell 56 of the hybrid evaluation notebook" | "independently verified cell" |

---

## Module 4 — Hybrid Evaluation (PyTorch CNN-ViT)

Source notebook: `lab_M4L1_Land_Classification_CNN-ViT_Integration_Evaluation.ipynb`

| Model | Family | Framework | Metric | Value | results/metrics.json key | CSV row | Classification report | Evidence confidence | Allowed wording | Forbidden wording |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch CNN-ViT Hybrid | CNN-ViT Hybrid | PyTorch | accuracy | 0.9990 | `historical_notebook_metrics.module4_hybrid_comparison.pytorch_cnn_vit.accuracy` | row 4 | [Module 4] PyTorch CNN-ViT Hybrid | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved accuracy of 0.9990 from historical hybrid evaluation" | "newly measured", "live accuracy", superlative benchmark claims |
| PyTorch CNN-ViT Hybrid | CNN-ViT Hybrid | PyTorch | precision | 0.9990 | `historical_notebook_metrics.module4_hybrid_comparison.pytorch_cnn_vit.precision` | row 4 | [Module 4] PyTorch CNN-ViT Hybrid | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved precision of 0.9990 from historical hybrid evaluation" | "live precision", superlative benchmark claims |
| PyTorch CNN-ViT Hybrid | CNN-ViT Hybrid | PyTorch | recall | 0.9990 | `historical_notebook_metrics.module4_hybrid_comparison.pytorch_cnn_vit.recall` | row 4 | [Module 4] PyTorch CNN-ViT Hybrid | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved recall of 0.9990 from historical hybrid evaluation" | "live recall", superlative benchmark claims |
| PyTorch CNN-ViT Hybrid | CNN-ViT Hybrid | PyTorch | f1_score | 0.9990 | `historical_notebook_metrics.module4_hybrid_comparison.pytorch_cnn_vit.f1_score` | row 4 | [Module 4] PyTorch CNN-ViT Hybrid | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved F1 of 0.9990 from historical hybrid evaluation" | "live F1", superlative benchmark claims |
| PyTorch CNN-ViT Hybrid | CNN-ViT Hybrid | PyTorch | roc_auc | 1.0000 | `historical_notebook_metrics.module4_hybrid_comparison.pytorch_cnn_vit.roc_auc` | row 4 | [Module 4] PyTorch CNN-ViT Hybrid | verified from results/metrics.json; verified from results/model_comparison.csv | "preserved ROC-AUC of 1.0000 from historical hybrid evaluation" | "live AUC", superlative benchmark claims |
| PyTorch CNN-ViT Hybrid | CNN-ViT Hybrid | PyTorch | loss | 0.0047 | `historical_notebook_metrics.module4_hybrid_comparison.pytorch_cnn_vit.loss` | — | [Module 4] PyTorch CNN-ViT Hybrid | verified from results/metrics.json | "preserved loss of 0.0047 from historical hybrid evaluation notebook" | "live loss", superlative benchmark claims |
| PyTorch CNN-ViT Hybrid | CNN-ViT Hybrid | PyTorch | source_cell | 60 | `historical_notebook_metrics.module4_hybrid_comparison.pytorch_cnn_vit.source_cell` | — | — | verified from results/metrics.json | "source cell 60 of the hybrid evaluation notebook" | "independently verified cell" |

---

## Training Snapshot Metrics (Not Comparison Metrics)

The `results/metrics.json` file also contains `training_snapshots` for all four
models covering short 3–5 epoch local training runs from the module-specific
notebooks. These are **distinct from the comparison metrics** above.

Key distinction: the preserved comparison metrics (Module 2 and Module 4 entries
above) reflect evaluation runs that may have used externally referenced pretrained
checkpoints, not only the short local training snapshots. The training snapshots
show lower validation accuracy in some cases (e.g., Keras CNN validation accuracy
of 0.5233 in the short run vs. preserved comparison score of 0.9925).

Do not conflate training snapshot metrics with the preserved comparison metrics.

---

## Unverified Claims

Any metric not appearing in `results/metrics.json`, `results/model_comparison.csv`,
or `results/classification_report.txt` is marked **unverified / do not state** and
must not be cited in research communications without independent verification.
