# Experiment Limitation Matrix

This matrix provides structured disclosure of known experimental limitations,
reviewer risks, safe wording guidance, and future work directions for each
limitation. It is a mandatory companion to the metric provenance matrix.

---

## Limitation 1 — Dataset Not Bundled

| Field | Detail |
|---|---|
| **Limitation** | The `images_dataSAT` archive is not committed to this repository. |
| **Why it matters** | Fresh training runs, evaluation reruns, and prediction examples all require the local dataset to be prepared first. Reviewers and collaborators cannot immediately reproduce results without running the data preparation script. |
| **Reviewer risk** | A reviewer who clones the repository without reading the data instructions may incorrectly conclude that the codebase is non-functional. |
| **Safe wording** | "The dataset is not bundled in this repository. It is publicly available and can be downloaded using `python scripts/run_prepare_data.py --config configs/data.yaml`." |
| **Future work** | Add a dataset download helper that validates the archive checksum after extraction. Document the expected file count and size in `data/README.md`. |

---

## Limitation 2 — Checkpoints Not Bundled

| Field | Detail |
|---|---|
| **Limitation** | Model weight files (`keras_cnn_best.keras`, `pytorch_cnn_best.pth`, etc.) are not committed to this repository. |
| **Why it matters** | Evaluation and prediction scripts require a local model file. The preserved comparison scores may reflect externally referenced pretrained checkpoints used in the evaluation notebooks, not only locally trained snapshots. |
| **Reviewer risk** | A reviewer may be unable to reproduce the preserved comparison metrics without access to the specific external checkpoints referenced in the evaluation notebooks. |
| **Safe wording** | "Checkpoints are not bundled in this repository. To reproduce the preserved evaluation scores, either train compatible checkpoints locally using the provided scripts, or place compatible external checkpoints into `models/` as described in `models/README.md`." |
| **Future work** | Add checkpoint download helpers that retrieve the externally referenced evaluation weights with provenance documentation. |

---

## Limitation 3 — Binary Task Only

| Field | Detail |
|---|---|
| **Limitation** | The implemented task is binary classification: agricultural (`class_1_agri`) vs. non-agricultural (`class_0_non_agri`). It is not a general multi-class land-cover taxonomy. |
| **Why it matters** | Binary classification is an important but narrow problem scope. The preserved results and model designs are not validated for multi-class settings. |
| **Reviewer risk** | A reviewer may over-interpret the preserved results as indicative of performance on richer land-cover benchmarks (e.g., LandCover.ai, BigEarthNet). |
| **Safe wording** | "This is a binary satellite image classification task. The preserved results apply only to the `class_0_non_agri` vs. `class_1_agri` binary setting and do not generalize to multi-class land-cover taxonomies." |
| **Future work** | Extend the dataset and model heads to support multi-class land-cover classification. |

---

## Limitation 4 — Class Folder Names Are Fixed

| Field | Detail |
|---|---|
| **Limitation** | The class folder names `class_0_non_agri` and `class_1_agri` are preserved exactly as they appeared in the original experiment notebooks. |
| **Why it matters** | The data loading scripts, configs, and tests all depend on these exact folder names. Renaming them without updating all dependent paths would break the pipeline. |
| **Reviewer risk** | Low, but a reviewer unfamiliar with the project history may question the non-standard class folder naming convention. |
| **Safe wording** | "The class folder names `class_0_non_agri` and `class_1_agri` are preserved from the original experiment records and are the actual names used throughout the codebase." |
| **Future work** | Add a config-driven class name mapping to decouple folder names from label names in the data loading layer. |

---

## Limitation 5 — Preserved Metrics Are Historical Artifacts

| Field | Detail |
|---|---|
| **Limitation** | All metrics in the README comparison table, `results/metrics.json`, and `results/model_comparison.csv` come from preserved experiment notebooks. They were not re-executed in this evidence-pack pass. |
| **Why it matters** | The preserved scores cannot be called "newly reproduced" because no rerun was performed. The scores reflect the specific configurations, hardware, and checkpoint states present during the original notebook runs. |
| **Reviewer risk** | A reviewer may interpret the preserved metrics as current benchmark results. This would be misleading without the explicit "preserved historical evaluation artifact" qualifier. |
| **Safe wording** | "These are preserved historical evaluation artifacts extracted from experiment notebooks. They have not been re-executed in the current environment." |
| **Future work** | Run end-to-end reruns from the refactored scripts once checkpoints and dataset are available locally, and add a `results/fresh_run/` subtree for newly generated artifacts. |

---

## Limitation 6 — Some Evaluation Weights Are Externally Referenced

| Field | Detail |
|---|---|
| **Limitation** | The `models/README.md` explicitly states that the preserved framework-comparison records rely on external pretrained checkpoints in addition to short local training runs. Those binaries are not bundled here. |
| **Why it matters** | The strongest preserved scores (especially the Module 4 hybrid evaluation) may reflect models that cannot be reproduced without access to those external weights. |
| **Reviewer risk** | High. A reviewer who trains from scratch using the provided scripts and short epoch counts may observe lower scores than the preserved comparison metrics. |
| **Safe wording** | "Some preserved comparison scores reflect evaluation notebooks that loaded externally referenced pretrained checkpoints. Those checkpoints are not bundled in this repository." |
| **Future work** | Document the external checkpoint sources with provenance and add download helpers. |

---

## Limitation 7 — No Fresh Retraining in This Pass

| Field | Detail |
|---|---|
| **Limitation** | No model training was performed during the evidence-pack pass. The validation tool and tests intentionally do not require the dataset or checkpoints. |
| **Why it matters** | This is an evidence, documentation, and reproducibility-strengthening pass only. All new files are documentation or validation scripts, not training artifacts. |
| **Reviewer risk** | Low, provided the pass is clearly labeled as a documentation pass. |
| **Safe wording** | "No model training was performed in this evidence-pack pass. All metrics cited are preserved historical evaluation artifacts." |
| **Future work** | Schedule a fresh end-to-end rerun pass once local dataset and checkpoint access is confirmed. |

---

## Limitation 8 — No Geospatial Metadata-Aware Analysis

| Field | Detail |
|---|---|
| **Limitation** | The current pipeline processes image patches without geospatial metadata (coordinates, acquisition time, sensor type, spatial resolution metadata). |
| **Why it matters** | Geospatial metadata can be critical for real-world remote-sensing applications. Its absence limits the practical scope of the current pipeline. |
| **Reviewer risk** | Medium. A reviewer focused on operational remote sensing may view the absence of geospatial context as a significant gap. |
| **Safe wording** | "The current implementation does not incorporate geospatial metadata. This is a known limitation and a direction for future work." |
| **Future work** | Integrate coordinate bounds, acquisition timestamps, and sensor metadata into a multimodal feature pipeline. |

---

## Limitation 9 — No Explainability Artifacts Generated

| Field | Detail |
|---|---|
| **Limitation** | No Grad-CAM maps, attention visualizations, saliency maps, or other explainability outputs have been generated for any model in this repository. |
| **Why it matters** | Explainability is increasingly important in remote-sensing research to validate that models are attending to land-cover features rather than acquisition artifacts. |
| **Reviewer risk** | Medium. Reviewers from a responsible AI or applied remote-sensing background may flag the absence of any model interpretability analysis. |
| **Safe wording** | "No explainability maps are generated in this pass. The `EXPLAINABILITY_PROTOCOL.md` document defines the required inputs and steps for generating interpretability artifacts in a future pass." |
| **Future work** | Generate Grad-CAM overlays for CNN baselines and attention/token visualizations for CNN-ViT hybrids once local checkpoints and sample images are available. |

---

## Limitation 10 — CNN-ViT Is a Hybrid, Not a Pure ViT

| Field | Detail |
|---|---|
| **Limitation** | The transformer stage operates on CNN feature tokens, not raw image patches. This is a CNN-ViT hybrid design, not a pure Vision Transformer (ViT) in the Dosovitskiy et al. sense. |
| **Why it matters** | Describing the model as "a ViT" without qualification would be technically inaccurate. The inductive biases, token construction method, and receptive field characteristics differ substantially from a patch-based ViT. |
| **Reviewer risk** | Medium. A reviewer specializing in Vision Transformers would immediately notice if the hybrid design were mischaracterized as a pure ViT. |
| **Safe wording** | "The CNN-ViT models in this repository are CNN-backed transformer hybrids. The transformer encoder operates on feature tokens produced by a CNN backbone, not on raw image patches." |
| **Future work** | Experiment with patch-based ViT variants (with appropriate pre-training or dataset size considerations) for comparison. |

---

## Limitation 11 — High Metrics Require Careful Provenance Wording

| Field | Detail |
|---|---|
| **Limitation** | Preserved accuracy values above 0.99 and ROC-AUC values of 1.0000 are unusually high and may raise reviewer skepticism without proper provenance context. |
| **Why it matters** | Very high metrics on a balanced binary task with a specific dataset are plausible but require careful qualification. Without provenance disclosure, reviewers may suspect data leakage, label imbalance masking, or evaluation methodology issues. |
| **Reviewer risk** | High. Without the provenance matrix and limitation disclosures, the preserved metrics could damage credibility rather than strengthen it. |
| **Safe wording** | "The preserved metrics reflect historical evaluation artifacts from the experiment notebooks. The balanced binary task, the specific dataset, and the possibility that evaluation notebooks used externally referenced pretrained checkpoints all contribute to the high scores. These results should be interpreted in that context." |
| **Future work** | Perform a fresh end-to-end rerun with full provenance logging to generate independently verifiable results. |
