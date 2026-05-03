# Academic Research Brief

## Title

Satellite Agricultural Land Classification with CNNs and CNN-ViT Hybrid Models

## Repository

`satellite-land-classification-cnn-vit`
Maintainer: Mohamad Sabbagh (ezedeem223)
License: MIT
Citation: `CITATION.cff`

---

## Problem Definition

Binary remote-sensing classification: given a satellite image patch, determine
whether the depicted land is agricultural (`class_1_agri`) or non-agricultural
(`class_0_non_agri`). The task operates on the public `images_dataSAT` archive,
which provides balanced two-class image sets at a standardized 64×64×3 resolution.

---

## Why Agricultural Satellite Classification Matters

Agricultural land monitoring from remote-sensing data is a foundational capability
for food security analysis, crop inventory estimation, land-use change detection,
and environmental compliance auditing. Binary agricultural vs. non-agricultural
separation is the practical first step before expanding into richer multi-class
land-cover taxonomies. This project targets that narrower but meaningful setting,
making it a suitable entry point for learning remote-sensing deep learning pipelines
before tackling more complex geospatial datasets.

---

## Task Scope

- **Task type:** Binary image classification
- **Input:** RGB satellite image patches (64×64×3)
- **Classes:** `class_0_non_agri` (non-agricultural), `class_1_agri` (agricultural)
- **Evaluation split:** 20% validation, 80% training (seed: 7331)
- **Test set size (preserved reports):** 6,000 images per class (3,000 non-agri, 3,000 agri)
- **Task boundary:** Binary only; not a general multi-class land-cover taxonomy

---

## Technical Pipeline

```
Raw satellite image patches (images_dataSAT)
        │
        ▼
Data loading and augmentation
(Keras generator / PyTorch Dataset+DataLoader)
        │
        ▼
CNN backbone feature extraction
(local spatial features: edges, textures, patterns)
        │
        ▼
[CNN baseline] → classifier head → binary output
        │
        ▼
[CNN-ViT hybrid] → transformer encoder on CNN tokens → classifier head → binary output
```

---

## Frameworks Used

| Framework | Role |
|---|---|
| Keras (TensorFlow backend) | CNN baseline + CNN-ViT hybrid (Keras path) |
| PyTorch | CNN baseline + CNN-ViT hybrid (PyTorch path) |

Both frameworks implement equivalent architectures, enabling direct cross-framework
comparison on identical data splits.

---

## CNN Baseline Role

The CNN baselines serve two purposes:

1. **Performance baseline:** Establish a reference accuracy against which the
   hybrid transformer architecture is compared.
2. **Feature extractor:** The same convolutional backbone used in the CNN baseline
   is reused as the token producer in the CNN-ViT hybrid, making the two model
   families directly comparable in architectural lineage.

---

## CNN-ViT Hybrid Role

The CNN-ViT hybrid models are **not** pure Vision Transformers (ViT). They apply
a transformer encoder on top of CNN-produced feature tokens rather than on raw image
patches. This hybrid design:

- Preserves the spatial inductive biases of CNNs for low-level feature extraction
- Adds global context modeling via multi-head self-attention on the token sequence
- Is appropriate for moderate-scale satellite imagery where pure ViT training from
  scratch would require substantially larger datasets

---

## Current Evidence Artifacts

All metrics listed below are **preserved historical evaluation artifacts** extracted
from experiment notebooks. They are not newly generated in this evidence-pack pass.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Keras CNN | 0.9925 | 1.0000 | 0.9850 | 0.9924 | 1.0000 |
| PyTorch CNN | 0.9988 | 0.9983 | 0.9993 | 0.9988 | 1.0000 |
| Keras CNN-ViT Hybrid | 0.9958 | 0.9990 | 0.9927 | 0.9958 | 0.9998 |
| PyTorch CNN-ViT Hybrid | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 1.0000 |

Source artifacts: `results/metrics.json`, `results/model_comparison.csv`,
`results/classification_report.txt`.

---

## Limitations

- The dataset (`images_dataSAT`) is not bundled in this repository.
- Model checkpoints are not bundled in this repository.
- Some preserved comparison scores come from evaluation notebooks that loaded
  externally referenced pretrained checkpoints, not only the short local training
  snapshots.
- The transformer experiments are CNN-ViT hybrids, not pure standalone ViTs.
- The task is binary; results do not generalize to multi-class land-cover benchmarks.
- No geospatial metadata-aware analysis has been performed.
- No explainability maps (Grad-CAM, attention visualizations) have been generated.

---

## Future Research Directions

- Extend to multi-class land-cover classification using a richer geospatial dataset
- Integrate geospatial metadata (coordinate bounds, acquisition timestamps, sensor
  type) into the feature pipeline
- Add Grad-CAM visualization for CNN baselines and attention/token visualization for
  CNN-ViT hybrids (requires local checkpoints and sample images)
- Add experiment tracking (MLflow or Weights & Biases) for fresh training reruns
- Evaluate on a held-out geographic region to assess spatial generalization
- Explore knowledge distillation from the CNN-ViT hybrid into the CNN baseline
- Add containerized training environments for reproducible cross-framework reruns
- Add checkpoint download helpers so preserved evaluation weights can be retrieved

---

## Allowed Wording

- "remote-sensing classification"
- "agricultural vs non-agricultural land classification"
- "comparative CNN and CNN-ViT workflow"
- "reproducible research repository"
- "preserved historical evaluation artifacts"
- "binary satellite image classification"

## Forbidden Wording

- superlative benchmark claims implying best-in-class or leading performance across all methods
- claims that the system is operational, deployed, or running in a production environment
- claims of validation for real-world field or policy deployment
- claims that this constitutes a geospatial production platform
- any claim about field use, policy use, or live deployment outside a research context
