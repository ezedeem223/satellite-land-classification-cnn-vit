# Satellite Research Evidence Pack

This directory is a curated evidence, documentation, and reproducibility pack for the
`satellite-land-classification-cnn-vit` research repository. It is part of the
KAUST AI research portfolio.

## Purpose

Strengthen the repository as a credible research artifact by providing structured
documentation, metric provenance, limitation disclosure, and a reproducibility
checklist — without adding new performance claims, retraining models, or fabricating
explainability outputs.

## Files

| File | Description |
|---|---|
| `PROJECT_BRIEF_KAUST.md` | Concise academic brief oriented toward KAUST research context |
| `MODEL_COMPARISON_BRIEF.md` | Side-by-side comparison of preserved metrics for all four model variants |
| `METRIC_PROVENANCE_MATRIX.md` | Traceability table linking every metric to its source artifact |
| `EXPERIMENT_LIMITATION_MATRIX.md` | Structured disclosure of experimental limitations and reviewer risks |
| `CNN_VS_CNN_VIT_EXPLAINER.md` | Accessible academic explainer on the CNN vs CNN-ViT hybrid design |
| `DATASET_AND_TASK_CARD.md` | Dataset and task card documenting scope, layout, and boundaries |
| `EXPLAINABILITY_PROTOCOL.md` | Protocol for future explainability work; no maps generated in this pass |
| `REPRODUCIBILITY_CHECKLIST.md` | Step-by-step checklist for reproducing the experimental workflow |

## Validation

Run the validation tool to check that all pack files exist, results artifacts are
present, and no forbidden overclaim phrases appear in any research pack document:

```bash
python tools/evidence/validate_research_pack.py
```

## Scope Boundaries

This evidence pack does **not**:

- Add new model performance numbers
- Generate attention maps, Grad-CAMs, heatmaps, or saliency images
- Claim geospatial or live deployment in a production environment
- Claim real-world agricultural monitoring validation
- Claim best-in-class or leading performance across all methods
- Download datasets or add new external datasets
- Retrain any model

All metrics cited come from preserved historical evaluation artifacts already present
in `results/`.
