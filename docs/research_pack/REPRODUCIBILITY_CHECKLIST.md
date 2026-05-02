# Reproducibility Checklist

This checklist covers the full experimental workflow from installation to evaluation.
Each item is marked with whether it requires the dataset, model checkpoints, or
neither, so reviewers can understand exactly what can and cannot be verified in
a fresh clone.

---

## 1. Installation

### 1a. Install runtime dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

**Requires dataset:** No
**Requires checkpoints:** No
**Expected outcome:** All packages install without error. The
`satellite_land_classification` package becomes importable.

### 1b. Verify package import

```bash
python -c "from satellite_land_classification import load_config; print('OK')"
```

**Requires dataset:** No
**Requires checkpoints:** No
**Expected outcome:** Prints `OK`.

### 1c. Install development dependencies (optional, for contributors)

```bash
python -m pip install -r requirements-dev.txt
```

**Requires dataset:** No
**Requires checkpoints:** No
**Expected outcome:** `pytest`, `ruff`, `black`, `isort`, and `pre-commit` install.

---

## 2. Data Preparation

### 2a. Download and extract the dataset

```bash
python scripts/run_prepare_data.py --config configs/data.yaml
```

**Requires dataset:** Downloads it (requires internet access)
**Requires checkpoints:** No
**Config file:** `configs/data.yaml`
**Expected outcome:** `data/images_dataSAT/class_0_non_agri/` and
`data/images_dataSAT/class_1_agri/` directories exist and contain `.jpg` images.

### 2b. Alternative: manual placement

Place the extracted archive at `data/images_dataSAT/` with the two class subdirectories.

**Requires dataset:** Yes (manually provided)
**Requires checkpoints:** No

---

## 3. Configuration Files

| Config file | Purpose |
|---|---|
| `configs/data.yaml` | Dataset path, class names, image size, split ratio, seed |
| `configs/keras_cnn.yaml` | Keras CNN model and training hyperparameters |
| `configs/pytorch_cnn.yaml` | PyTorch CNN model and training hyperparameters |
| `configs/keras_vit.yaml` | Keras CNN-ViT hybrid model and training hyperparameters |
| `configs/pytorch_vit.yaml` | PyTorch CNN-ViT hybrid model and training hyperparameters |
| `configs/integration.yaml` | Multi-model evaluation settings |

All config files are YAML and can be inspected or modified without running any
training. They do not require the dataset or checkpoints to be viewed.

---

## 4. Training Commands

All training commands require the dataset to be prepared locally first.

### 4a. Keras CNN baseline

```bash
python scripts/run_train_keras_cnn.py \
    --config configs/keras_cnn.yaml \
    --data-config configs/data.yaml
```

**Requires dataset:** Yes
**Requires checkpoints:** No (trains from scratch)
**Output:** `models/keras_cnn_best.keras` (or path configured in `keras_cnn.yaml`)

### 4b. PyTorch CNN baseline

```bash
python scripts/run_train_pytorch_cnn.py \
    --config configs/pytorch_cnn.yaml \
    --data-config configs/data.yaml
```

**Requires dataset:** Yes
**Requires checkpoints:** No (trains from scratch)
**Output:** `models/pytorch_cnn_best.pth`

### 4c. Keras CNN-ViT hybrid

```bash
python scripts/run_train_keras_vit.py \
    --config configs/keras_vit.yaml \
    --data-config configs/data.yaml
```

**Requires dataset:** Yes
**Requires checkpoints:** No (trains from scratch)
**Output:** `models/keras_cnn_vit_best.keras`

### 4d. PyTorch CNN-ViT hybrid

```bash
python scripts/run_train_pytorch_vit.py \
    --config configs/pytorch_vit.yaml \
    --data-config configs/data.yaml
```

**Requires dataset:** Yes
**Requires checkpoints:** No (trains from scratch)
**Output:** `models/pytorch_cnn_vit_best.pth`

---

## 5. Evaluation Commands

Evaluation requires both the dataset and model checkpoints.

```bash
python scripts/run_evaluate.py \
    --config configs/integration.yaml \
    --data-config configs/data.yaml
```

**Requires dataset:** Yes
**Requires checkpoints:** Yes (all model files referenced in `integration.yaml`)
**Expected output:** Accuracy, precision, recall, F1, ROC-AUC printed to stdout
and optionally written to `results/`.

---

## 6. Prediction Command

Single-image prediction requires a checkpoint but not the full dataset.

```bash
python scripts/run_predict.py \
    --model-type pytorch_vit \
    --model-path models/pytorch_cnn_vit_best.pth \
    --image-path path/to/your/image.jpg
```

**Requires dataset:** No (only a single image)
**Requires checkpoints:** Yes (the specified model file)
**Expected output:** Predicted class label and probability.

---

## 7. Tests

```bash
pytest tests/ -v
```

**Requires dataset:** No
**Requires checkpoints:** No
**Expected outcome:** All tests pass. Currently 7 core tests plus research pack
tests. No test requires the dataset or model checkpoints.

### Research pack tests

```bash
pytest tests/test_research_pack_exists.py tests/test_metric_provenance.py -v
```

**Requires dataset:** No
**Requires checkpoints:** No

---

## 8. Validation Tool

```bash
python tools/evidence/validate_research_pack.py
```

**Requires dataset:** No
**Requires checkpoints:** No
**Expected outcome:** All checks pass. Reports presence of research pack files,
results artifacts, and absence of forbidden overclaim phrases.

---

## 9. Linting

```bash
python -m ruff check .
```

**Requires dataset:** No
**Requires checkpoints:** No
**Expected outcome:** No ruff violations.

---

## 10. Required Local Assets Summary

| Asset | Required for | How to obtain |
|---|---|---|
| `data/images_dataSAT/` | Training, evaluation | `python scripts/run_prepare_data.py --config configs/data.yaml` |
| `models/keras_cnn_best.keras` | Keras CNN evaluation/prediction | Train locally or obtain compatible external checkpoint |
| `models/pytorch_cnn_best.pth` | PyTorch CNN evaluation/prediction | Train locally or obtain compatible external checkpoint |
| `models/keras_cnn_vit_best.keras` | Keras CNN-ViT evaluation/prediction | Train locally or obtain compatible external checkpoint |
| `models/pytorch_cnn_vit_best.pth` | PyTorch CNN-ViT evaluation/prediction | Train locally or obtain compatible external checkpoint |

---

## 11. Expected Outputs

| Output | Generated by | Location |
|---|---|---|
| Model checkpoint | Training scripts | `models/` |
| Evaluation metrics | `run_evaluate.py` | stdout + optionally `results/` |
| Prediction label | `run_predict.py` | stdout |
| Confusion matrix | Evaluation/visualization | `results/confusion_matrix.png` |
| ROC curve | Evaluation/visualization | `results/roc_curve.png` |

---

## 12. What Can Be Verified Without Dataset or Checkpoints

- Package import: `from satellite_land_classification import load_config`
- Config loading: `load_config("configs/data.yaml")`
- Data utilities: `validate_dataset_dir`, `build_image_index` (via tests with
  synthetic tmp_path fixtures)
- Metric computation: `compute_binary_metrics` (via tests with synthetic inputs)
- Prediction error handling: `predict_image` raises `AssetNotFoundError` correctly
- Research pack file existence
- Validation tool passes
- Ruff lint check passes

---

## 13. What Cannot Be Verified Without Dataset or Checkpoints

- Training script execution and convergence
- Evaluation script output metrics
- Freshly generated confusion matrix and ROC curve
- Single-image prediction on real satellite imagery
- Fresh replication of the preserved comparison metrics
- Explainability map generation (also requires checkpoints)

---

## 14. Preserved vs. Fresh Results

The preserved metrics in `results/` are **historical evaluation artifacts** from
experiment notebooks. They should not be interpreted as results generated by running
the current scripts in the current environment. A fresh end-to-end rerun may produce
different numerical values depending on:

- The checkpoint used (locally trained vs. the externally referenced evaluation weights)
- The number of training epochs
- Hardware-specific numerical precision differences
- Framework and library version differences
