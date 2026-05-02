# Satellite Land Classification CNN-ViT

## Project Overview

A Python-based machine learning research project for binary satellite image classification (agricultural vs. non-agricultural land). Implements both Keras and PyTorch versions of CNN baselines and CNN-ViT hybrid models.

## Architecture

- **Language**: Python 3.11
- **Frameworks**: TensorFlow/Keras, PyTorch
- **Type**: Pure Python ML research library (no frontend/web server)

## Project Structure

```
src/satellite_land_classification/   # Main package
  config.py          # YAML config loading and merging
  data.py            # Dataset loading, image indexing, validation
  augmentation.py    # Image augmentation pipelines
  features.py        # Feature extraction utilities
  keras_cnn.py       # Keras CNN model definition
  keras_vit.py       # Keras CNN-ViT hybrid model
  pytorch_cnn.py     # PyTorch CNN model definition
  pytorch_vit.py     # PyTorch CNN-ViT hybrid model
  evaluate.py        # Metrics computation (accuracy, F1, ROC-AUC, etc.)
  predict.py         # Single image inference
  integration.py     # Multi-model integration/comparison
  visualization.py   # Confusion matrix, ROC curve plotting
  utils.py           # Shared utilities and error types
scripts/             # CLI entry points for training, evaluation, prediction
configs/             # YAML configuration files for each model/experiment
tests/               # pytest test suite (7 tests)
models/              # Model checkpoints (not tracked in git)
data/                # Dataset directory (not tracked in git)
results/             # Evaluation artifacts, metrics, plots
notebooks/           # Cleaned experiment notebooks
source_notebooks/    # Original preserved experiment notebooks
```

## Setup

Install core dependencies (lightweight):
```bash
pip install pyyaml numpy pillow tqdm scikit-learn matplotlib
pip install -e .
```

Install ML frameworks (heavy, may take time):
```bash
pip install -r requirements.txt
```

Install dev tools:
```bash
pip install pytest black isort ruff
```

## Workflow

The configured workflow runs the test suite:
```
pytest tests/ -v
```

## Key Commands

```bash
# Run tests
pytest tests/ -v

# Prepare dataset (downloads from IBM Cloud)
python scripts/run_prepare_data.py --config configs/data.yaml

# Train models (requires dataset)
python scripts/run_train_pytorch_cnn.py --config configs/pytorch_cnn.yaml --data-config configs/data.yaml
python scripts/run_train_keras_cnn.py --config configs/keras_cnn.yaml --data-config configs/data.yaml
python scripts/run_train_pytorch_vit.py --config configs/pytorch_vit.yaml --data-config configs/data.yaml
python scripts/run_train_keras_vit.py --config configs/keras_vit.yaml --data-config configs/data.yaml

# Evaluate
python scripts/run_evaluate.py --config configs/integration.yaml --data-config configs/data.yaml

# Predict single image
python scripts/run_predict.py --model-type pytorch_vit --model-path models/pytorch_cnn_vit_best.pth --image-path path/to/image.jpg
```

## Environment Variables

See `.env.example` for reference:
- `DATA_DIR` — path to dataset (default: `data/images_dataSAT`)
- `MODELS_DIR` — path to model checkpoints (default: `models`)
- `RESULTS_DIR` — path to evaluation artifacts (default: `results`)
- `DATASET_URL` — download URL for the satellite image archive

## Dataset

The dataset (`images-dataSAT.tar`) is not bundled. After downloading/extracting:
```
data/images_dataSAT/
  class_0_non_agri/
  class_1_agri/
```

## Historical Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Keras CNN | 0.9925 | 0.9924 | 1.0000 |
| PyTorch CNN | 0.9988 | 0.9988 | 1.0000 |
| Keras CNN-ViT | 0.9958 | 0.9958 | 0.9998 |
| PyTorch CNN-ViT | 0.9990 | 0.9990 | 1.0000 |
