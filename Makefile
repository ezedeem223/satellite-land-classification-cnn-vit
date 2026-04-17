PYTHON ?= python

.PHONY: setup install install-dev lint format test prepare-data train-keras-cnn train-pytorch-cnn train-keras-vit train-pytorch-vit evaluate predict

setup:
	$(PYTHON) -m pip install --upgrade pip

install: setup
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev: setup
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

lint:
	ruff check .

format:
	black .
	isort .

test:
	pytest

prepare-data:
	$(PYTHON) scripts/run_prepare_data.py --config configs/data.yaml

train-keras-cnn:
	$(PYTHON) scripts/run_train_keras_cnn.py --config configs/keras_cnn.yaml --data-config configs/data.yaml

train-pytorch-cnn:
	$(PYTHON) scripts/run_train_pytorch_cnn.py --config configs/pytorch_cnn.yaml --data-config configs/data.yaml

train-keras-vit:
	$(PYTHON) scripts/run_train_keras_vit.py --config configs/keras_vit.yaml --data-config configs/data.yaml

train-pytorch-vit:
	$(PYTHON) scripts/run_train_pytorch_vit.py --config configs/pytorch_vit.yaml --data-config configs/data.yaml

evaluate:
	$(PYTHON) scripts/run_evaluate.py --config configs/integration.yaml --data-config configs/data.yaml

predict:
	$(PYTHON) scripts/run_predict.py --help

