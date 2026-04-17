from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from torchvision import datasets, transforms

from satellite_land_classification.config import load_config, merge_configs
from satellite_land_classification.evaluate import (
    compute_binary_metrics,
    save_confusion_matrix,
    save_roc_curve,
)
from satellite_land_classification.integration import persist_evaluation_bundle
from satellite_land_classification.keras_cnn import load_keras_model
from satellite_land_classification.keras_vit import load_hybrid_model
from satellite_land_classification.pytorch_cnn import load_pytorch_cnn
from satellite_land_classification.pytorch_vit import load_pytorch_hybrid
from satellite_land_classification.utils import AssetNotFoundError, require_path


def _evaluate_keras_binary(model_path: str | Path, dataset_dir: str | Path):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
    generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(64, 64),
        batch_size=128,
        class_mode="binary",
        shuffle=False,
    )
    model = load_keras_model(model_path)
    probs = model.predict(generator, verbose=0).reshape(-1)
    preds = (probs > 0.5).astype(int)
    return generator.classes, preds, probs


def _evaluate_keras_hybrid(model_path: str | Path, dataset_dir: str | Path):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
    generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(64, 64),
        batch_size=128,
        class_mode="binary",
        shuffle=False,
    )
    model = load_hybrid_model(model_path)
    probs = model.predict(generator, verbose=0)
    preds = np.argmax(probs, axis=1)
    return generator.classes, preds, probs[:, 1]


def _evaluate_torch_model(model, dataset_dir: str | Path):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    labels = []
    preds = []
    probs = []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels.extend(batch_labels.numpy().tolist())
            preds.extend(batch_preds.tolist())
            probs.extend(batch_probs.tolist())
    return labels, preds, probs


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate configured models on the dataset.")
    parser.add_argument("--config", required=True, help="Path to configs/integration.yaml")
    parser.add_argument("--data-config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    config = merge_configs(load_config(args.data_config), load_config(args.config))
    dataset_dir = require_path(config["dataset"]["extracted_dir"], "Dataset directory")
    label_names = config["dataset"]["label_names"]
    evaluation = config["evaluation"]
    metrics_by_model = {}
    for model_cfg in evaluation["models"]:
        model_path = Path(model_cfg["path"])
        if not model_path.exists():
            print(f"Skipping missing model: {model_path}")
            continue
        model_type = model_cfg["type"]
        if model_type == "keras_cnn":
            y_true, y_pred, y_prob = _evaluate_keras_binary(model_path, dataset_dir)
        elif model_type == "keras_vit":
            y_true, y_pred, y_prob = _evaluate_keras_hybrid(model_path, dataset_dir)
        elif model_type == "pytorch_cnn":
            model = load_pytorch_cnn(model_path, num_classes=model_cfg.get("num_classes", 2))
            y_true, y_pred, y_prob = _evaluate_torch_model(model, dataset_dir)
        elif model_type == "pytorch_vit":
            model = load_pytorch_hybrid(
                model_path,
                num_classes=model_cfg.get("num_classes", 2),
                depth=model_cfg.get("depth", 3),
                attention_heads=model_cfg.get("attention_heads", 6),
                embed_dim=model_cfg.get("embed_dim", 768),
            )
            y_true, y_pred, y_prob = _evaluate_torch_model(model, dataset_dir)
        else:
            raise ValueError(f"Unsupported model type in evaluation config: {model_type}")
        metrics = compute_binary_metrics(y_true, y_pred, y_prob, label_names)
        metrics_by_model[model_cfg["name"]] = metrics
        save_confusion_matrix(
            y_true,
            y_pred,
            label_names,
            Path(evaluation["confusion_matrix_dir"]) / f"{model_cfg['name']}.png",
            f"{model_cfg['name']} confusion matrix",
        )
        save_roc_curve(
            y_true,
            y_prob,
            Path(evaluation["roc_dir"]) / f"{model_cfg['name']}.png",
            f"{model_cfg['name']} ROC curve",
        )
    if not metrics_by_model:
        print("No configured models were available for evaluation.")
        return 1
    persisted = persist_evaluation_bundle(
        metrics_by_model=metrics_by_model,
        output_dir=evaluation["output_dir"],
        metrics_json=evaluation["metrics_json"],
        comparison_csv=evaluation["comparison_csv"],
        classification_report_txt=evaluation["classification_report_txt"],
    )
    for key, value in persisted.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssetNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

