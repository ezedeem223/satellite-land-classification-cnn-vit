from __future__ import annotations

import argparse
import sys

from satellite_land_classification.config import load_config, merge_configs
from satellite_land_classification.keras_vit import train_keras_cnn_vit
from satellite_land_classification.utils import AssetNotFoundError, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Module 3 Keras CNN-ViT hybrid.")
    parser.add_argument("--config", required=True, help="Path to configs/keras_vit.yaml")
    parser.add_argument("--data-config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    config = merge_configs(load_config(args.data_config), load_config(args.config))
    dataset = config["dataset"]
    training = config["training"]
    model = config["model"]
    try:
        history = train_keras_cnn_vit(
            dataset_dir=dataset["extracted_dir"],
            pretrained_cnn_path=model["pretrained_cnn_path"],
            output_model_path=model["output_path"],
            feature_layer_name=model["feature_layer_name"],
            batch_size=training["batch_size"],
            learning_rate=training["learning_rate"],
            epochs=training["epochs"],
            steps_per_epoch=training["steps_per_epoch"],
            validation_split=dataset["validation_split"],
            num_transformer_layers=training["num_transformer_layers"],
            attention_heads=training["attention_heads"],
            mlp_dim=training["mlp_dim"],
        )
    except AssetNotFoundError as exc:
        print(str(exc))
        return 1
    write_json(history, model["history_path"])
    print(f"Best model saved to: {model['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

