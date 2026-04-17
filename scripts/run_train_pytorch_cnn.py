from __future__ import annotations

import argparse
import sys

from satellite_land_classification.config import load_config, merge_configs
from satellite_land_classification.pytorch_cnn import train_pytorch_cnn
from satellite_land_classification.utils import AssetNotFoundError, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Module 2 PyTorch CNN.")
    parser.add_argument("--config", required=True, help="Path to configs/pytorch_cnn.yaml")
    parser.add_argument("--data-config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    config = merge_configs(load_config(args.data_config), load_config(args.config))
    dataset = config["dataset"]
    training = config["training"]
    model = config["model"]
    try:
        history = train_pytorch_cnn(
            dataset_dir=dataset["extracted_dir"],
            output_model_path=model["output_path"],
            image_size=dataset["image_size"][0],
            batch_size=training["batch_size"],
            learning_rate=training["learning_rate"],
            epochs=training["epochs"],
            validation_split=dataset["validation_split"],
            seed=dataset["seed"],
            num_workers=training["num_workers"],
        )
    except AssetNotFoundError as exc:
        print(str(exc))
        return 1
    write_json(history, model["history_path"])
    print(f"Best model saved to: {model['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

