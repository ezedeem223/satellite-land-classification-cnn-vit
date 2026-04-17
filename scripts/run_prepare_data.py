from __future__ import annotations

import argparse
import sys

from satellite_land_classification.config import load_config
from satellite_land_classification.data import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and extract the satellite dataset.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    config = load_config(args.config)["dataset"]
    dataset_dir = prepare_dataset(
        root_dir=config["root_dir"],
        dataset_url=config["url"],
        archive_name=config.get("archive_name", "images-dataSAT.tar"),
        extracted_dir=config["extracted_dir"],
    )
    print(f"Dataset ready at: {dataset_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

