from __future__ import annotations

import argparse
import json
import sys

from satellite_land_classification.predict import predict_image
from satellite_land_classification.utils import AssetNotFoundError


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict one satellite image.")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["keras_cnn", "pytorch_cnn", "keras_vit", "pytorch_vit"],
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=6)
    parser.add_argument("--embed-dim", type=int, default=768)
    args = parser.parse_args()
    prediction = predict_image(
        model_type=args.model_type,
        model_path=args.model_path,
        image_path=args.image_path,
        depth=args.depth,
        attention_heads=args.attention_heads,
        embed_dim=args.embed_dim,
    )
    print(json.dumps(prediction, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssetNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

