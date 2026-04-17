from __future__ import annotations

from pathlib import Path

from .utils import require_path


def predict_image(
    model_type: str,
    model_path: str | Path,
    image_path: str | Path,
    **kwargs,
) -> dict[str, float]:
    """Dispatch prediction to the requested framework-specific implementation."""

    require_path(model_path, "Model artifact")
    require_path(image_path, "Input image")
    if model_type == "keras_cnn":
        from .keras_cnn import predict_with_keras_model

        return predict_with_keras_model(model_path, image_path)
    if model_type == "pytorch_cnn":
        from .pytorch_cnn import predict_with_pytorch_cnn

        return predict_with_pytorch_cnn(model_path, image_path)
    if model_type == "keras_vit":
        from .keras_vit import predict_with_keras_hybrid

        return predict_with_keras_hybrid(model_path, image_path)
    if model_type == "pytorch_vit":
        from .pytorch_vit import predict_with_pytorch_hybrid

        return predict_with_pytorch_hybrid(model_path, image_path, **kwargs)
    raise ValueError(
        "Unsupported model type. Expected one of: keras_cnn, pytorch_cnn, keras_vit, pytorch_vit."
    )
