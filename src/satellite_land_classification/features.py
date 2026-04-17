from __future__ import annotations

DEFAULT_KERAS_CNN_FEATURE_LAYER = "batch_normalization_5"


def resolve_feature_layer_name(candidate: str | None = None) -> str:
    """Return the feature layer name used by the Keras CNN-ViT notebook."""

    return candidate or DEFAULT_KERAS_CNN_FEATURE_LAYER

