import pytest

from satellite_land_classification.predict import predict_image
from satellite_land_classification.utils import AssetNotFoundError


def test_predict_image_requires_assets(tmp_path):
    with pytest.raises(AssetNotFoundError):
        predict_image("keras_cnn", tmp_path / "missing.keras", tmp_path / "missing.jpg")


def test_predict_image_rejects_unknown_model_type(tmp_path):
    model_path = tmp_path / "model.bin"
    image_path = tmp_path / "image.jpg"
    model_path.write_bytes(b"")
    image_path.write_bytes(b"")
    with pytest.raises(ValueError):
        predict_image("unknown", model_path, image_path)

