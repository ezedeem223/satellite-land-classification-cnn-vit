import pytest

from satellite_land_classification.data import build_image_index, validate_dataset_dir
from satellite_land_classification.utils import AssetNotFoundError


def test_validate_dataset_dir_requires_class_folders(tmp_path):
    dataset_dir = tmp_path / "images_dataSAT"
    dataset_dir.mkdir()
    (dataset_dir / "class_0_non_agri").mkdir()
    with pytest.raises(AssetNotFoundError):
        validate_dataset_dir(dataset_dir)


def test_build_image_index_collects_both_classes(tmp_path):
    dataset_dir = tmp_path / "images_dataSAT"
    class_zero = dataset_dir / "class_0_non_agri"
    class_one = dataset_dir / "class_1_agri"
    class_zero.mkdir(parents=True)
    class_one.mkdir(parents=True)
    (class_zero / "a.jpg").write_bytes(b"")
    (class_one / "b.jpg").write_bytes(b"")
    index = build_image_index(dataset_dir)
    assert len(index) == 2
    assert {label for _, label in index} == {0, 1}

