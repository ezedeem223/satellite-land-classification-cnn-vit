from satellite_land_classification.config import load_config, merge_configs


def test_load_config_reads_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("section:\n  value: 3\n", encoding="utf-8")
    config = load_config(config_file)
    assert config["section"]["value"] == 3


def test_merge_configs_is_recursive():
    merged = merge_configs({"a": {"b": 1}}, {"a": {"c": 2}})
    assert merged == {"a": {"b": 1, "c": 2}}

