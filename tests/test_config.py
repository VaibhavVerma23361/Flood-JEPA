from pathlib import Path

from flood_jepa.config import load_config


def test_primary_config_uses_every_worldfloods_band_and_disables_modality_embedding(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("WORLDFLOODS_LMDB_DIR", str(tmp_path))
    config = load_config("configs/pretrain_vitb16_256_10band.yaml")
    assert config["data"]["crop_size"] == 256
    assert config["data"]["input_bands"] == [
        "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"
    ]
    assert config["model"]["input_channels"] == 10
    assert config["model"]["modality_embedding"] is False
    assert config["model"]["patch_size"] == 16


def test_four_band_config_remains_available_for_future_ufo_transfer(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORLDFLOODS_LMDB_DIR", str(tmp_path))
    config = load_config("configs/pretrain_vitb16_256_4band.yaml")
    assert config["data"]["input_bands"] == ["B2", "B3", "B4", "B8"]
    assert config["model"]["input_channels"] == 4
