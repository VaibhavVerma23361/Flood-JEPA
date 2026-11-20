import copy
from pathlib import Path

import pytest

from flood_jepa.config import apply_overrides, load_config
from flood_jepa.training.segmentation import (
    _validate_dataset_splits,
    _validate_checkpoint_compatibility,
    _validate_pretrain_identity,
)


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
    assert config["loss"]["sigreg"]["enabled"] is False


def test_sigreg_pretrain_config_is_a_separate_experiment(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORLDFLOODS_LMDB_DIR", str(tmp_path))
    config = load_config("configs/pretrain_vitb16_256_10band_sigreg.yaml")

    assert config["loss"]["sigreg"]["enabled"] is True
    assert config["loss"]["sigreg"]["weight"] == 0.05
    assert config["loss"]["sigreg"]["projector_output_dim"] == 128
    assert config["optimization"]["per_device_batch_size"] == 8
    assert "sigreg" in config["experiment"]["output_dir"]


def test_four_band_config_remains_available_for_future_ufo_transfer(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORLDFLOODS_LMDB_DIR", str(tmp_path))
    config = load_config("configs/pretrain_vitb16_256_4band.yaml")
    assert config["data"]["input_bands"] == ["B2", "B3", "B4", "B8"]
    assert config["model"]["input_channels"] == 4


def test_a30_vit_tiny_configs_are_size_locked(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORLDFLOODS_LMDB_DIR", str(tmp_path))
    monkeypatch.setenv("FLOOD_JEPA_PRETRAINED", str(tmp_path / "epoch_100.pt"))

    pretrain = load_config("configs/pretrain_vitt16_256_10band_a30.yaml")
    assert pretrain["model"]["model_name"] == "vit_tiny"
    assert pretrain["model"]["encoder_dim"] == 192
    assert pretrain["model"]["predictor_depth"] == 4
    assert pretrain["model"]["predictor_dim"] == 192
    assert pretrain["model"]["predictor_heads"] == 6
    assert pretrain["optimization"]["dtype"] == "bfloat16"

    finetune = load_config("configs/finetune_worldfloods_vitt10_a30.yaml")
    assert finetune["model"]["expected_pretrain_model_name"] == "vit_tiny"
    assert finetune["model"]["decoder_type"] == "seg_upernet_lite"
    assert finetune["model"]["decoder_dim"] == 128
    assert finetune["optimization"]["dtype"] == "bfloat16"


def test_ablation_configs_change_only_the_supervision_ingredients():
    masked = load_config("configs/pretrain_vitt16_256_10band_h100_A.yaml")
    visible = load_config("configs/pretrain_vitt16_256_10band_h100_B.yaml")
    deep = load_config("configs/pretrain_vitt16_256_10band_h100_C.yaml")
    full = load_config("configs/pretrain_vitt16_256_10band_h100.yaml")
    assert masked["model"]["deep_self_supervision_levels"] == 1
    assert masked["loss"]["predict_all"] is False
    assert masked["loss"]["context_weight"] == 0.0
    assert visible["model"]["deep_self_supervision_levels"] == 1
    assert visible["loss"]["predict_all"] is True
    assert deep["model"]["deep_self_supervision_levels"] == 4
    assert deep["loss"]["predict_all"] is False
    assert full["loss"]["predict_all"] is True
    assert full["model"]["deep_self_supervision_levels"] == 4
    for config in (masked, visible, deep, full):
        assert config["model"]["model_name"] == "vit_tiny"
        assert config["optimization"]["per_device_batch_size"] == 32
        assert config["optimization"]["epochs"] == 100


def test_h100_vit_tiny_configs_match_the_reported_worldfloods_recipe(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("WORLDFLOODS_LMDB_DIR", str(tmp_path))
    monkeypatch.setenv("FLOOD_JEPA_PRETRAINED", str(tmp_path / "epoch_100.pt"))

    pretrain = load_config("configs/pretrain_vitt16_256_10band_h100.yaml")
    assert pretrain["model"]["model_name"] == "vit_tiny"
    assert pretrain["model"]["encoder_dim"] == 192
    assert pretrain["model"]["predictor_depth"] == 4
    assert pretrain["model"]["predictor_dim"] == 192
    assert pretrain["model"]["predictor_heads"] == 6
    assert pretrain["model"]["modality_embedding"] is False
    assert pretrain["optimization"]["epochs"] == 100
    assert pretrain["optimization"]["per_device_batch_size"] == 32
    assert pretrain["optimization"]["learning_rate"] == 0.0002
    assert pretrain["optimization"]["ema_start"] == 0.99925
    assert pretrain["optimization"]["dtype"] == "bfloat16"
    assert pretrain["loss"]["predict_all"] is True
    assert pretrain["mask"][0]["num_blocks"] == 10
    assert pretrain["mask"][0]["spatial_scale"] == [0.15, 0.15]

    finetune = load_config("configs/finetune_worldfloods_vitt10_h100.yaml")
    assert finetune["model"]["expected_pretrain_model_name"] == "vit_tiny"
    assert finetune["model"]["encoder_source"] == "target"
    assert finetune["model"]["freeze_encoder"] is True
    assert finetune["model"]["decoder_type"] == "seg_upernet"
    assert finetune["model"]["decoder_dim"] == 512
    assert finetune["model"]["feature_layers"] == [2, 5, 8, 11]
    assert finetune["distributed"]["required_world_size"] == 1
    assert finetune["distributed"]["global_batch_size"] == 14
    assert finetune["optimization"]["per_device_batch_size"] == 14
    assert finetune["optimization"]["epochs"] == 30
    assert finetune["optimization"]["decoder_learning_rate"] == 2.0e-5
    assert finetune["optimization"]["dtype"] == "bfloat16"
    assert finetune["loss"]["ce_weight"] == 1.0
    assert finetune["loss"]["dice_weight"] == 1.0
    assert finetune["loss"]["focal_weight"] == 0.5
    assert finetune["evaluation"]["allow_repeat_test"] is False
    assert finetune["data"]["expected_tile_counts"] == {
        "train": 21364,
        "val": 751,
        "test": 841,
    }


def test_v100_vit_small_pretraining_config_is_size_locked(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORLDFLOODS_LMDB_DIR", str(tmp_path))
    config = load_config("configs/pretrain_vits16_256_10band_v100.yaml")

    assert config["experiment"]["name"] == "vjepa21_vits16_worldfloods_10band_256_v100"
    assert config["data"]["input_bands"] == [
        "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"
    ]
    assert config["model"]["model_name"] == "vit_small"
    assert config["model"]["input_channels"] == 10
    assert config["model"]["encoder_dim"] == 384
    assert config["model"]["predictor_depth"] == 12
    assert config["model"]["predictor_dim"] == 256
    assert config["model"]["predictor_heads"] == 8
    assert config["model"]["modality_embedding"] is False
    assert config["model"]["use_activation_checkpointing"] is True
    assert config["optimization"]["per_device_batch_size"] == 8
    assert config["optimization"]["dtype"] == "float16"


def test_typed_configuration_overrides(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORLDFLOODS_LMDB_DIR", str(tmp_path))
    config = load_config("configs/pretrain_vitb16_256_10band.yaml")
    apply_overrides(
        config,
        [
            "optimization.epochs=3",
            "optimization.learning_rate=1.0e-4",
            "model.modality_embedding=false",
            "data.input_bands=[B2, B3, B4]",
            "mask.0.num_blocks=7",
        ],
    )
    assert config["optimization"]["epochs"] == 3
    assert config["optimization"]["learning_rate"] == 1.0e-4
    assert config["model"]["modality_embedding"] is False
    assert config["data"]["input_bands"] == ["B2", "B3", "B4"]
    assert config["mask"][0]["num_blocks"] == 7

    with pytest.raises(KeyError):
        apply_overrides(config, ["optimization.not_a_setting=1"])


def test_benchmark_config_uses_frozen_upernet_for_thirty_epochs(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("WORLDFLOODS_LMDB_DIR", str(tmp_path))
    monkeypatch.setenv("FLOOD_JEPA_PRETRAINED", str(tmp_path / "epoch_100.pt"))
    config = load_config("configs/finetune_worldfloods_10band.yaml")

    assert config["model"]["decoder_type"] == "seg_upernet"
    assert config["model"]["decoder_dim"] == 512
    assert config["model"]["decoder_dropout"] == 0.1
    assert config["model"]["freeze_encoder"] is True
    assert config["model"]["expected_pretrain_model_name"] == "vit_base"
    assert config["model"]["expected_pretrain_input_channels"] == 10
    assert config["model"]["expected_pretrain_patch_size"] == 16
    assert config["optimization"]["epochs"] == 30
    assert config["optimization"]["encoder_learning_rate"] == 0.0
    assert config["optimization"]["decoder_learning_rate"] == 2.0e-5
    assert config["optimization"]["scheduler"] == "cosine"
    assert config["optimization"]["warmup_epochs"] == 3
    assert config["optimization"]["per_device_batch_size"] == 2
    assert config["optimization"]["dtype"] == "float16"
    assert config["optimization"]["validation_every_epochs"] == 1
    assert config["loss"]["use_class_weights"] is True
    assert config["loss"]["ce_weight"] == 1.0
    assert config["loss"]["dice_weight"] == 1.0
    assert config["loss"]["focal_weight"] == 0.5
    assert config["loss"]["focal_alpha"] == 0.25
    assert config["loss"]["focal_gamma"] == 2.0
    assert config["distributed"]["required_world_size"] == 7
    assert config["distributed"]["global_batch_size"] == 14
    assert config["data"]["d4_probability"] == 0.5
    assert config["data"]["rotate_probability"] == 0.2
    assert config["data"]["crop_attempts"] == 10
    assert config["data"]["minimum_supervised_fraction"] == 0.0
    assert config["data"]["expected_tile_counts"] == {
        "train": 21364,
        "val": 751,
        "test": 841,
    }
    assert config["evaluation"]["primary_metric"] == "global_miou"


def test_evaluation_rejects_changed_decoder_semantics(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORLDFLOODS_LMDB_DIR", str(tmp_path))
    monkeypatch.setenv("FLOOD_JEPA_PRETRAINED", str(tmp_path / "epoch_100.pt"))
    config = load_config("configs/finetune_worldfloods_10band.yaml")
    checkpoint = {"config": copy.deepcopy(config)}

    _validate_checkpoint_compatibility(config, checkpoint)
    changed = copy.deepcopy(config)
    changed["model"]["decoder_dim"] = 256
    with pytest.raises(ValueError, match="model.decoder_dim"):
        _validate_checkpoint_compatibility(changed, checkpoint)


def test_evaluation_rejects_wrong_pretraining_provenance():
    saved = {
        "data": {"crop_size": 256, "input_bands": ["B2", "B3"]},
        "model": {"model_name": "vit_base", "input_channels": 2, "patch_size": 16},
    }
    _validate_pretrain_identity(copy.deepcopy(saved), saved)
    changed = copy.deepcopy(saved)
    changed["model"]["model_name"] = "vit_large"
    with pytest.raises(ValueError, match="model.model_name"):
        _validate_pretrain_identity(changed, saved)


def test_benchmark_split_validation_checks_counts_and_identity():
    class DatasetStub:
        def __init__(self, sample_ids, source_ids):
            self.sample_ids = sample_ids
            self.source_ids = source_ids

        def __len__(self):
            return len(self.sample_ids)

    datasets = {
        "train": DatasetStub(
            ["train-a", "train-b"],
            ["train-scene", "train-scene"],
        ),
        "val": DatasetStub(["val-a"], ["val-scene"]),
        "test": DatasetStub(["test-a"], ["test-scene"]),
    }
    _validate_dataset_splits(datasets, {"train": 2, "val": 1, "test": 1})

    datasets["test"].source_ids = ["val-scene"]
    with pytest.raises(RuntimeError, match="source-scene IDs overlap"):
        _validate_dataset_splits(datasets, {"train": 2, "val": 1, "test": 1})
