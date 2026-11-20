import pickle
import struct
import zlib
from pathlib import Path

import lmdb
import numpy as np
import torch

from flood_jepa.data.supervised import WorldFloodsSegmentationDataset
from flood_jepa.data.worldfloods import WORLDFLOODS_BANDS, WorldFloodsPretrainDataset


def _write_fixture(path: Path, split: str) -> None:
    image = np.arange(10 * 32 * 32, dtype=np.uint16).reshape(10, 32, 32) + 1
    label = np.zeros((32, 32), dtype=np.uint8)
    label[:, 16:] = 1
    label[:4, :] = 255
    environment = lmdb.open(str(path), subdir=False, map_size=16 * 1024 * 1024)
    with environment.begin(write=True) as transaction:
        payload = (
            zlib.compress(image.tobytes()),
            image.shape,
            zlib.compress(label.tobytes()),
            label.shape,
            f"{split}-sample",
        )
        transaction.put(b"0", pickle.dumps(payload))
        transaction.put(b"length", struct.pack("<q", 1))
        metadata = {
            b"__len__": 1,
            b"split": split,
            b"storage_img_dtype": "uint16",
            b"storage_lbl_dtype": "uint8",
            b"compression": "zlib",
            b"bands": WORLDFLOODS_BANDS,
            b"normalization_mean": [0.0] * 10,
            b"normalization_std": [1.0] * 10,
            b"class_weights": [1.0, 2.0],
            b"sample_ids": [f"{split}-sample"],
            b"source_ids": [f"{split}-source"],
            b"event_ids": [f"{split}-event"],
            b"tile_size": 32,
        }
        for key, value in metadata.items():
            transaction.put(key, pickle.dumps(value))
    environment.close()


def test_pretraining_reader_is_label_free_and_selects_all_bands(tmp_path: Path):
    path = tmp_path / "train.lmdb"
    _write_fixture(path, "train")
    dataset = WorldFloodsPretrainDataset(
        path,
        crop_size=16,
        input_bands=WORLDFLOODS_BANDS,
        augment_d4=False,
    )
    sample = dataset[0]
    assert set(sample) == {"image", "sample_id"}
    assert sample["image"].shape == (10, 16, 16)


def test_pretraining_reader_retries_fully_padded_tiles(monkeypatch, tmp_path: Path):
    path = tmp_path / "train.lmdb"
    _write_fixture(path, "train")
    dataset = WorldFloodsPretrainDataset(
        path,
        crop_size=32,
        input_bands=WORLDFLOODS_BANDS,
        augment_d4=False,
        sample_retry_attempts=2,
    )
    dataset.length = 2
    dataset.sample_ids = ["padded", "valid"]
    padded = np.zeros((10, 32, 32), dtype=np.uint16)
    valid = np.ones((10, 32, 32), dtype=np.uint16)
    monkeypatch.setattr(
        dataset,
        "_decode_image",
        lambda index: (padded, "padded") if index == 0 else (valid, "valid"),
    )
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([1]))

    sample = dataset[0]
    assert sample["sample_id"] == "valid"
    assert sample["image"].shape == (10, 32, 32)


def test_crop_falls_back_to_valid_boundary_region(monkeypatch, tmp_path: Path):
    path = tmp_path / "train.lmdb"
    _write_fixture(path, "train")
    dataset = WorldFloodsPretrainDataset(path, crop_size=16, crop_attempts=2)
    image = torch.zeros(10, 32, 32)
    image[:, 24:, 24:] = 1
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([0]))

    crop = dataset._random_crop(image)
    assert crop.shape == (10, 16, 16)
    assert torch.any(crop != 0)


def test_supervised_reader_returns_one_complete_eval_tile(tmp_path: Path):
    path = tmp_path / "val.lmdb"
    _write_fixture(path, "val")
    dataset = WorldFloodsSegmentationDataset(
        path,
        split="val",
        crop_size=16,
        input_bands=WORLDFLOODS_BANDS,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["image"].shape == (10, 32, 32)
    assert sample["label"].shape == (32, 32)
    assert sample["source_id"] == "val-source"


def test_supervised_augmentation_casts_uint16_before_d4(tmp_path: Path):
    path = tmp_path / "train.lmdb"
    _write_fixture(path, "train")
    dataset = WorldFloodsSegmentationDataset(
        path,
        split="train",
        crop_size=16,
        input_bands=WORLDFLOODS_BANDS,
        augment_d4=True,
    )
    sample = dataset[0]
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (10, 16, 16)
    assert sample["label"].shape == (16, 16)


def test_supervised_benchmark_rotation_preserves_label_vocabulary(tmp_path: Path):
    path = tmp_path / "train.lmdb"
    _write_fixture(path, "train")
    dataset = WorldFloodsSegmentationDataset(
        path,
        split="train",
        crop_size=16,
        input_bands=WORLDFLOODS_BANDS,
        augment_d4=True,
        d4_probability=0.5,
        rotate_probability=1.0,
        rotate_limit_degrees=90.0,
    )
    sample = dataset[0]

    assert sample["image"].shape == (10, 16, 16)
    assert set(torch.unique(sample["label"]).tolist()).issubset({0, 1, 255})
