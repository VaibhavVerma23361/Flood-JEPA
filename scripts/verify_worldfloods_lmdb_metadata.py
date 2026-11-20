#!/usr/bin/env python3
"""Verify WorldFloods LMDB metadata before Flood-JEPA training.

This script is intentionally read-only. It checks that the packed LMDBs expose
the band order and WorldFloods normalization constants expected by the training
code, then optionally scans records with tqdm to catch corrupt payloads.
"""

from __future__ import annotations

import argparse
import pickle
import struct
import zlib
from pathlib import Path

import lmdb
import numpy as np
from tqdm.auto import tqdm


EXPECTED_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
EXPECTED_MEAN = np.asarray(
    [
        3758.07467509,
        3238.08247208,
        3418.90147615,
        3450.23315812,
        4030.94700446,
        4164.17468251,
        3981.96268494,
        4226.74862547,
        2391.66101119,
        1790.32497137,
    ],
    dtype=np.float64,
)
EXPECTED_STD = np.asarray(
    [
        2794.09579088,
        2549.49406140,
        2811.78109878,
        2776.93269704,
        2632.13814197,
        2657.43035126,
        2500.47885249,
        2589.29159887,
        1500.02533014,
        1241.98176280,
    ],
    dtype=np.float64,
)
EXPECTED_COUNTS = {"train": 21364, "val": 751, "test": 841}
EXPECTED_TILE_SIZE = 512


def _metadata(transaction, key: bytes, default=None):
    value = transaction.get(key)
    return default if value is None else pickle.loads(value)


def _length(transaction) -> int:
    packed = transaction.get(b"length")
    if packed is not None:
        return int(struct.unpack("<q", packed)[0])
    value = _metadata(transaction, b"__len__")
    if value is None:
        raise RuntimeError("LMDB has no length metadata")
    return int(value)


def _open_lmdb(path: Path):
    return lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
    )


def _assert_close(name: str, actual: np.ndarray, expected: np.ndarray, tolerance: float):
    if actual.shape != expected.shape:
        raise RuntimeError(f"{name} shape mismatch: expected {expected.shape}, got {actual.shape}")
    if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
        delta = np.abs(actual - expected)
        index = int(delta.argmax())
        raise RuntimeError(
            f"{name} mismatch at {EXPECTED_BANDS[index]}: "
            f"expected {expected[index]}, got {actual[index]}"
        )


def verify_split(lmdb_dir: Path, split: str, scan: bool, tolerance: float) -> dict:
    path = lmdb_dir / f"worldfloods_{split}.lmdb"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} LMDB: {path}")

    with _open_lmdb(path) as environment:
        with environment.begin(write=False) as transaction:
            length = _length(transaction)
            stored_split = str(_metadata(transaction, b"split", "unknown"))
            bands = list(_metadata(transaction, b"bands", []))
            mean = np.asarray(_metadata(transaction, b"normalization_mean", []), dtype=np.float64)
            std = np.asarray(_metadata(transaction, b"normalization_std", []), dtype=np.float64)
            tile_size = int(_metadata(transaction, b"tile_size", EXPECTED_TILE_SIZE))
            image_dtype = str(_metadata(transaction, b"storage_img_dtype", "unknown"))
            label_dtype = str(_metadata(transaction, b"storage_lbl_dtype", "unknown"))
            compression = str(_metadata(transaction, b"compression", "none"))

            if stored_split != split:
                raise RuntimeError(f"{path.name}: expected split={split}, got {stored_split}")
            if length != EXPECTED_COUNTS[split]:
                raise RuntimeError(
                    f"{path.name}: expected {EXPECTED_COUNTS[split]} samples, got {length}"
                )
            if bands != EXPECTED_BANDS:
                raise RuntimeError(f"{path.name}: expected bands {EXPECTED_BANDS}, got {bands}")
            _assert_close(f"{path.name} normalization_mean", mean, EXPECTED_MEAN, tolerance)
            _assert_close(f"{path.name} normalization_std", std, EXPECTED_STD, tolerance)
            if tile_size != EXPECTED_TILE_SIZE:
                raise RuntimeError(f"{path.name}: expected tile_size=512, got {tile_size}")
            if image_dtype != "uint16":
                raise RuntimeError(f"{path.name}: expected image dtype uint16, got {image_dtype}")
            if label_dtype != "uint8":
                raise RuntimeError(f"{path.name}: expected label dtype uint8, got {label_dtype}")

            zero_tiles = 0
            bad_payloads = 0
            if scan:
                for index in tqdm(range(length), desc=f"scan {split}", unit="tile"):
                    payload = transaction.get(str(index).encode("ascii"))
                    if payload is None:
                        bad_payloads += 1
                        continue
                    image_bytes, image_shape, label_bytes, label_shape, sample_id = pickle.loads(payload)
                    if compression == "zlib":
                        image_bytes = zlib.decompress(image_bytes)
                        label_bytes = zlib.decompress(label_bytes)
                    elif compression != "none":
                        raise RuntimeError(f"{path.name}: unsupported compression {compression}")
                    image = np.frombuffer(image_bytes, dtype=np.uint16).reshape(image_shape)
                    label = np.frombuffer(label_bytes, dtype=np.uint8).reshape(label_shape)
                    if image.shape != (10, EXPECTED_TILE_SIZE, EXPECTED_TILE_SIZE):
                        raise RuntimeError(f"{sample_id}: image shape mismatch {image.shape}")
                    if label.shape != (EXPECTED_TILE_SIZE, EXPECTED_TILE_SIZE):
                        raise RuntimeError(f"{sample_id}: label shape mismatch {label.shape}")
                    if not np.any(image):
                        zero_tiles += 1

    return {
        "split": split,
        "path": str(path),
        "samples": length,
        "bands": bands,
        "tile_size": tile_size,
        "image_dtype": image_dtype,
        "label_dtype": label_dtype,
        "compression": compression,
        "zero_tiles": zero_tiles if scan else None,
        "bad_payloads": bad_payloads if scan else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmdb-dir",
        type=Path,
        required=True,
        help="Directory containing worldfloods_train/val/test.lmdb",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan every record with tqdm after checking metadata",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-5,
        help="Absolute tolerance for mean/std metadata comparisons",
    )
    args = parser.parse_args()

    lmdb_dir = args.lmdb_dir.expanduser().resolve()
    summaries = [
        verify_split(lmdb_dir, split, scan=args.scan, tolerance=args.tolerance)
        for split in ("train", "val", "test")
    ]
    print("WorldFloods LMDB metadata verification passed.")
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
