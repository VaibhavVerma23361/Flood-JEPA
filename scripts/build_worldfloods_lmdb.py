#!/usr/bin/env python3
"""Pack official WorldFloods V2 GeoTIFFs into the benchmark LMDB contract.

Scenes are read in sorted filename order. Each scene is cut into
non-overlapping 512 x 512 windows, including a padded border window.
Image padding is 0 and label padding is 255. A window is kept only when
it contains at least one clear land or clear water pixel.

The ten stored bands are the ml4floods ``sub_20`` Sentinel-2 subset:
B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12. Normalization constants are
the published training-split values and are not refit here.
"""

from __future__ import annotations

import argparse
import pickle
import re
import struct
import zlib
from pathlib import Path

import lmdb
import numpy as np
import tifffile
from tqdm.auto import tqdm

TILE_SIZE = 512
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
# Indices into the 13-band Sentinel-2 stack stored at the front of each S2 file.
BAND_INDEX = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 11, 12], dtype=np.int64)
MEAN = np.asarray(
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
STD = np.asarray(
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
EVENT_ID = re.compile(r"^(EMSR\d+|UNOSATST\d+|[A-Z]+\d+)")


def _event_id(stem: str) -> str:
    match = EVENT_ID.match(stem)
    return match.group(1) if match else stem.split("_", 1)[0]


def _packed_label(clear: np.ndarray, land_water: np.ndarray) -> np.ndarray:
    label = np.full(clear.shape, 255, dtype=np.uint8)
    clear_pixels = clear == 1
    label[clear_pixels & (land_water == 1)] = 0
    label[clear_pixels & (land_water == 2)] = 1
    return label


def _read_scene(s2_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = tifffile.imread(s2_path)
    mask = tifffile.imread(gt_path)
    if image.ndim != 3 or image.shape[-1] < 13:
        raise RuntimeError(f"{s2_path.name} is not a 13-band-or-more HWC image: {image.shape}")
    if mask.ndim != 3 or mask.shape[-1] != 2:
        raise RuntimeError(f"{gt_path.name} is not a two-plane HWC mask: {mask.shape}")
    if image.shape[:2] != mask.shape[:2]:
        raise RuntimeError(
            f"{s2_path.name} spatial shape {image.shape[:2]} != mask {mask.shape[:2]}"
        )
    selected = np.nan_to_num(image[..., BAND_INDEX], nan=0.0)
    selected = np.clip(selected, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
    selected = np.moveaxis(selected, -1, 0)
    label = _packed_label(mask[..., 0], mask[..., 1])
    return selected, label


def _windows(height: int, width: int):
    for row in range(0, height, TILE_SIZE):
        for column in range(0, width, TILE_SIZE):
            yield row, column


def iter_tiles(s2_path: Path, gt_path: Path):
    image, label = _read_scene(s2_path, gt_path)
    _, height, width = image.shape
    stem = s2_path.stem
    for row, column in _windows(height, width):
        row_end = min(row + TILE_SIZE, height)
        column_end = min(column + TILE_SIZE, width)
        label_tile = np.full((TILE_SIZE, TILE_SIZE), 255, dtype=np.uint8)
        label_view = label[row:row_end, column:column_end]
        label_tile[: row_end - row, : column_end - column] = label_view
        if not np.any(label_tile != 255):
            continue
        image_tile = np.zeros((len(BANDS), TILE_SIZE, TILE_SIZE), dtype=np.uint16)
        image_tile[:, : row_end - row, : column_end - column] = image[
            :, row:row_end, column:column_end
        ]
        sample_id = f"{stem}_r{row}_c{column}"
        yield sample_id, stem, _event_id(stem), image_tile, label_tile


def _put_metadata(transaction, split: str, sample_ids, source_ids, event_ids, class_weights):
    payloads = {
        b"length": struct.pack("<q", len(sample_ids)),
        b"__len__": pickle.dumps(len(sample_ids), protocol=pickle.HIGHEST_PROTOCOL),
        b"split": pickle.dumps(split, protocol=pickle.HIGHEST_PROTOCOL),
        b"bands": pickle.dumps(BANDS, protocol=pickle.HIGHEST_PROTOCOL),
        b"normalization_mean": pickle.dumps(MEAN.tolist(), protocol=pickle.HIGHEST_PROTOCOL),
        b"normalization_std": pickle.dumps(STD.tolist(), protocol=pickle.HIGHEST_PROTOCOL),
        b"class_weights": pickle.dumps(class_weights, protocol=pickle.HIGHEST_PROTOCOL),
        b"sample_ids": pickle.dumps(sample_ids, protocol=pickle.HIGHEST_PROTOCOL),
        b"source_ids": pickle.dumps(source_ids, protocol=pickle.HIGHEST_PROTOCOL),
        b"event_ids": pickle.dumps(event_ids, protocol=pickle.HIGHEST_PROTOCOL),
        b"tile_size": pickle.dumps(TILE_SIZE, protocol=pickle.HIGHEST_PROTOCOL),
        b"storage_img_dtype": pickle.dumps("uint16", protocol=pickle.HIGHEST_PROTOCOL),
        b"storage_lbl_dtype": pickle.dumps("uint8", protocol=pickle.HIGHEST_PROTOCOL),
        b"compression": pickle.dumps("zlib", protocol=pickle.HIGHEST_PROTOCOL),
    }
    for key, value in payloads.items():
        transaction.put(key, value)


def build_split(source_root: Path, output_dir: Path, split: str, map_size: int) -> dict:
    scenes = sorted((source_root / split / "S2").glob("*.tif"))
    if not scenes:
        raise FileNotFoundError(f"No Sentinel-2 scenes in {source_root / split / 'S2'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"worldfloods_{split}.lmdb"
    if destination.exists():
        destination.unlink()

    sample_ids: list[str] = []
    source_ids: list[str] = []
    event_ids: list[str] = []
    class_counts = np.zeros(2, dtype=np.int64)
    environment = lmdb.open(
        str(destination),
        subdir=False,
        map_size=map_size,
        writemap=True,
        map_async=True,
        lock=True,
        meminit=False,
    )
    try:
        index = 0
        for scene in tqdm(scenes, desc=f"pack {split}", unit="scene"):
            ground_truth = source_root / split / "gt" / scene.name
            records = []
            for sample_id, source_id, event_id, image_tile, label_tile in iter_tiles(
                scene, ground_truth
            ):
                class_counts[0] += int(np.count_nonzero(label_tile == 0))
                class_counts[1] += int(np.count_nonzero(label_tile == 1))
                payload = pickle.dumps(
                    (
                        zlib.compress(np.ascontiguousarray(image_tile).tobytes(), level=1),
                        (len(BANDS), TILE_SIZE, TILE_SIZE),
                        zlib.compress(np.ascontiguousarray(label_tile).tobytes(), level=1),
                        (TILE_SIZE, TILE_SIZE),
                        sample_id,
                    ),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                records.append((sample_id, source_id, event_id, payload))
            if not records:
                continue
            with environment.begin(write=True) as transaction:
                for sample_id, source_id, event_id, payload in records:
                    transaction.put(str(index).encode("ascii"), payload)
                    sample_ids.append(sample_id)
                    source_ids.append(source_id)
                    event_ids.append(event_id)
                    index += 1
        total = int(class_counts.sum())
        if total == 0:
            class_weights = [1.0, 1.0]
        else:
            class_weights = [
                float(total / (2.0 * max(int(class_counts[0]), 1))),
                float(total / (2.0 * max(int(class_counts[1]), 1))),
            ]
        with environment.begin(write=True) as transaction:
            _put_metadata(
                transaction, split, sample_ids, source_ids, event_ids, class_weights
            )
        environment.sync()
    finally:
        environment.close()

    count = len(sample_ids)
    expected = EXPECTED_COUNTS[split]
    print(
        f"{split}: wrote {count} tiles to {destination} "
        f"(expected {expected}; class counts {class_counts.tolist()}; weights {class_weights})"
    )
    if count != expected:
        raise RuntimeError(f"{split} tile count {count} != benchmark count {expected}")
    return {
        "split": split,
        "tiles": count,
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights,
        "path": str(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "WorldFloodsv2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "WorldFloodsv2" / "lmdb",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--map-size-gb", type=int, default=160)
    args = parser.parse_args()
    for split in args.splits:
        build_split(args.source_root, args.output_dir, split, args.map_size_gb * (1024**3))


if __name__ == "__main__":
    main()
