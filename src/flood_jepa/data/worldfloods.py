"""Label-free reader for the existing packed WorldFloods V2 LMDB.

The segmentation label is intentionally not returned during JEPA pretraining.
Crop selection uses only non-zero image pixels, so segmentation annotations do
not influence the self-supervised sampling policy.
"""

from __future__ import annotations

import argparse
import os
import pickle
import struct
import zlib
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


WORLDFLOODS_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
DEFAULT_MEAN = np.asarray(
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
    dtype=np.float32,
)
DEFAULT_STD = np.asarray(
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
    dtype=np.float32,
)


class _InsufficientValidPixels(RuntimeError):
    """The image has no crop satisfying the configured no-data threshold."""


def _metadata(transaction, key: bytes, default=None):
    value = transaction.get(key)
    return default if value is None else pickle.loads(value)


def _canonical_band(name: str) -> str:
    name = name.upper()
    if name.startswith("B0") and name[2:].isdigit():
        return "B" + name[2:]
    return name


class WorldFloodsPretrainDataset(Dataset):
    """Read 10-band WorldFloods tiles and sample label-free 256-pixel crops."""

    def __init__(
        self,
        lmdb_path: str | Path,
        crop_size: int = 256,
        input_bands: list[str] | None = None,
        normalization: str = "worldfloods",
        augment_d4: bool = True,
        minimum_nonpadding_fraction: float = 0.25,
        crop_attempts: int = 20,
        sample_retry_attempts: int = 20,
    ) -> None:
        self.lmdb_path = str(Path(lmdb_path).expanduser().resolve())
        self.crop_size = int(crop_size)
        self.input_bands = list(input_bands or WORLDFLOODS_BANDS)
        self.normalization = str(normalization)
        self.augment_d4 = bool(augment_d4)
        self.minimum_nonpadding_fraction = float(minimum_nonpadding_fraction)
        self.crop_attempts = int(crop_attempts)
        self.sample_retry_attempts = int(sample_retry_attempts)
        self._environment = None
        self._environment_pid = None

        if not Path(self.lmdb_path).is_file():
            raise FileNotFoundError(f"WorldFloods training LMDB does not exist: {self.lmdb_path}")
        if self.crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if not 0.0 <= self.minimum_nonpadding_fraction <= 1.0:
            raise ValueError("minimum_nonpadding_fraction must be within [0, 1]")
        if self.sample_retry_attempts <= 0:
            raise ValueError("sample_retry_attempts must be positive")

        environment = self._open_environment()
        try:
            with environment.begin(write=False) as transaction:
                packed_length = transaction.get(b"length")
                self.length = (
                    struct.unpack("<q", packed_length)[0]
                    if packed_length is not None
                    else _metadata(transaction, b"__len__")
                )
                if self.length is None:
                    raise RuntimeError(f"LMDB has no length metadata: {self.lmdb_path}")
                self.split = str(_metadata(transaction, b"split", "unknown"))
                self.storage_dtype = np.dtype(
                    _metadata(transaction, b"storage_img_dtype", "uint16")
                )
                self.compression = str(_metadata(transaction, b"compression", "none"))
                self.source_bands = list(
                    _metadata(transaction, b"bands", WORLDFLOODS_BANDS)
                )
                self.mean = np.asarray(
                    _metadata(transaction, b"normalization_mean", DEFAULT_MEAN.tolist()),
                    dtype=np.float32,
                )
                self.std = np.asarray(
                    _metadata(transaction, b"normalization_std", DEFAULT_STD.tolist()),
                    dtype=np.float32,
                )
                self.sample_ids = list(
                    _metadata(
                        transaction,
                        b"sample_ids",
                        [str(index) for index in range(int(self.length))],
                    )
                )
        finally:
            environment.close()

        if self.split != "train":
            raise ValueError(
                f"JEPA pretraining must use the WorldFloods train split, got split={self.split!r}"
            )
        if self.mean.shape != (len(self.source_bands),) or self.std.shape != (
            len(self.source_bands),
        ):
            raise RuntimeError(
                f"Invalid normalization metadata: mean={self.mean.shape}, std={self.std.shape}, "
                f"bands={len(self.source_bands)}"
            )
        if np.any(self.std <= 0):
            raise RuntimeError("WorldFloods normalization contains a non-positive standard deviation")

        source_lookup = {
            _canonical_band(name): index for index, name in enumerate(self.source_bands)
        }
        missing = [name for name in self.input_bands if _canonical_band(name) not in source_lookup]
        if missing:
            raise ValueError(f"Requested bands are absent from the LMDB: {missing}")
        self.band_indices = [source_lookup[_canonical_band(name)] for name in self.input_bands]

    def _open_environment(self):
        return lmdb.open(
            self.lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
        )

    def _env(self):
        pid = os.getpid()
        if self._environment is None or self._environment_pid != pid:
            if self._environment is not None:
                self._environment.close()
            self._environment = self._open_environment()
            self._environment_pid = pid
        return self._environment

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_environment"] = None
        state["_environment_pid"] = None
        return state

    def __len__(self) -> int:
        return int(self.length)

    def _decode_image(self, index: int) -> tuple[np.ndarray, str]:
        with self._env().begin(write=False) as transaction:
            payload = transaction.get(str(index).encode("ascii"))
        if payload is None:
            raise IndexError(f"Missing LMDB key {index}")
        image_bytes, image_shape, _label_bytes, _label_shape, payload_sample_id = pickle.loads(
            payload
        )
        if self.compression == "zlib":
            image_bytes = zlib.decompress(image_bytes)
        elif self.compression != "none":
            raise ValueError(f"Unsupported LMDB compression: {self.compression}")
        image = np.frombuffer(image_bytes, dtype=self.storage_dtype).reshape(image_shape).copy()
        return image, str(payload_sample_id)

    def _random_crop(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        if height < self.crop_size or width < self.crop_size:
            raise RuntimeError(
                f"Tile {height}x{width} is smaller than requested crop {self.crop_size}"
            )
        valid = torch.any(image != 0, dim=0)
        required = int(self.crop_size * self.crop_size * self.minimum_nonpadding_fraction)
        if int(valid.sum().item()) < required:
            raise _InsufficientValidPixels(
                "WorldFloods tile cannot supply the requested non-padding crop fraction"
            )
        if height == self.crop_size and width == self.crop_size:
            return image

        max_top = height - self.crop_size
        max_left = width - self.crop_size
        best_crop = None
        best_valid = -1
        for _ in range(self.crop_attempts):
            top = int(torch.randint(0, max_top + 1, (1,)).item())
            left = int(torch.randint(0, max_left + 1, (1,)).item())
            valid_count = int(
                valid[top : top + self.crop_size, left : left + self.crop_size].sum().item()
            )
            candidate = image[:, top : top + self.crop_size, left : left + self.crop_size]
            if valid_count > best_valid:
                best_crop = candidate
                best_valid = valid_count
            if valid_count >= required:
                return candidate
        # Random proposals can miss a small valid footprint. A summed-area table
        # finds the globally best crop without scanning segmentation labels.
        integral = torch.nn.functional.pad(
            valid.to(torch.int32).cumsum(0).cumsum(1),
            (1, 0, 1, 0),
        )
        counts = (
            integral[self.crop_size :, self.crop_size :]
            - integral[: -self.crop_size, self.crop_size :]
            - integral[self.crop_size :, : -self.crop_size]
            + integral[: -self.crop_size, : -self.crop_size]
        )
        flat_index = int(torch.argmax(counts).item())
        top = flat_index // counts.shape[1]
        left = flat_index % counts.shape[1]
        best_valid = int(counts[top, left].item())
        if best_valid < required:
            raise _InsufficientValidPixels(
                "WorldFloods tile cannot supply the requested non-padding crop fraction"
            )
        return image[:, top : top + self.crop_size, left : left + self.crop_size]

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(torch.float32)[self.band_indices]
        if self.normalization == "worldfloods":
            indices = np.asarray(self.band_indices)
            mean = torch.as_tensor(self.mean[indices], dtype=image.dtype).view(-1, 1, 1)
            std = torch.as_tensor(self.std[indices], dtype=image.dtype).view(-1, 1, 1)
            image = (image - mean) / std
        elif self.normalization == "reflectance":
            image = image / 10000.0
        elif self.normalization != "raw":
            raise ValueError(f"Unknown normalization: {self.normalization}")
        return image

    def _d4(self, image: torch.Tensor) -> torch.Tensor:
        if not self.augment_d4:
            return image
        transform_index = int(torch.randint(0, 8, (1,)).item())
        if transform_index >= 4:
            image = torch.flip(image, dims=(-1,))
        return torch.rot90(image, transform_index % 4, dims=(-2, -1))

    def __getitem__(self, index: int) -> dict[str, object]:
        candidate_index = int(index)
        for attempt in range(self.sample_retry_attempts):
            if attempt > 0:
                candidate_index = int(torch.randint(0, len(self), (1,)).item())
            image_array, payload_sample_id = self._decode_image(candidate_index)
            if image_array.ndim != 3 or image_array.shape[0] != len(self.source_bands):
                raise RuntimeError(
                    f"Unexpected image shape {image_array.shape}: {payload_sample_id}"
                )

            # Packed WorldFloods can contain fully padded tiles. They carry no
            # self-supervised signal, so retry another train image without ever
            # consulting its segmentation label.
            if not np.any(image_array):
                continue

            image = torch.from_numpy(image_array)
            try:
                image = self._random_crop(image)
            except _InsufficientValidPixels:
                continue
            image = self._normalize(image)
            image = self._d4(image)
            if not torch.isfinite(image).all():
                raise RuntimeError(f"Non-finite image values in {payload_sample_id}")
            return {
                "image": image.contiguous(),
                "sample_id": self.sample_ids[candidate_index],
            }

        raise RuntimeError(
            f"Could not find a non-padding WorldFloods tile after "
            f"{self.sample_retry_attempts} attempts (initial index {index})"
        )

    def summary(self) -> dict[str, object]:
        return {
            "path": self.lmdb_path,
            "split": self.split,
            "samples": len(self),
            "source_bands": self.source_bands,
            "input_bands": self.input_bands,
            "crop_size": self.crop_size,
            "normalization": self.normalization,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the packed WorldFloods training LMDB")
    parser.add_argument("lmdb_path", type=Path)
    parser.add_argument("--crop-size", type=int, default=256)
    args = parser.parse_args()
    dataset = WorldFloodsPretrainDataset(args.lmdb_path, crop_size=args.crop_size)
    print(dataset.summary())
    sample = dataset[0]
    print({"sample_id": sample["sample_id"], "image_shape": tuple(sample["image"].shape)})


if __name__ == "__main__":
    main()
