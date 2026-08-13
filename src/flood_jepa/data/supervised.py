"""Supervised WorldFloods LMDB reader for segmentation fine-tuning/evaluation."""

from __future__ import annotations

import os
import pickle
import struct
import zlib
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from flood_jepa.data.worldfloods import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    WORLDFLOODS_BANDS,
    _canonical_band,
    _metadata,
)


IGNORE_INDEX = 255


class _InsufficientSupervisedPixels(RuntimeError):
    """The source tile cannot provide a sufficiently supervised train crop."""


class WorldFloodsSegmentationDataset(Dataset):
    """Return paired image/label crops while preserving the official split."""

    def __init__(
        self,
        lmdb_path: str | Path,
        split: str,
        crop_size: int = 256,
        input_bands: list[str] | None = None,
        normalization: str = "worldfloods",
        augment_d4: bool = False,
        crop_attempts: int = 20,
        minimum_supervised_fraction: float = 0.25,
        sample_retry_attempts: int = 20,
    ) -> None:
        self.lmdb_path = str(Path(lmdb_path).expanduser().resolve())
        self.split = str(split)
        self.crop_size = int(crop_size)
        self.input_bands = list(input_bands or WORLDFLOODS_BANDS)
        self.normalization = str(normalization)
        self.augment_d4 = bool(augment_d4)
        self.crop_attempts = int(crop_attempts)
        self.minimum_supervised_fraction = float(minimum_supervised_fraction)
        self.sample_retry_attempts = int(sample_retry_attempts)
        self._environment = None
        self._environment_pid = None

        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported WorldFloods split: {self.split}")
        if not Path(self.lmdb_path).is_file():
            raise FileNotFoundError(f"WorldFloods LMDB does not exist: {self.lmdb_path}")
        if not 0.0 <= self.minimum_supervised_fraction <= 1.0:
            raise ValueError("minimum_supervised_fraction must be within [0, 1]")
        if self.sample_retry_attempts <= 0:
            raise ValueError("sample_retry_attempts must be positive")

        environment = self._open_environment()
        try:
            with environment.begin(write=False) as transaction:
                packed_length = transaction.get(b"length")
                self.source_length = (
                    struct.unpack("<q", packed_length)[0]
                    if packed_length is not None
                    else _metadata(transaction, b"__len__")
                )
                stored_split = str(_metadata(transaction, b"split", "unknown"))
                self.storage_dtype = np.dtype(
                    _metadata(transaction, b"storage_img_dtype", "uint16")
                )
                self.label_dtype = np.dtype(
                    _metadata(transaction, b"storage_lbl_dtype", "uint8")
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
                self.class_weights = np.asarray(
                    _metadata(transaction, b"class_weights", [1.0, 1.0]),
                    dtype=np.float32,
                )
                self.sample_ids = list(
                    _metadata(
                        transaction,
                        b"sample_ids",
                        [str(index) for index in range(int(self.source_length))],
                    )
                )
                self.source_ids = list(
                    _metadata(transaction, b"source_ids", self.sample_ids)
                )
                self.event_ids = list(_metadata(transaction, b"event_ids", self.source_ids))
                self.tile_size = int(_metadata(transaction, b"tile_size", 512))
        finally:
            environment.close()

        if self.source_length is None:
            raise RuntimeError(f"LMDB has no length metadata: {self.lmdb_path}")
        if stored_split != self.split:
            raise ValueError(
                f"Requested split {self.split!r}, but LMDB declares {stored_split!r}"
            )
        if self.tile_size < self.crop_size or self.tile_size % self.crop_size:
            raise ValueError(
                f"tile_size={self.tile_size} must be a multiple of crop_size={self.crop_size}"
            )
        if self.mean.shape != (len(self.source_bands),) or self.std.shape != (
            len(self.source_bands),
        ):
            raise RuntimeError("WorldFloods normalization metadata has an invalid shape")
        if np.any(self.std <= 0):
            raise RuntimeError("WorldFloods normalization contains non-positive std")

        source_lookup = {
            _canonical_band(name): index for index, name in enumerate(self.source_bands)
        }
        missing = [name for name in self.input_bands if _canonical_band(name) not in source_lookup]
        if missing:
            raise ValueError(f"Requested bands are absent from the LMDB: {missing}")
        self.band_indices = [source_lookup[_canonical_band(name)] for name in self.input_bands]
        self.crops_per_side = self.tile_size // self.crop_size
        self.crops_per_tile = self.crops_per_side**2

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
        multiplier = 1 if self.split == "train" else self.crops_per_tile
        return int(self.source_length) * multiplier

    def _decode(self, source_index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        with self._env().begin(write=False) as transaction:
            payload = transaction.get(str(source_index).encode("ascii"))
        if payload is None:
            raise IndexError(f"Missing LMDB key {source_index}")
        image_bytes, image_shape, label_bytes, label_shape, sample_id = pickle.loads(payload)
        if self.compression == "zlib":
            image_bytes = zlib.decompress(image_bytes)
            label_bytes = zlib.decompress(label_bytes)
        elif self.compression != "none":
            raise ValueError(f"Unsupported LMDB compression: {self.compression}")
        image = np.frombuffer(image_bytes, dtype=self.storage_dtype).reshape(image_shape).copy()
        label = np.frombuffer(label_bytes, dtype=self.label_dtype).reshape(label_shape).copy()
        return torch.from_numpy(image), torch.from_numpy(label.astype(np.int64)), str(sample_id)

    def _training_crop(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        height, width = target.shape
        max_top = height - self.crop_size
        max_left = width - self.crop_size
        valid_mask = target != IGNORE_INDEX
        required = max(
            1,
            int(self.crop_size * self.crop_size * self.minimum_supervised_fraction),
        )
        if int(valid_mask.sum().item()) < required:
            raise _InsufficientSupervisedPixels(
                "WorldFloods tile cannot supply the requested supervised crop fraction"
            )
        best_valid = -1
        for _ in range(self.crop_attempts):
            top = int(torch.randint(0, max_top + 1, (1,)).item())
            left = int(torch.randint(0, max_left + 1, (1,)).item())
            label_crop = target[top : top + self.crop_size, left : left + self.crop_size]
            valid = int((label_crop != IGNORE_INDEX).sum().item())
            if valid > best_valid:
                best_valid = valid
            if valid >= required:
                return (
                    image[:, top : top + self.crop_size, left : left + self.crop_size],
                    label_crop,
                    f"r{top}_c{left}",
                )

        # Random proposals can miss a valid footprint near a tile boundary.
        # Use a summed-area table to locate the crop with the most labeled pixels.
        integral = torch.nn.functional.pad(
            valid_mask.to(torch.int32).cumsum(0).cumsum(1),
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
            raise _InsufficientSupervisedPixels(
                "WorldFloods tile cannot supply the requested supervised crop fraction"
            )
        return (
            image[:, top : top + self.crop_size, left : left + self.crop_size],
            target[top : top + self.crop_size, left : left + self.crop_size],
            f"r{top}_c{left}",
        )

    def _evaluation_crop(
        self, image: torch.Tensor, target: torch.Tensor, crop_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = crop_index // self.crops_per_side
        column = crop_index % self.crops_per_side
        top = row * self.crop_size
        left = column * self.crop_size
        return (
            image[:, top : top + self.crop_size, left : left + self.crop_size],
            target[top : top + self.crop_size, left : left + self.crop_size],
            f"r{top}_c{left}",
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(torch.float32)[self.band_indices]
        if self.normalization == "worldfloods":
            indices = np.asarray(self.band_indices)
            mean = torch.as_tensor(self.mean[indices], dtype=image.dtype).view(-1, 1, 1)
            std = torch.as_tensor(self.std[indices], dtype=image.dtype).view(-1, 1, 1)
            return (image - mean) / std
        if self.normalization == "reflectance":
            return image / 10000.0
        if self.normalization == "raw":
            return image
        raise ValueError(f"Unknown normalization: {self.normalization}")

    def _d4_pair(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.augment_d4:
            return image, target
        transform_index = int(torch.randint(0, 8, (1,)).item())
        if transform_index >= 4:
            image = torch.flip(image, dims=(-1,))
            target = torch.flip(target, dims=(-1,))
        turns = transform_index % 4
        return (
            torch.rot90(image, turns, dims=(-2, -1)),
            torch.rot90(target, turns, dims=(-2, -1)),
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        initial_source_index, crop_index = (
            (int(index), 0)
            if self.split == "train"
            else divmod(int(index), self.crops_per_tile)
        )
        attempts = self.sample_retry_attempts if self.split == "train" else 1

        for attempt in range(attempts):
            source_index = initial_source_index
            if attempt > 0:
                source_index = int(torch.randint(0, self.source_length, (1,)).item())
            image, target, payload_sample_id = self._decode(source_index)
            if image.ndim != 3 or target.ndim != 2 or image.shape[-2:] != target.shape:
                raise RuntimeError(f"Invalid image/label shapes for {payload_sample_id}")
            valid_labels = target[target != IGNORE_INDEX]
            if valid_labels.numel() and torch.any((valid_labels < 0) | (valid_labels > 1)):
                raise RuntimeError(f"Unexpected label value in {payload_sample_id}")

            if self.split == "train":
                try:
                    image, target, crop_id = self._training_crop(image, target)
                except _InsufficientSupervisedPixels:
                    continue
            else:
                image, target, crop_id = self._evaluation_crop(image, target, crop_index)

            # Convert uint16 satellite data before flip/rot90; these CPU kernels
            # are not implemented for torch.uint16 in the server PyTorch build.
            image = self._normalize(image)
            image, target = self._d4_pair(image, target)
            return {
                "image": image.contiguous(),
                "label": target.long().contiguous(),
                "sample_id": f"{self.sample_ids[source_index]}:{crop_id}",
                "source_id": self.source_ids[source_index],
                "event_id": self.event_ids[source_index],
            }

        raise RuntimeError(
            f"Could not find a WorldFloods tile with sufficient supervised pixels after "
            f"{attempts} attempts (initial index {initial_source_index})"
        )
