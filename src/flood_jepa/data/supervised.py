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
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

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
        d4_probability: float | None = None,
        rotate_probability: float = 0.0,
        rotate_limit_degrees: float = 0.0,
        crop_attempts: int = 10,
        minimum_supervised_fraction: float = 0.0,
        sample_retry_attempts: int = 1,
        allow_source_ids: list[str] | None = None,
    ) -> None:
        self.lmdb_path = str(Path(lmdb_path).expanduser().resolve())
        self.split = str(split)
        self.crop_size = int(crop_size)
        self.input_bands = list(input_bands or WORLDFLOODS_BANDS)
        self.normalization = str(normalization)
        self.augment_d4 = bool(augment_d4)
        # Retain the old boolean API for probe/ablation configs: augment_d4=True
        # means apply a random D4 transform every time unless an explicit
        # probability is provided.
        self.d4_probability = (
            float(d4_probability)
            if d4_probability is not None
            else (1.0 if self.augment_d4 else 0.0)
        )
        self.rotate_probability = float(rotate_probability)
        self.rotate_limit_degrees = float(rotate_limit_degrees)
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
        if not 0.0 <= self.d4_probability <= 1.0:
            raise ValueError("d4_probability must be within [0, 1]")
        if not 0.0 <= self.rotate_probability <= 1.0:
            raise ValueError("rotate_probability must be within [0, 1]")
        if not 0.0 <= self.rotate_limit_degrees <= 180.0:
            raise ValueError("rotate_limit_degrees must be within [0, 180]")
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
        self.index_map = list(range(int(self.source_length)))
        if allow_source_ids is not None:
            allowed = {str(source_id) for source_id in allow_source_ids}
            unknown = allowed.difference(self.source_ids)
            if unknown:
                raise ValueError(
                    f"{len(unknown)} requested source scenes are absent from {self.lmdb_path}"
                )
            self.index_map = [
                index
                for index, source_id in enumerate(self.source_ids)
                if source_id in allowed
            ]
            if len(self.index_map) < int(self.source_length):
                # Partial-label runs must not consume full-label class frequencies.
                self.class_weights = np.ones(2, dtype=np.float32)
            if not self.index_map:
                raise ValueError("Source allow-list matched no WorldFloods tiles")

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
        return len(self.index_map)

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
        # The executable benchmark accepts any crop containing at least one
        # supervised pixel. It does not bias sampling toward densely labeled
        # or easier tiles.
        required = 1
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

        # Match the benchmark fallback: select a supervised pixel, then sample
        # a crop origin from the complete range of origins that contains it.
        coordinates = torch.nonzero(valid_mask, as_tuple=False)
        selected = coordinates[int(torch.randint(0, coordinates.shape[0], (1,)).item())]
        row, column = int(selected[0]), int(selected[1])
        top_min = max(0, row - self.crop_size + 1)
        top_max = min(row, max_top)
        left_min = max(0, column - self.crop_size + 1)
        left_max = min(column, max_left)
        top = int(torch.randint(top_min, top_max + 1, (1,)).item())
        left = int(torch.randint(left_min, left_max + 1, (1,)).item())
        return (
            image[:, top : top + self.crop_size, left : left + self.crop_size],
            target[top : top + self.crop_size, left : left + self.crop_size],
            f"r{top}_c{left}",
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(torch.float32)[self.band_indices]
        return self._normalize_selected(image)

    def _normalize_selected(self, image: torch.Tensor) -> torch.Tensor:
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
        if self.d4_probability <= 0.0 or torch.rand(()) >= self.d4_probability:
            return image, target
        # The gate and transform sampling are separate in the benchmark. The
        # identity is one of the eight D4 members.
        transform_index = int(torch.randint(0, 8, (1,)).item())
        if transform_index >= 4:
            image = torch.flip(image, dims=(-1,))
            target = torch.flip(target, dims=(-1,))
        turns = transform_index % 4
        return (
            torch.rot90(image, turns, dims=(-2, -1)),
            torch.rot90(target, turns, dims=(-2, -1)),
        )

    def _rotate_pair(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.rotate_probability <= 0.0
            or self.rotate_limit_degrees <= 0.0
            or torch.rand(()) >= self.rotate_probability
        ):
            return image, target
        angle = float(
            torch.empty(()).uniform_(
                -self.rotate_limit_degrees,
                self.rotate_limit_degrees,
            )
        )
        image = TF.rotate(
            image,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        target = TF.rotate(
            target.unsqueeze(0),
            angle,
            interpolation=InterpolationMode.NEAREST,
            fill=IGNORE_INDEX,
        ).squeeze(0)
        return image, target

    def __getitem__(self, index: int) -> dict[str, object]:
        source_index = int(self.index_map[int(index)])
        image, target, payload_sample_id = self._decode(source_index)
        if image.ndim != 3 or target.ndim != 2 or image.shape[-2:] != target.shape:
            raise RuntimeError(f"Invalid image/label shapes for {payload_sample_id}")
        valid_labels = target[target != IGNORE_INDEX]
        if valid_labels.numel() and torch.any((valid_labels < 0) | (valid_labels > 1)):
            raise RuntimeError(f"Unexpected label value in {payload_sample_id}")

        if self.split == "train":
            image, target, crop_id = self._training_crop(image, target)
            # Geometry is applied in raw-value space so rotation fill=0 means
            # raw Sentinel-2 no-data, exactly as in the executable benchmark.
            image = image.to(torch.float32)[self.band_indices]
            original_image, original_target = image, target
            image, target = self._d4_pair(image, target)
            image, target = self._rotate_pair(image, target)
            if not torch.any(target != IGNORE_INDEX):
                image, target = original_image, original_target
            image = self._normalize_selected(image)
        else:
            # Validation/test return one complete 512x512 tile. The evaluation
            # engine performs deterministic model-native sliding windows and
            # reconstructs the tile before computing any grouped metric.
            crop_id = "full"
            image = self._normalize(image)

        return {
            "image": image.contiguous(),
            "label": target.long().contiguous(),
            "sample_id": self.sample_ids[source_index],
            "source_id": self.source_ids[source_index],
            "event_id": self.event_ids[source_index],
            "crop_id": crop_id,
        }
