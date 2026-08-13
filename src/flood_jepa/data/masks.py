"""Collation adapter around the official V-JEPA 2.1 block-mask generator."""

from __future__ import annotations

from torch.utils.data import default_collate

from flood_jepa.vendor import activate_upstream

activate_upstream()

from src.masks.multiseq_multiblock3d import MaskCollator  # noqa: E402


class WorldFloodsMaskCollator:
    """Collate dictionaries and generate the official single-image JEPA masks."""

    def __init__(self, mask_config: list[dict], crop_size: int, patch_size: int) -> None:
        self._official = MaskCollator(
            cfgs_mask=mask_config,
            dataset_fpcs=[1],
            crop_size=crop_size,
            patch_size=patch_size,
            tubelet_size=1,
        )

    def __call__(self, batch: list[dict]):
        collated = default_collate(batch)
        batch_size = len(batch)
        masks_encoder = []
        masks_predictor = []
        for generator in self._official.mask_generators[1]:
            encoder_mask, predictor_mask = generator(batch_size)
            masks_encoder.append(encoder_mask)
            masks_predictor.append(predictor_mask)
        return collated, masks_encoder, masks_predictor

    def step(self) -> None:
        self._official.step()

