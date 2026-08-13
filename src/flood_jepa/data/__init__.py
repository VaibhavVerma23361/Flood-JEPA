"""Dataset adapters used by Flood-JEPA."""

from .worldfloods import WORLDFLOODS_BANDS, WorldFloodsPretrainDataset

__all__ = ["WORLDFLOODS_BANDS", "WorldFloodsPretrainDataset"]
from flood_jepa.data.supervised import IGNORE_INDEX, WorldFloodsSegmentationDataset
from flood_jepa.data.worldfloods import WorldFloodsPretrainDataset

__all__ = [
    "IGNORE_INDEX",
    "WorldFloodsPretrainDataset",
    "WorldFloodsSegmentationDataset",
]
