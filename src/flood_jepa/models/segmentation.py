"""Downstream segmentation decoder for a pretrained Flood-JEPA encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FloodSegmentationDecoder(nn.Module):
    """Fuse four V-JEPA 2.1 encoder levels and predict land/water logits."""

    def __init__(self, encoder_dim: int, decoder_dim: int = 256, classes: int = 2):
        super().__init__()
        self.level_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(encoder_dim, decoder_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(32, decoder_dim),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, decoder_dim),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(decoder_dim, classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]):
        if len(features) != 4:
            raise ValueError(f"Expected four deeply supervised encoder levels, got {len(features)}")
        projected = [layer(feature) for layer, feature in zip(self.level_projections, features)]
        fused = self.fuse(torch.cat(projected, dim=1))
        logits = self.classifier(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class LinearFloodSegmenter(nn.Module):
    """Frozen-encoder linear probe using only the final ViT feature level."""

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int,
        encoder_dim: int,
        classes: int = 2,
    ):
        super().__init__()
        self.backbone = encoder.backbone if hasattr(encoder, "backbone") else encoder
        self.patch_size = int(patch_size)
        self.decoder = nn.Conv2d(encoder_dim, classes, kernel_size=1)
        self.backbone.out_layers = [self.backbone.hierarchical_layers[-1]]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be divisible by the patch size")
        token_levels = self.backbone(images)
        if not isinstance(token_levels, list) or len(token_levels) != 1:
            raise RuntimeError("Linear probe expected exactly one final encoder feature level")
        tokens = token_levels[0]
        feature_map = tokens.transpose(1, 2).reshape(
            images.shape[0],
            tokens.shape[-1],
            height // self.patch_size,
            width // self.patch_size,
        )
        logits = self.decoder(feature_map)
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


class FloodSegmenter(nn.Module):
    """Attach a lightweight dense decoder to the pretrained online encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int,
        encoder_dim: int,
        decoder_dim: int = 256,
        classes: int = 2,
    ):
        super().__init__()
        self.backbone = encoder.backbone if hasattr(encoder, "backbone") else encoder
        self.patch_size = int(patch_size)
        self.decoder = FloodSegmentationDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            classes=classes,
        )
        self.backbone.out_layers = list(self.backbone.hierarchical_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be divisible by the patch size")
        token_levels = self.backbone(images)
        grid_height = height // self.patch_size
        grid_width = width // self.patch_size
        feature_levels = [
            tokens.transpose(1, 2).reshape(images.shape[0], tokens.shape[-1], grid_height, grid_width)
            for tokens in token_levels
        ]
        return self.decoder(feature_levels, output_size=(height, width))
