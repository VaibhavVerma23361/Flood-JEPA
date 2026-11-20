"""Downstream segmentation decoder for a pretrained Flood-JEPA encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    """Choose the largest conventional GroupNorm count that divides channels."""
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualConvUnit(nn.Module):
    """A normalization-safe residual refinement unit for dense features."""

    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.activation(self.norm1(self.conv1(features)))
        features = self.norm2(self.conv2(features))
        return self.activation(features + residual)


class _FeatureFusionBlock(nn.Module):
    """Merge a coarse top-down feature with a same-resolution skip feature."""

    def __init__(self, channels: int):
        super().__init__()
        self.merge = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            _ResidualConvUnit(channels),
            _ResidualConvUnit(channels),
        )

    def forward(self, coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        coarse = F.interpolate(
            coarse,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(self.merge(torch.cat((coarse, skip), dim=1)))


class _UpsampleBlock(nn.Module):
    """Bilinear upsampling followed by learned spatial refinement."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
            _ResidualConvUnit(output_channels),
        )

    def forward(self, features: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        features = F.interpolate(
            features,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(features)


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


class DPTFloodSegmentationDecoder(nn.Module):
    """DPT-style pyramid decoder for four same-grid ViT feature levels.

    V-JEPA exposes four intermediate representations at the patch-grid
    resolution. The decoder reassembles them at 4x, 2x, 1x, and 0.5x grid
    scales, then performs top-down fusion before two learned upsampling stages.
    It deliberately has no raw-image skip path, so segmentation quality remains
    attributable to the pretrained encoder representation.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int = 256,
        classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if decoder_dim < 16:
            raise ValueError("decoder_dim must be at least 16")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        groups = _group_count(decoder_dim)
        self.level_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(encoder_dim, decoder_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, decoder_dim),
                    nn.GELU(),
                    _ResidualConvUnit(decoder_dim),
                )
                for _ in range(4)
            ]
        )
        self.deep_refine = _ResidualConvUnit(decoder_dim)
        self.fuse_level_3 = _FeatureFusionBlock(decoder_dim)
        self.fuse_level_2 = _FeatureFusionBlock(decoder_dim)
        self.fuse_level_1 = _FeatureFusionBlock(decoder_dim)

        half_dim = max(16, decoder_dim // 2)
        quarter_dim = max(16, decoder_dim // 4)
        self.upsample_half = _UpsampleBlock(decoder_dim, half_dim)
        self.upsample_full = _UpsampleBlock(half_dim, quarter_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(quarter_dim, classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]):
        if len(features) != 4:
            raise ValueError(f"Expected four encoder feature levels, got {len(features)}")
        grid_size = features[0].shape[-2:]
        if any(feature.shape[-2:] != grid_size for feature in features):
            raise ValueError("DPT reassembly expects all encoder levels on one patch grid")

        grid_height, grid_width = grid_size
        pyramid_sizes = (
            (grid_height * 4, grid_width * 4),
            (grid_height * 2, grid_width * 2),
            grid_size,
            (max(1, grid_height // 2), max(1, grid_width // 2)),
        )
        pyramid = []
        for projection, feature, size in zip(
            self.level_projections,
            features,
            pyramid_sizes,
        ):
            feature = projection(feature)
            pyramid.append(
                F.interpolate(
                    feature,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        decoded = self.deep_refine(pyramid[3])
        decoded = self.fuse_level_3(decoded, pyramid[2])
        decoded = self.fuse_level_2(decoded, pyramid[1])
        decoded = self.fuse_level_1(decoded, pyramid[0])

        half_size = (
            min(output_size[0], decoded.shape[-2] * 2),
            min(output_size[1], decoded.shape[-1] * 2),
        )
        decoded = self.upsample_half(decoded, half_size)
        decoded = self.upsample_full(decoded, output_size)
        return self.classifier(self.dropout(decoded))


class _PyramidPoolingModule(nn.Module):
    """UPerNet pyramid pooling over the deepest encoder feature."""

    def __init__(self, channels: int, pool_scales: tuple[int, ...]):
        super().__init__()
        branch_channels = max(16, channels // len(pool_scales))
        self.pool_scales = pool_scales
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(_group_count(branch_channels), branch_channels),
                    nn.GELU(),
                )
                for _ in pool_scales
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                channels + branch_channels * len(pool_scales),
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = [features]
        for scale, branch in zip(self.pool_scales, self.branches):
            pooled = F.adaptive_avg_pool2d(features, output_size=(scale, scale))
            pooled = branch(pooled)
            outputs.append(
                F.interpolate(
                    pooled,
                    size=features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return self.bottleneck(torch.cat(outputs, dim=1))


class UPerNetFloodSegmentationDecoder(nn.Module):
    """Common UPerNet-512 benchmark decoder for ViT feature levels.

    Four same-resolution ViT feature maps are first reassembled at strides
    4, 8, 16, and 32. Pyramid pooling supplies global context at the deepest
    scale, and a top-down feature pyramid fuses semantic and spatial detail.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int = 512,
        classes: int = 2,
        dropout: float = 0.1,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        if decoder_dim < 16:
            raise ValueError("decoder_dim must be at least 16")
        if not pool_scales or any(scale <= 0 for scale in pool_scales):
            raise ValueError("pool_scales must contain positive integers")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        groups = _group_count(decoder_dim)
        self.level_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(encoder_dim, decoder_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, decoder_dim),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        self.psp = _PyramidPoolingModule(decoder_dim, pool_scales)
        self.fpn_refines = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        decoder_dim,
                        decoder_dim,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, decoder_dim),
                    nn.GELU(),
                )
                for _ in range(3)
            ]
        )
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                decoder_dim * 4,
                decoder_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, decoder_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(decoder_dim, classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]):
        if len(features) != 4:
            raise ValueError(f"Expected four encoder feature levels, got {len(features)}")
        grid_size = features[0].shape[-2:]
        if any(feature.shape[-2:] != grid_size for feature in features):
            raise ValueError("UPerNet reassembly expects one shared ViT patch grid")

        grid_height, grid_width = grid_size
        pyramid_sizes = (
            (grid_height * 4, grid_width * 4),
            (grid_height * 2, grid_width * 2),
            grid_size,
            (max(1, grid_height // 2), max(1, grid_width // 2)),
        )
        pyramid = []
        for projection, feature, size in zip(
            self.level_projections,
            features,
            pyramid_sizes,
        ):
            projected = projection(feature)
            pyramid.append(
                F.interpolate(
                    projected,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        outputs: list[torch.Tensor] = [pyramid[0], pyramid[1], pyramid[2], self.psp(pyramid[3])]
        for level in range(2, -1, -1):
            top_down = F.interpolate(
                outputs[level + 1],
                size=outputs[level].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            outputs[level] = self.fpn_refines[level](outputs[level] + top_down)

        fusion_size = outputs[0].shape[-2:]
        fused = torch.cat(
            [
                output
                if output.shape[-2:] == fusion_size
                else F.interpolate(
                    output,
                    size=fusion_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for output in outputs
            ],
            dim=1,
        )
        logits = self.classifier(self.fpn_bottleneck(fused))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class _PangaeaFeature2Pyramid(nn.Module):
    """Pinned PANGAEA Feature2Pyramid operations for four flat ViT levels."""

    def __init__(self, channels: list[int]):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("PANGAEA SegUPerNet requires exactly four feature levels")
        self.ops = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(channels[0], channels[0], kernel_size=2, stride=2),
                    nn.SyncBatchNorm(channels[0]),
                    nn.GELU(),
                    nn.ConvTranspose2d(channels[0], channels[0], kernel_size=2, stride=2),
                ),
                nn.ConvTranspose2d(channels[1], channels[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.ops):
            raise ValueError("PANGAEA Feature2Pyramid received the wrong number of levels")
        return [operation(feature) for operation, feature in zip(self.ops, features)]


class _PangaeaPPM(nn.ModuleList):
    def __init__(
        self,
        pool_scales: tuple[int, ...],
        input_channels: int,
        output_channels: int,
    ):
        super().__init__(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(input_channels, output_channels, kernel_size=1),
                    nn.SyncBatchNorm(output_channels),
                    nn.ReLU(inplace=True),
                )
                for scale in pool_scales
            ]
        )

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        return [
            F.interpolate(
                branch(features),
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for branch in self
        ]


class PangaeaUPerNetFloodSegmentationDecoder(nn.Module):
    """PANGAEA SegUPerNet at pinned commit ca19b837... with width 512.

    The implementation mirrors the single-temporal ``SegUPerNet`` path:
    learned Feature2Pyramid rescaling, SyncBatchNorm/ReLU PSP and FPN heads,
    Dropout2d(0.1), and a two-class 1x1 classifier.
    """

    source_revision = "ca19b837a16797cd809225225b671473c7c93768"

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int = 512,
        classes: int = 2,
        dropout: float = 0.1,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        enforce_benchmark_width: bool = True,
    ):
        super().__init__()
        if enforce_benchmark_width and decoder_dim != 512:
            raise ValueError("The locked benchmark requires PANGAEA UPerNet width 512")
        if decoder_dim < 32:
            raise ValueError("SegUPerNet decoder_dim must be at least 32")
        if dropout != 0.1:
            raise ValueError("The locked benchmark requires PANGAEA UPerNet dropout 0.1")
        self.input_channels = [encoder_dim] * 4
        self.channels = decoder_dim
        self.neck = _PangaeaFeature2Pyramid(self.input_channels)
        self.psp_modules = _PangaeaPPM(
            pool_scales,
            self.input_channels[-1],
            decoder_dim,
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                self.input_channels[-1] + len(pool_scales) * decoder_dim,
                decoder_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, decoder_dim, kernel_size=1),
                    nn.SyncBatchNorm(decoder_dim),
                    nn.ReLU(inplace=False),
                )
                for channels in self.input_channels[:-1]
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
                    nn.SyncBatchNorm(decoder_dim),
                    nn.ReLU(inplace=False),
                )
                for _ in self.input_channels[:-1]
            ]
        )
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                len(self.input_channels) * decoder_dim,
                decoder_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(decoder_dim, classes, kernel_size=1)

    def _psp_forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = [features, *self.psp_modules(features)]
        return self.bottleneck(torch.cat(pooled, dim=1))

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]):
        features = self.neck(features)
        laterals = [
            lateral(features[index])
            for index, lateral in enumerate(self.lateral_convs)
        ]
        laterals.append(self._psp_forward(features[-1]))
        for level in range(len(laterals) - 1, 0, -1):
            laterals[level - 1] = laterals[level - 1] + F.interpolate(
                laterals[level],
                size=laterals[level - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        outputs = [
            self.fpn_convs[index](laterals[index])
            for index in range(len(laterals) - 1)
        ]
        outputs.append(laterals[-1])
        fusion_size = outputs[0].shape[-2:]
        outputs = [
            output
            if output.shape[-2:] == fusion_size
            else F.interpolate(
                output,
                size=fusion_size,
                mode="bilinear",
                align_corners=False,
            )
            for output in outputs
        ]
        decoded = self.fpn_bottleneck(torch.cat(outputs, dim=1))
        logits = self.classifier(self.dropout(decoded))
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class PangaeaLiteUPerNetFloodSegmentationDecoder(
    PangaeaUPerNetFloodSegmentationDecoder
):
    """Width-reduced SegUPerNet for the ViT-T/A30 efficiency experiment."""

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int = 128,
        classes: int = 2,
        dropout: float = 0.1,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            classes=classes,
            dropout=dropout,
            pool_scales=pool_scales,
            enforce_benchmark_width=False,
        )


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


class DPTFloodSegmenter(nn.Module):
    """Attach the DPT-style multi-scale decoder to a Flood-JEPA encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int,
        encoder_dim: int,
        decoder_dim: int = 256,
        classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = encoder.backbone if hasattr(encoder, "backbone") else encoder
        self.patch_size = int(patch_size)
        self.decoder = DPTFloodSegmentationDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            classes=classes,
            dropout=dropout,
        )
        self.backbone.out_layers = list(self.backbone.hierarchical_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be divisible by the patch size")
        token_levels = self.backbone(images)
        if not isinstance(token_levels, list) or len(token_levels) != 4:
            raise RuntimeError("DPT decoder expected four encoder feature levels")
        grid_height = height // self.patch_size
        grid_width = width // self.patch_size
        feature_levels = [
            tokens.transpose(1, 2).reshape(
                images.shape[0],
                tokens.shape[-1],
                grid_height,
                grid_width,
            )
            for tokens in token_levels
        ]
        return self.decoder(feature_levels, output_size=(height, width))


class UPerNetFloodSegmenter(nn.Module):
    """Attach the common UPerNet benchmark decoder to a Flood-JEPA encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int,
        encoder_dim: int,
        decoder_dim: int = 512,
        classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = encoder.backbone if hasattr(encoder, "backbone") else encoder
        self.patch_size = int(patch_size)
        self.decoder = UPerNetFloodSegmentationDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            classes=classes,
            dropout=dropout,
        )
        self.backbone.out_layers = list(self.backbone.hierarchical_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be divisible by the patch size")
        token_levels = self.backbone(images)
        if not isinstance(token_levels, list) or len(token_levels) != 4:
            raise RuntimeError("UPerNet decoder expected four encoder feature levels")
        grid_height = height // self.patch_size
        grid_width = width // self.patch_size
        feature_levels = [
            tokens.transpose(1, 2).reshape(
                images.shape[0],
                tokens.shape[-1],
                grid_height,
                grid_width,
            )
            for tokens in token_levels
        ]
        return self.decoder(feature_levels, output_size=(height, width))


class PangaeaUPerNetFloodSegmenter(nn.Module):
    """Frozen Flood-JEPA encoder with the benchmark-controlled SegUPerNet."""

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int,
        encoder_dim: int,
        decoder_dim: int = 512,
        classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = encoder.backbone if hasattr(encoder, "backbone") else encoder
        self.patch_size = int(patch_size)
        self.decoder = PangaeaUPerNetFloodSegmentationDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            classes=classes,
            dropout=dropout,
        )
        self.backbone.out_layers = list(self.backbone.hierarchical_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be divisible by the patch size")
        token_levels = self.backbone(images)
        if not isinstance(token_levels, list) or len(token_levels) != 4:
            raise RuntimeError("PANGAEA UPerNet expected four encoder feature levels")
        grid_height = height // self.patch_size
        grid_width = width // self.patch_size
        feature_levels = [
            tokens.transpose(1, 2).reshape(
                images.shape[0],
                tokens.shape[-1],
                grid_height,
                grid_width,
            )
            for tokens in token_levels
        ]
        return self.decoder(feature_levels, output_size=(height, width))


class PangaeaLiteUPerNetFloodSegmenter(nn.Module):
    """Frozen Flood-JEPA encoder with the width-reduced SegUPerNet decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int,
        encoder_dim: int,
        decoder_dim: int = 128,
        classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = encoder.backbone if hasattr(encoder, "backbone") else encoder
        self.patch_size = int(patch_size)
        self.decoder = PangaeaLiteUPerNetFloodSegmentationDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            classes=classes,
            dropout=dropout,
        )
        self.backbone.out_layers = list(self.backbone.hierarchical_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be divisible by the patch size")
        token_levels = self.backbone(images)
        if not isinstance(token_levels, list) or len(token_levels) != 4:
            raise RuntimeError("SegUPerNet-Lite expected four encoder feature levels")
        grid_height = height // self.patch_size
        grid_width = width // self.patch_size
        feature_levels = [
            tokens.transpose(1, 2).reshape(
                images.shape[0],
                tokens.shape[-1],
                grid_height,
                grid_width,
            )
            for tokens in token_levels
        ]
        return self.decoder(feature_levels, output_size=(height, width))
