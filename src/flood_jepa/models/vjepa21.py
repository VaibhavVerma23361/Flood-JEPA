"""Image-only multispectral V-JEPA 2.1 using the official Meta modules."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from flood_jepa.vendor import activate_upstream

activate_upstream()

import app.vjepa_2_1.models.predictor as predictor_models  # noqa: E402
import app.vjepa_2_1.models.vision_transformer as encoder_models  # noqa: E402
from app.vjepa_2_1.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper  # noqa: E402


@dataclass(frozen=True)
class FloodJEPAInfo:
    model_name: str
    input_channels: int
    image_size: int
    patch_size: int
    encoder_dim: int
    predictor_dim: int
    predictor_depth: int
    modality_embedding: bool


class FloodJEPA(nn.Module):
    """Online encoder, EMA target encoder, and V-JEPA 2.1 predictor."""

    def __init__(
        self,
        online_encoder: nn.Module,
        predictor: nn.Module,
        info: FloodJEPAInfo,
        sigreg_projector: nn.Module | None = None,
        sigreg_levels: int = 4,
    ):
        super().__init__()
        self.online_encoder = MultiSeqWrapper(online_encoder)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = PredictorMultiSeqWrapper(predictor)
        self.sigreg_projector = sigreg_projector
        self.sigreg_levels = int(sigreg_levels)
        self.info = info
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.target_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def encode_target(self, images: torch.Tensor) -> torch.Tensor:
        return self.target_encoder([images], training_mode=True)[0]

    def encode_context(self, images: torch.Tensor, masks_encoder: list[torch.Tensor]):
        return self.online_encoder(
            [images],
            [masks_encoder],
            training_mode=True,
        )

    def predict(
        self,
        context,
        masks_encoder: list[torch.Tensor],
        masks_predictor: list[torch.Tensor],
    ):
        return self.predictor(
            context,
            [masks_encoder],
            [masks_predictor],
            mod="image",
        )

    def forward(
        self,
        images: torch.Tensor,
        masks_encoder: list[torch.Tensor],
        masks_predictor: list[torch.Tensor],
    ):
        """Run the complete train-time graph through one DDP entry point."""
        target = self.encode_target(images)
        context = self.encode_context(images, masks_encoder)
        predicted, predicted_context = self.predict(
            context,
            masks_encoder,
            masks_predictor,
        )
        sigreg_embeddings = self.project_sigreg_embeddings(context)
        return target, predicted, predicted_context, sigreg_embeddings

    def project_sigreg_embeddings(self, context) -> torch.Tensor | None:
        """Pool visible tokens per image and project every supervised level."""
        if self.sigreg_projector is None:
            return None
        context_tensors = context[0]
        if not context_tensors:
            raise ValueError("SIGReg requires at least one context-mask output")

        projected = []
        expected_width = self.info.encoder_dim * self.sigreg_levels
        for features in context_tensors:
            if features.shape[-1] != expected_width:
                raise ValueError(
                    f"Expected SIGReg context width {expected_width}, got {features.shape[-1]}"
                )
            levels = features.reshape(
                features.shape[0],
                features.shape[1],
                self.sigreg_levels,
                self.info.encoder_dim,
            )
            pooled = levels.mean(dim=1).transpose(0, 1)
            projected.append(self.sigreg_projector(pooled))
        return torch.cat(projected, dim=1)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"EMA momentum must be within [0, 1], got {momentum}")
        target_parameters = list(self.target_encoder.parameters())
        online_parameters = list(self.online_encoder.parameters())
        torch._foreach_mul_(target_parameters, momentum)
        torch._foreach_add_(target_parameters, online_parameters, alpha=1.0 - momentum)

    def trainable_parameters(self):
        yield from self.online_encoder.parameters()
        yield from self.predictor.parameters()
        if self.sigreg_projector is not None:
            yield from self.sigreg_projector.parameters()


def build_flood_jepa(
    model_config: dict,
    image_size: int,
    mask_count: int,
    sigreg_config: dict | None = None,
) -> FloodJEPA:
    if model_config.get("modality_embedding", False):
        raise ValueError(
            "The initial Flood-JEPA experiment is image-only and must keep modality_embedding=false"
        )
    model_name = str(model_config.get("model_name", "vit_base"))
    if model_name not in encoder_models.__dict__:
        raise ValueError(f"Unknown official V-JEPA 2.1 encoder: {model_name}")

    patch_size = int(model_config.get("patch_size", 16))
    input_channels = int(model_config.get("input_channels", 10))
    levels = int(model_config.get("deep_self_supervision_levels", 4))
    use_rope = bool(model_config.get("use_rope", True))
    use_sdpa = bool(model_config.get("use_sdpa", True))
    use_checkpointing = bool(model_config.get("use_activation_checkpointing", True))

    encoder = encoder_models.__dict__[model_name](
        img_size=image_size,
        patch_size=patch_size,
        num_frames=1,
        tubelet_size=1,
        in_chans=input_channels,
        uniform_power=bool(model_config.get("uniform_power", True)),
        use_sdpa=use_sdpa,
        use_rope=use_rope,
        use_activation_checkpointing=use_checkpointing,
        interpolate_rope=False,
        modality_embedding=False,
        n_output_distillation=levels,
    )
    encoder_dim = int(encoder.embed_dim)

    predictor_depth = int(model_config.get("predictor_depth", 12))
    predictor_dim = int(model_config.get("predictor_dim", 384))
    predictor_heads = int(model_config.get("predictor_heads", encoder.num_heads))
    predictor = predictor_models.vit_predictor(
        img_size=image_size,
        patch_size=patch_size,
        num_frames=1,
        tubelet_size=1,
        embed_dim=encoder_dim,
        predictor_embed_dim=predictor_dim,
        depth=predictor_depth,
        num_heads=predictor_heads,
        uniform_power=bool(model_config.get("uniform_power", True)),
        use_mask_tokens=True,
        num_mask_tokens=mask_count,
        zero_init_mask_tokens=True,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_activation_checkpointing=use_checkpointing,
        return_all_tokens=True,
        modality_embedding=False,
        n_output_distillation=levels,
    )

    info = FloodJEPAInfo(
        model_name=model_name,
        input_channels=input_channels,
        image_size=image_size,
        patch_size=patch_size,
        encoder_dim=encoder_dim,
        predictor_dim=predictor_dim,
        predictor_depth=predictor_depth,
        modality_embedding=False,
    )
    sigreg_projector = None
    sigreg_config = sigreg_config or {}
    if bool(sigreg_config.get("enabled", False)):
        hidden_dim = int(sigreg_config.get("projector_hidden_dim", 2048))
        output_dim = int(sigreg_config.get("projector_output_dim", 128))
        if hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("SIGReg projector dimensions must be positive")
        sigreg_projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    return FloodJEPA(
        encoder,
        predictor,
        info,
        sigreg_projector=sigreg_projector,
        sigreg_levels=levels,
    )
