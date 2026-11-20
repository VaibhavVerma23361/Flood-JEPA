"""Dense V-JEPA 2.1 losses and optional SIGReg regularization."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F

from flood_jepa.vendor import activate_upstream

activate_upstream()

from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance  # noqa: E402
from src.masks.utils import apply_masks  # noqa: E402


def normalize_deep_targets(target: torch.Tensor, encoder_dim: int, levels: int) -> torch.Tensor:
    expected = encoder_dim * levels
    if target.shape[-1] != expected:
        raise ValueError(f"Expected target width {expected}, got {target.shape[-1]}")
    chunks = [
        F.layer_norm(target[..., index * encoder_dim : (index + 1) * encoder_dim], (encoder_dim,))
        for index in range(levels)
    ]
    return torch.cat(chunks, dim=-1)


def _regression(prediction: torch.Tensor, target: torch.Tensor, exponent: float) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target).pow(exponent)) / exponent


class SketchedIsotropicGaussianRegularizer(nn.Module):
    """Epps--Pulley normality statistic over random one-dimensional slices.

    The input is a set of embedding samples shaped ``[levels, samples, dim]``.
    Empirical characteristic-function moments are averaged across all DDP ranks,
    so the statistic uses the global rather than per-device batch.
    """

    def __init__(
        self,
        num_projections: int = 256,
        num_knots: int = 17,
        t_max: float = 3.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if num_projections <= 0:
            raise ValueError("SIGReg num_projections must be positive")
        if num_knots < 3 or num_knots % 2 == 0:
            raise ValueError("SIGReg num_knots must be an odd integer >= 3")
        if t_max <= 0:
            raise ValueError("SIGReg t_max must be positive")

        self.num_projections = int(num_projections)
        self.seed = int(seed)
        t = torch.linspace(0.0, float(t_max), int(num_knots), dtype=torch.float32)
        dt = float(t_max) / (int(num_knots) - 1)
        quadrature = torch.full((int(num_knots),), 2.0 * dt, dtype=torch.float32)
        quadrature[[0, -1]] = dt
        normal_cf = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("normal_cf", normal_cf)
        self.register_buffer("integration_weights", quadrature * normal_cf)

    @staticmethod
    def _global_mean(value: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return value
        return dist_nn.all_reduce(value, op=dist.ReduceOp.SUM) / dist.get_world_size()

    def forward(self, embeddings: torch.Tensor, step: int = 0) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(
                "SIGReg embeddings must have shape [levels, samples, dim], "
                f"got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[1] < 2:
            raise ValueError("SIGReg requires at least two embedding samples per rank")

        # Characteristic functions and trigonometry are deliberately evaluated
        # in float32 even when the main JEPA graph uses fp16/bf16 autocast.
        embeddings = embeddings.float()
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(self.seed + int(step))
        directions = torch.randn(
            embeddings.shape[-1],
            self.num_projections,
            device=embeddings.device,
            dtype=torch.float32,
            generator=generator,
        )
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1.0e-12)

        projected = embeddings @ directions
        phases = projected.unsqueeze(-1) * self.t
        cosine_mean = self._global_mean(phases.cos().mean(dim=1))
        sine_mean = self._global_mean(phases.sin().mean(dim=1))
        error = (cosine_mean - self.normal_cf).square() + sine_mean.square()

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        global_samples = embeddings.shape[1] * world_size
        per_projection = (error * self.integration_weights).sum(dim=-1) * global_samples
        return per_projection.mean()


def dense_jepa_loss(
    predicted_nested,
    context_nested,
    target_full: torch.Tensor,
    masks_encoder: list[torch.Tensor],
    masks_predictor: list[torch.Tensor],
    grid_size: int,
    exponent: float = 1.0,
    context_weight: float = 0.5,
    distance_weighted_context: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute masked prediction plus the V-JEPA 2.1 visible-context loss."""
    predictions = predicted_nested[0]
    context_predictions = context_nested[0]
    if not (len(predictions) == len(masks_predictor) == len(masks_encoder)):
        raise ValueError("Prediction and mask configuration counts differ")

    prediction_loss = target_full.new_zeros(())
    context_loss = target_full.new_zeros(())
    distances = None
    if distance_weighted_context and context_weight > 0:
        distances = compute_mask_distance(
            [masks_predictor],
            [masks_encoder],
            grid_size,
            False,
        )[0]

    for index, (prediction, context_prediction, mask_encoder, mask_predictor) in enumerate(
        zip(predictions, context_predictions, masks_encoder, masks_predictor)
    ):
        target_predictor = apply_masks(target_full, [mask_predictor])
        prediction_loss = prediction_loss + _regression(prediction, target_predictor, exponent)

        if context_weight > 0:
            target_context = apply_masks(target_full, [mask_encoder])
            error = torch.abs(context_prediction - target_context).pow(exponent) / exponent
            if distances is not None:
                error = error / distances[index].clamp_min(1.0e-6).unsqueeze(-1)
            context_loss = context_loss + error.mean()
        else:
            # The context head is intentionally warm-started at weight zero.
            # Keep it connected to autograd so DDP receives zero (rather than
            # missing) gradients for predictor_proj_context during that period.
            context_loss = context_loss + context_prediction.sum() * 0.0

    divisor = float(len(predictions))
    prediction_loss = prediction_loss / divisor
    context_loss = context_loss / divisor
    total = prediction_loss + context_weight * context_loss
    return total, {
        "prediction": prediction_loss.detach(),
        "context": context_loss.detach(),
        "context_weight": target_full.new_tensor(context_weight),
    }
