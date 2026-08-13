"""Dense and deeply supervised V-JEPA 2.1 latent prediction loss."""

from __future__ import annotations

import torch
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
