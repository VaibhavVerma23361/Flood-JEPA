import torch

from flood_jepa.models.vjepa21 import build_flood_jepa
from flood_jepa.models.segmentation import FloodSegmenter
from flood_jepa.training.losses import dense_jepa_loss, normalize_deep_targets


def test_tiny_ten_band_model_contract():
    model = build_flood_jepa(
        {
            "model_name": "vit_tiny",
            "input_channels": 10,
            "patch_size": 16,
            "predictor_depth": 4,
            "predictor_dim": 96,
            "predictor_heads": 3,
            "deep_self_supervision_levels": 4,
            "use_rope": True,
            "use_sdpa": False,
            "use_activation_checkpointing": False,
            "modality_embedding": False,
        },
        image_size=32,
        mask_count=1,
    )
    assert model.info.input_channels == 10
    assert model.info.modality_embedding is False
    assert model.online_encoder.backbone.patch_embed.proj.in_channels == 10
    assert not hasattr(model.online_encoder.backbone, "img_mod_embed")


def test_tiny_pretrain_and_segmentation_forward_contract():
    model = build_flood_jepa(
        {
            "model_name": "vit_tiny",
            "input_channels": 10,
            "patch_size": 16,
            "predictor_depth": 4,
            "predictor_dim": 96,
            "predictor_heads": 3,
            "deep_self_supervision_levels": 4,
            "use_rope": True,
            "use_sdpa": False,
            "use_activation_checkpointing": False,
            "modality_embedding": False,
        },
        image_size=32,
        mask_count=1,
    )
    images = torch.randn(2, 10, 32, 32)
    masks_encoder = [torch.tensor([[0, 1], [0, 1]])]
    masks_predictor = [torch.tensor([[2, 3], [2, 3]])]

    target, predicted, predicted_context = model(
        images, masks_encoder, masks_predictor
    )
    target = normalize_deep_targets(target, model.info.encoder_dim, levels=4)
    loss, _parts = dense_jepa_loss(
        predicted,
        predicted_context,
        target,
        masks_encoder,
        masks_predictor,
        grid_size=2,
    )
    loss.backward()
    assert torch.isfinite(loss)

    segmenter = FloodSegmenter(
        model.online_encoder,
        patch_size=16,
        encoder_dim=model.info.encoder_dim,
        decoder_dim=32,
    )
    logits = segmenter(images.detach())
    assert logits.shape == (2, 2, 32, 32)
