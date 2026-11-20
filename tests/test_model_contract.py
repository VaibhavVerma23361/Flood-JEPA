import torch

from flood_jepa.models.vjepa21 import build_flood_jepa
from flood_jepa.models.segmentation import (
    DPTFloodSegmenter,
    FloodSegmenter,
    LinearFloodSegmenter,
    PangaeaLiteUPerNetFloodSegmenter,
    PangaeaUPerNetFloodSegmenter,
    UPerNetFloodSegmenter,
)
from flood_jepa.training.segmentation import (
    _replace_sync_batchnorm,
    _focal_loss,
    _group_miou,
    _loss,
    _optimizer,
    _scheduler,
    _window_starts,
)
from flood_jepa.training.losses import (
    SketchedIsotropicGaussianRegularizer,
    dense_jepa_loss,
    normalize_deep_targets,
)


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

    target, predicted, predicted_context, sigreg_embeddings = model(
        images, masks_encoder, masks_predictor
    )
    assert sigreg_embeddings is None
    target = normalize_deep_targets(target, model.info.encoder_dim, levels=4)
    loss, _parts = dense_jepa_loss(
        predicted,
        predicted_context,
        target,
        masks_encoder,
        masks_predictor,
        grid_size=2,
        context_weight=0.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    context_projection = model.predictor.backbone.predictor_proj_context
    assert context_projection.weight.grad is not None
    assert context_projection.bias.grad is not None

    segmenter = FloodSegmenter(
        model.online_encoder,
        patch_size=16,
        encoder_dim=model.info.encoder_dim,
        decoder_dim=32,
    )
    logits = segmenter(images.detach())
    assert logits.shape == (2, 2, 32, 32)


def test_sigreg_projector_and_loss_backpropagate_to_online_encoder():
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
        sigreg_config={
            "enabled": True,
            "projector_hidden_dim": 32,
            "projector_output_dim": 16,
        },
    )
    images = torch.randn(2, 10, 32, 32)
    masks_encoder = [torch.tensor([[0, 1], [0, 1]])]
    masks_predictor = [torch.tensor([[2, 3], [2, 3]])]

    _target, _predicted, _context, embeddings = model(
        images, masks_encoder, masks_predictor
    )
    assert embeddings is not None
    assert embeddings.shape == (4, 2, 16)
    regularizer = SketchedIsotropicGaussianRegularizer(
        num_projections=32,
        num_knots=17,
        seed=42,
    )
    loss = regularizer(embeddings, step=7)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.sigreg_projector[0].weight.grad is not None
    assert model.online_encoder.backbone.patch_embed.proj.weight.grad is not None


def test_sigreg_penalizes_collapsed_embeddings_more_than_gaussian_samples():
    generator = torch.Generator().manual_seed(11)
    gaussian = torch.randn(4, 512, 32, generator=generator, requires_grad=True)
    collapsed = torch.zeros_like(gaussian)
    regularizer = SketchedIsotropicGaussianRegularizer(
        num_projections=128,
        num_knots=17,
        seed=3,
    )

    gaussian_loss = regularizer(gaussian, step=0)
    collapsed_loss = regularizer(collapsed, step=0)
    gaussian_loss.backward()

    assert gaussian_loss < collapsed_loss
    assert gaussian.grad is not None
    assert torch.isfinite(gaussian.grad).all()


def test_linear_probe_uses_only_final_encoder_level():
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
    probe = LinearFloodSegmenter(
        model.target_encoder,
        patch_size=16,
        encoder_dim=model.info.encoder_dim,
    )
    for parameter in probe.backbone.parameters():
        parameter.requires_grad = False
    logits = probe(torch.randn(2, 10, 32, 32))
    logits.mean().backward()
    assert logits.shape == (2, 2, 32, 32)
    assert all(parameter.grad is None for parameter in probe.backbone.parameters())
    assert probe.decoder.weight.grad is not None


def test_dpt_decoder_reassembles_four_levels_and_backpropagates():
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
    segmenter = DPTFloodSegmenter(
        model.target_encoder,
        patch_size=16,
        encoder_dim=model.info.encoder_dim,
        decoder_dim=32,
        dropout=0.1,
    )
    # The JEPA target starts frozen; the fine-tuning builder explicitly
    # re-enables it when model.freeze_encoder is false.
    for parameter in segmenter.backbone.parameters():
        parameter.requires_grad = True
    logits = segmenter(torch.randn(2, 10, 32, 32))
    logits.mean().backward()

    assert logits.shape == (2, 2, 32, 32)
    assert len(segmenter.backbone.out_layers) == 4
    assert segmenter.decoder.classifier.weight.grad is not None
    assert any(parameter.grad is not None for parameter in segmenter.backbone.parameters())


def test_upernet_decoder_trains_without_modifying_frozen_encoder():
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
    segmenter = UPerNetFloodSegmenter(
        model.target_encoder,
        patch_size=16,
        encoder_dim=model.info.encoder_dim,
        decoder_dim=32,
        dropout=0.1,
    )
    optimizer = _optimizer(
        segmenter,
        {
            "encoder_learning_rate": 0.0,
            "decoder_learning_rate": 1.0e-4,
            "weight_decay": 0.05,
        },
    )
    logits = segmenter(torch.randn(2, 10, 32, 32))
    logits.mean().backward()

    assert logits.shape == (2, 2, 32, 32)
    assert [group["name"] for group in optimizer.param_groups] == ["decoder"]
    assert all(parameter.grad is None for parameter in segmenter.backbone.parameters())
    assert segmenter.decoder.classifier.weight.grad is not None


def test_benchmark_combined_loss_includes_focal_and_ignores_padding():
    logits = torch.tensor(
        [[[[3.0, -2.0], [0.0, 9.0]], [[-3.0, 2.0], [0.0, -9.0]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[0, 1], [1, 255]]])
    focal = _focal_loss(logits, target, alpha=0.25, gamma=2.0)
    total, parts = _loss(
        logits,
        target,
        torch.tensor([1.0, 2.0]),
        {
            "ce_weight": 1.0,
            "dice_weight": 1.0,
            "focal_weight": 0.5,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
        },
    )
    total.backward()

    assert torch.isfinite(focal)
    assert torch.isclose(total.detach(), parts["ce"] + parts["dice"] + 0.5 * parts["focal"])
    assert torch.all(logits.grad[:, :, 1, 1] == 0)


def test_benchmark_focal_uses_common_alpha_and_target_class_weights():
    logits = torch.zeros((1, 2, 1, 2))
    target = torch.tensor([[[0, 1]]])
    weights = torch.tensor([1.0, 2.0])
    focal = _focal_loss(
        logits,
        target,
        class_weights=weights,
        alpha=0.25,
        gamma=2.0,
    )
    expected = 0.25 * (0.5**2) * torch.log(torch.tensor(2.0)) * 1.5
    assert torch.isclose(focal, expected)


def test_cosine_scheduler_starts_with_three_epoch_warmup():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 2.0e-5}])
    scheduler = _scheduler(
        optimizer,
        {
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "warmup_start_learning_rate": 1.0e-6,
            "minimum_learning_rate": 1.0e-6,
        },
        epochs=30,
        steps_per_epoch=10,
    )
    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(30):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] == 0.0
    assert learning_rates[1] > learning_rates[0]
    assert learning_rates[2] > learning_rates[1]
    assert learning_rates[30] == 2.0e-5


def test_benchmark_sliding_windows_and_absent_class_group_metric():
    assert _window_starts(512, 256) == [0, 256]
    assert _window_starts(512, 224) == [0, 224, 288]
    land_only = torch.tensor([[100.0, 0.0], [0.0, 0.0]])
    assert _group_miou(land_only) == 1.0


def test_local_seg_upernet_matches_pangaea_controlled_components():
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
    segmenter = PangaeaUPerNetFloodSegmenter(
        model.target_encoder,
        patch_size=16,
        encoder_dim=model.info.encoder_dim,
        decoder_dim=512,
        dropout=0.1,
    )

    assert segmenter.decoder.source_revision == "ca19b837a16797cd809225225b671473c7c93768"
    assert any(isinstance(module, torch.nn.SyncBatchNorm) for module in segmenter.decoder.modules())
    assert any(isinstance(module, torch.nn.ConvTranspose2d) for module in segmenter.decoder.modules())
    assert all(not parameter.requires_grad for parameter in segmenter.backbone.parameters())


def test_a30_vit_tiny_and_lite_decoder_parameter_contract():
    model = build_flood_jepa(
        {
            "model_name": "vit_tiny",
            "input_channels": 10,
            "patch_size": 16,
            "predictor_depth": 4,
            "predictor_dim": 192,
            "predictor_heads": 6,
            "deep_self_supervision_levels": 4,
            "use_rope": True,
            "use_sdpa": False,
            "use_activation_checkpointing": False,
            "modality_embedding": False,
        },
        image_size=32,
        mask_count=1,
    )
    assert sum(parameter.numel() for parameter in model.online_encoder.parameters()) == 5_831_616
    assert sum(parameter.numel() for parameter in model.predictor.parameters()) == 2_261_184
    assert sum(parameter.numel() for parameter in model.parameters()) == 13_924_416
    assert sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    ) == 8_092_800

    segmenter = PangaeaLiteUPerNetFloodSegmenter(
        model.target_encoder,
        patch_size=16,
        encoder_dim=192,
        decoder_dim=128,
        dropout=0.1,
    )
    assert sum(parameter.numel() for parameter in segmenter.decoder.parameters()) == 2_463_426
    assert sum(parameter.numel() for parameter in segmenter.parameters()) == 8_295_042
    segmenter.eval()
    with torch.inference_mode():
        logits = segmenter(torch.randn(1, 10, 32, 32))
    assert logits.shape == (1, 2, 32, 32)
    assert torch.isfinite(logits).all()


def test_single_gpu_seg_upernet_train_step_uses_local_batchnorm():
    model = build_flood_jepa(
        {
            "model_name": "vit_tiny",
            "input_channels": 10,
            "patch_size": 16,
            "predictor_depth": 4,
            "predictor_dim": 192,
            "predictor_heads": 6,
            "deep_self_supervision_levels": 4,
            "use_rope": True,
            "use_sdpa": False,
            "use_activation_checkpointing": False,
            "modality_embedding": False,
        },
        image_size=32,
        mask_count=1,
    )
    segmenter = PangaeaUPerNetFloodSegmenter(
        model.target_encoder,
        patch_size=16,
        encoder_dim=model.info.encoder_dim,
        decoder_dim=512,
        dropout=0.1,
    )
    _replace_sync_batchnorm(segmenter)
    assert not any(isinstance(module, torch.nn.SyncBatchNorm) for module in segmenter.modules())
    assert any(isinstance(module, torch.nn.BatchNorm2d) for module in segmenter.modules())

    segmenter.train()
    logits = segmenter(torch.randn(2, 10, 32, 32))
    assert logits.shape == (2, 2, 32, 32)
    logits.sum().backward()
    assert torch.isfinite(logits).all()

    if torch.cuda.is_available():
        cuda_segmenter = PangaeaUPerNetFloodSegmenter(
            model.target_encoder,
            patch_size=16,
            encoder_dim=model.info.encoder_dim,
            decoder_dim=512,
            dropout=0.1,
        ).cuda()
        _replace_sync_batchnorm(cuda_segmenter)
        cuda_segmenter.train()
        for parameter in cuda_segmenter.parameters():
            assert parameter.device.type == "cuda"
        cuda_logits = cuda_segmenter(torch.randn(2, 10, 32, 32, device="cuda"))
        assert cuda_logits.shape == (2, 2, 32, 32)
        cuda_logits.sum().backward()
        assert torch.isfinite(cuda_logits).all()
