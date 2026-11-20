"""Fine-tune and evaluate the Flood-JEPA encoder on WorldFloods segmentation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm.auto import tqdm

from flood_jepa.config import apply_overrides, load_config, require_sections
from flood_jepa.data.supervised import IGNORE_INDEX, WorldFloodsSegmentationDataset
from flood_jepa.models.segmentation import (
    DPTFloodSegmenter,
    FloodSegmenter,
    LinearFloodSegmenter,
    PangaeaLiteUPerNetFloodSegmenter,
    PangaeaUPerNetFloodSegmenter,
    UPerNetFloodSegmenter,
)
from flood_jepa.models.vjepa21 import build_flood_jepa


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return world_size, rank, local_rank


def _replace_sync_batchnorm(module: torch.nn.Module) -> torch.nn.Module:
    """Use local BatchNorm when this process is not in a distributed group.

    SegUPerNet-512 is built with SyncBatchNorm so a multi-GPU run shares
    normalization statistics across the global batch. One process has no
    process group, and SyncBatchNorm then refuses to train. On one GPU the
    local batch is the global batch, so BatchNorm2d computes the same
    statistics as synchronized normalization over that batch.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.SyncBatchNorm):
            replacement = torch.nn.BatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            # The model is already on GPU. A freshly built BatchNorm stays on
            # CPU, and its affine weight then crashes the CUDA batch-norm kernel.
            replacement = replacement.to(device=child.running_mean.device)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            replacement.running_mean.copy_(child.running_mean)
            replacement.running_var.copy_(child.running_var)
            replacement.num_batches_tracked.copy_(child.num_batches_tracked)
            replacement.train(child.training)
            setattr(module, name, replacement)
        else:
            _replace_sync_batchnorm(child)
    return module


def _seed(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _random_pretrain_config(model_cfg: dict) -> dict:
    """Matched ViT-Tiny recipe used when the encoder is drawn from scratch."""
    return {
        "model": {
            "model_name": str(model_cfg.get("expected_pretrain_model_name", "vit_tiny")),
            "input_channels": int(model_cfg.get("expected_pretrain_input_channels", 10)),
            "patch_size": int(model_cfg.get("expected_pretrain_patch_size", 16)),
            "predictor_depth": 4,
            "predictor_dim": 192,
            "predictor_heads": 6,
            "deep_self_supervision_levels": 4,
            "use_rope": True,
            "use_sdpa": True,
            "use_activation_checkpointing": False,
            "uniform_power": True,
            "modality_embedding": False,
        },
        "data": {
            "crop_size": int(model_cfg.get("expected_pretrain_crop_size", 256)),
            "input_bands": list(model_cfg.get("input_bands", [])),
        },
        "mask": [{}],
    }


def _attach_decoder(model_cfg, pretrain_config, encoder, jepa, device):
    decoder_type = str(model_cfg.get("decoder_type", "fusion"))
    common = {
        "encoder": encoder,
        "patch_size": int(pretrain_config["model"]["patch_size"]),
        "encoder_dim": int(jepa.info.encoder_dim),
        "classes": 2,
    }
    if decoder_type == "linear":
        segmenter = LinearFloodSegmenter(**common)
    elif decoder_type == "fusion":
        segmenter = FloodSegmenter(
            **common,
            decoder_dim=int(model_cfg.get("decoder_dim", 256)),
        )
    elif decoder_type == "dpt":
        segmenter = DPTFloodSegmenter(
            **common,
            decoder_dim=int(model_cfg.get("decoder_dim", 256)),
            dropout=float(model_cfg.get("decoder_dropout", 0.1)),
        )
    elif decoder_type == "upernet":
        segmenter = UPerNetFloodSegmenter(
            **common,
            decoder_dim=int(model_cfg.get("decoder_dim", 512)),
            dropout=float(model_cfg.get("decoder_dropout", 0.1)),
        )
    elif decoder_type == "seg_upernet":
        segmenter = PangaeaUPerNetFloodSegmenter(
            **common,
            decoder_dim=int(model_cfg.get("decoder_dim", 512)),
            dropout=float(model_cfg.get("decoder_dropout", 0.1)),
        )
    elif decoder_type == "seg_upernet_lite":
        segmenter = PangaeaLiteUPerNetFloodSegmenter(
            **common,
            decoder_dim=int(model_cfg.get("decoder_dim", 128)),
            dropout=float(model_cfg.get("decoder_dropout", 0.1)),
        )
    else:
        raise ValueError(
            "model.decoder_type must be 'linear', 'fusion', 'dpt', 'upernet', "
            "'seg_upernet', or 'seg_upernet_lite'"
        )
    del jepa
    feature_layers = model_cfg.get("feature_layers")
    if feature_layers is not None:
        feature_layers = [int(layer) for layer in feature_layers]
        expected_levels = 1 if decoder_type == "linear" else 4
        if len(feature_layers) != expected_levels:
            raise ValueError(
                f"model.feature_layers must contain {expected_levels} layer(s) "
                f"for decoder_type={decoder_type!r}"
            )
        segmenter.backbone.out_layers = feature_layers
    freeze_encoder = bool(model_cfg.get("freeze_encoder", False))
    for parameter in segmenter.backbone.parameters():
        parameter.requires_grad = not freeze_encoder
    return segmenter.to(device)


def _build_model(config: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    model_cfg = config["model"]
    if bool(model_cfg.get("random_init", False)):
        pretrain_config = _random_pretrain_config(model_cfg)
        jepa = build_flood_jepa(
            pretrain_config["model"],
            image_size=int(pretrain_config["data"]["crop_size"]),
            mask_count=1,
        )
        encoder = jepa.target_encoder
        encoder.backbone.use_activation_checkpointing = bool(
            model_cfg.get("use_activation_checkpointing", False)
        )
        return _attach_decoder(model_cfg, pretrain_config, encoder, jepa, device), pretrain_config
    pretrain_path = Path(model_cfg["pretrained_checkpoint"]).expanduser().resolve()
    if not pretrain_path.is_file():
        raise FileNotFoundError(
            f"Missing JEPA pretraining checkpoint: {pretrain_path}. "
            "Set FLOOD_JEPA_PRETRAINED or update the fine-tuning config."
        )
    # These are complete, locally generated training checkpoints containing
    # optimizer/config metadata, not tensor-only weight files. PyTorch 2.6+
    # defaults to weights_only=True, so trusted run checkpoints must opt out.
    checkpoint = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    pretrain_config = checkpoint["config"]
    expected_contract = {
        ("model", "model_name"): model_cfg.get("expected_pretrain_model_name"),
        ("model", "input_channels"): model_cfg.get("expected_pretrain_input_channels"),
        ("model", "patch_size"): model_cfg.get("expected_pretrain_patch_size"),
        ("data", "crop_size"): model_cfg.get("expected_pretrain_crop_size"),
    }
    for (section, key), expected in expected_contract.items():
        if expected is None:
            continue
        actual = pretrain_config.get(section, {}).get(key)
        if actual != expected:
            raise ValueError(
                f"Pretraining checkpoint contract mismatch for {section}.{key}: "
                f"expected {expected!r}, got {actual!r}"
            )
    source = str(model_cfg.get("encoder_source", "target"))
    if source not in {"target", "online"}:
        raise ValueError("model.encoder_source must be 'target' or 'online'")

    jepa = build_flood_jepa(
        pretrain_config["model"],
        image_size=int(pretrain_config["data"]["crop_size"]),
        mask_count=len(pretrain_config["mask"]),
    )
    encoder = jepa.target_encoder if source == "target" else jepa.online_encoder
    encoder.load_state_dict(checkpoint[f"{source}_encoder"], strict=True)
    encoder.backbone.use_activation_checkpointing = bool(
        model_cfg.get("use_activation_checkpointing", True)
    )
    return _attach_decoder(model_cfg, pretrain_config, encoder, jepa, device), pretrain_config


def _dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, num_classes=2).permute(0, 3, 1, 2)
    # Accumulate dense pixel losses in FP32 even when the model runs under
    # FP16 autocast; a 256x256 reduction can overflow half precision.
    one_hot = one_hot.to(dtype=torch.float32)
    valid_float = valid.unsqueeze(1).to(dtype=torch.float32)
    probabilities = logits.float().softmax(dim=1) * valid_float
    one_hot = one_hot * valid_float
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    target_pixels = one_hot.sum(dim=(0, 2, 3))
    present = target_pixels > 0
    dice = (2.0 * intersection[present] + 1.0e-6) / (
        denominator[present] + 1.0e-6
    )
    class_losses = 1.0 - dice
    if class_weights is None:
        return class_losses.mean()
    weights = class_weights.float()[present]
    return (class_losses * weights).sum() / weights.sum().clamp_min(1.0e-12)


def _focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Softmax focal loss with a common alpha and per-target class weights."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("loss.focal_alpha must be within [0, 1]")
    if gamma < 0.0:
        raise ValueError("loss.focal_gamma must be non-negative")
    valid = target != IGNORE_INDEX
    safe_target = target.masked_fill(~valid, 0)
    log_probabilities = F.log_softmax(logits.float(), dim=1)
    log_pt = log_probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    # The executable benchmark uses alpha as one common scalar, then applies
    # the stored target-class weight independently.
    target_weights = torch.ones_like(pt)
    if class_weights is not None:
        target_weights = class_weights.float()[safe_target]
    focal = -float(alpha) * target_weights * (1.0 - pt).pow(gamma) * log_pt
    return focal[valid].mean()


def _loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None,
    loss_cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not torch.any(target != IGNORE_INDEX):
        raise RuntimeError("Batch contains no supervised WorldFloods pixels")
    ce = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
    )
    dice = _dice_loss(logits, target, class_weights)
    focal = _focal_loss(
        logits,
        target,
        class_weights=class_weights,
        alpha=float(loss_cfg.get("focal_alpha", 0.25)),
        gamma=float(loss_cfg.get("focal_gamma", 2.0)),
    )
    total = (
        float(loss_cfg.get("ce_weight", 0.5)) * ce
        + float(loss_cfg.get("dice_weight", 0.5)) * dice
        + float(loss_cfg.get("focal_weight", 0.0)) * focal
    )
    return total, {
        "ce": ce.detach(),
        "dice": dice.detach(),
        "focal": focal.detach(),
    }


def _confusion(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = logits.argmax(dim=1)
    valid = target != IGNORE_INDEX
    encoded = target[valid] * 2 + prediction[valid]
    return torch.bincount(encoded, minlength=4).reshape(2, 2).to(torch.float64)


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices without padding or duplicating any tile."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.world_size + 1


def _window_starts(length: int, crop_size: int) -> list[int]:
    if crop_size > length:
        raise ValueError(f"Evaluation crop {crop_size} exceeds tile dimension {length}")
    starts = list(range(0, length - crop_size + 1, crop_size))
    edge = length - crop_size
    if not starts or starts[-1] != edge:
        starts.append(edge)
    return starts


def _sliding_window_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    crop_size: int,
    window_batch_size: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    mixed_precision: bool,
) -> torch.Tensor:
    """Reconstruct one full tile by averaging float32 overlapping logits."""
    if image.shape[0] != 1:
        raise ValueError("Full-tile evaluation expects a tile-loader batch size of one")
    height, width = image.shape[-2:]
    locations = [
        (top, left)
        for top in _window_starts(height, crop_size)
        for left in _window_starts(width, crop_size)
    ]
    accumulated = torch.zeros((1, 2, height, width), dtype=torch.float32, device=device)
    coverage = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    for offset in range(0, len(locations), window_batch_size):
        group = locations[offset : offset + window_batch_size]
        windows = torch.cat(
            [
                image[:, :, top : top + crop_size, left : left + crop_size]
                for top, left in group
            ],
            dim=0,
        ).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=mixed_precision,
        ):
            window_logits = model(windows)
        window_logits = window_logits.float()
        for index, (top, left) in enumerate(group):
            accumulated[
                :,
                :,
                top : top + crop_size,
                left : left + crop_size,
            ] += window_logits[index : index + 1]
            coverage[
                :,
                :,
                top : top + crop_size,
                left : left + crop_size,
            ] += 1.0
    if torch.any(coverage == 0):
        raise RuntimeError("Sliding-window evaluation left uncovered tile pixels")
    return accumulated / coverage


def _metrics(confusion: torch.Tensor) -> dict[str, float]:
    true_positive = confusion.diag()
    actual = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = actual + predicted - true_positive
    iou = true_positive / union.clamp_min(1.0)
    per_class_precision = true_positive / predicted.clamp_min(1.0)
    per_class_recall = true_positive / actual.clamp_min(1.0)
    per_class_f1 = (
        2.0
        * per_class_precision
        * per_class_recall
        / (per_class_precision + per_class_recall).clamp_min(1.0e-12)
    )
    total = confusion.sum().clamp_min(1.0)
    global_miou = float(iou.mean())
    return {
        "global_miou": global_miou,
        "miou": global_miou,
        "global_iou_land": float(iou[0]),
        "global_iou_water": float(iou[1]),
        "land_iou": float(iou[0]),
        "water_iou": float(iou[1]),
        "macro_f1": float(per_class_f1.mean()),
        "precision_water": float(per_class_precision[1]),
        "recall_water": float(per_class_recall[1]),
        "f1_water": float(per_class_f1[1]),
        "water_precision": float(per_class_precision[1]),
        "water_recall": float(per_class_recall[1]),
        "water_f1": float(per_class_f1[1]),
        "overall_accuracy": float(true_positive.sum() / total),
        "valid_pixels": int(confusion.sum().item()),
    }


def _group_miou(confusion: torch.Tensor) -> float:
    true_positive = confusion.diag()
    union = confusion.sum(dim=1) + confusion.sum(dim=0) - true_positive
    present = union > 0
    if not torch.any(present):
        return float("nan")
    return float((true_positive[present] / union[present]).mean())


def _grouped_metric_summary(records: list[dict]) -> dict[str, float | int]:
    def summarize(groups: dict[str, torch.Tensor], prefix: str) -> dict[str, float]:
        values = np.asarray(
            [_group_miou(confusion) for confusion in groups.values()],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        return {
            f"{prefix}_miou_mean": float(values.mean()) if values.size else float("nan"),
            f"{prefix}_miou_std": float(values.std(ddof=0)) if values.size else float("nan"),
        }

    patch_values = np.asarray(
        [float(record["patch_miou"]) for record in records],
        dtype=np.float64,
    )
    patch_values = patch_values[np.isfinite(patch_values)]
    source_groups: dict[str, torch.Tensor] = {}
    event_groups: dict[str, torch.Tensor] = {}
    for record in records:
        confusion = torch.tensor(
            [
                [record["target_land_pred_land"], record["target_land_pred_water"]],
                [record["target_water_pred_land"], record["target_water_pred_water"]],
            ],
            dtype=torch.float64,
        )
        for groups, key in (
            (source_groups, str(record["source_id"])),
            (event_groups, str(record["event_id"])),
        ):
            groups[key] = groups.get(key, torch.zeros((2, 2), dtype=torch.float64)) + confusion
    result: dict[str, float | int] = {
        "patch_miou_mean": float(patch_values.mean()) if patch_values.size else float("nan"),
        "patch_miou_std": float(patch_values.std(ddof=0)) if patch_values.size else float("nan"),
        "samples": len(records),
    }
    result.update(summarize(source_groups, "source"))
    result.update(summarize(event_groups, "event"))
    return result


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    mixed_precision: bool,
    class_weights: torch.Tensor | None = None,
    loss_cfg: dict | None = None,
    per_sample_records: list[dict] | None = None,
    crop_size: int | None = None,
    window_batch_size: int = 8,
    progress_description: str = "Evaluation",
) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros((2, 2), dtype=torch.float64, device=device)
    loss_totals = torch.zeros(5, dtype=torch.float64, device=device)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    total_tiles = len(loader.dataset)
    progress = (
        tqdm(
            total=total_tiles,
            desc=progress_description,
            dynamic_ncols=True,
            unit="tile",
        )
        if rank == 0
        else None
    )
    local_tiles_processed = 0
    displayed_global_tiles = 0
    for batch in loader:
        images = batch["image"]
        target = batch["label"].to(device, non_blocking=True)
        if crop_size is None:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=mixed_precision,
            ):
                logits = model(images)
        else:
            logits = _sliding_window_logits(
                model,
                images,
                crop_size,
                window_batch_size,
                device,
                autocast_dtype,
                mixed_precision,
            )
        if loss_cfg is not None:
            loss, parts = _loss(logits, target, class_weights, loss_cfg)
        confusion += _confusion(logits, target)
        if per_sample_records is not None:
            for sample_index in range(images.shape[0]):
                sample_confusion = _confusion(
                    logits[sample_index : sample_index + 1],
                    target[sample_index : sample_index + 1],
                ).cpu()
                sample_metrics = _metrics(sample_confusion)
                per_sample_records.append(
                    {
                        "sample_id": str(batch["sample_id"][sample_index]),
                        "source_id": str(batch["source_id"][sample_index]),
                        "event_id": str(batch["event_id"][sample_index]),
                        "patch_miou": _group_miou(sample_confusion),
                        "target_land_pred_land": int(sample_confusion[0, 0]),
                        "target_land_pred_water": int(sample_confusion[0, 1]),
                        "target_water_pred_land": int(sample_confusion[1, 0]),
                        "target_water_pred_water": int(sample_confusion[1, 1]),
                        **sample_metrics,
                    }
                )
        if loss_cfg is not None:
            batch_size = target.shape[0]
            loss_totals[0] += loss.detach().double() * batch_size
            loss_totals[1] += parts["ce"].double() * batch_size
            loss_totals[2] += parts["dice"].double() * batch_size
            loss_totals[3] += parts["focal"].double() * batch_size
            loss_totals[4] += batch_size
        if progress is not None:
            local_tiles_processed += int(target.shape[0])
            estimated_global_tiles = min(
                local_tiles_processed * world_size,
                total_tiles,
            )
            progress.update(estimated_global_tiles - displayed_global_tiles)
            displayed_global_tiles = estimated_global_tiles
    if progress is not None:
        progress.close()
    if distributed:
        dist.all_reduce(confusion)
        dist.all_reduce(loss_totals)
    metrics = _metrics(confusion.cpu())
    if loss_cfg is not None:
        denominator = loss_totals[4].clamp_min(1)
        metrics.update(
            {
                "loss": float(loss_totals[0] / denominator),
                "ce_loss": float(loss_totals[1] / denominator),
                "dice_loss": float(loss_totals[2] / denominator),
                "focal_loss": float(loss_totals[3] / denominator),
            }
        )
    return metrics


def _dataset(config: dict, split: str) -> WorldFloodsSegmentationDataset:
    data_cfg = config["data"]
    augment_d4 = split == "train" and bool(data_cfg.get("augment_d4", True))
    allow_source_ids = None
    if split == "train" and data_cfg.get("train_source_manifest"):
        manifest = json.loads(
            Path(data_cfg["train_source_manifest"]).expanduser().read_text(encoding="utf-8")
        )
        allow_source_ids = [str(source_id) for source_id in manifest["source_ids"]]
    dataset = WorldFloodsSegmentationDataset(
        lmdb_path=data_cfg[f"{split}_lmdb"],
        split=split,
        crop_size=int(data_cfg["crop_size"]),
        input_bands=list(data_cfg["input_bands"]),
        normalization=str(data_cfg.get("normalization", "worldfloods")),
        augment_d4=augment_d4,
        d4_probability=(
            float(data_cfg.get("d4_probability", 1.0 if augment_d4 else 0.0))
            if split == "train"
            else 0.0
        ),
        rotate_probability=(
            float(data_cfg.get("rotate_probability", 0.0)) if split == "train" else 0.0
        ),
        rotate_limit_degrees=float(data_cfg.get("rotate_limit_degrees", 0.0)),
        crop_attempts=int(data_cfg.get("crop_attempts", 10)),
        minimum_supervised_fraction=float(
            data_cfg.get("minimum_supervised_fraction", 0.25)
        ),
        sample_retry_attempts=int(data_cfg.get("sample_retry_attempts", 20)),
        allow_source_ids=allow_source_ids,
    )
    expected_tile_size = data_cfg.get("source_tile_size")
    if expected_tile_size is not None and dataset.tile_size != int(expected_tile_size):
        raise ValueError(
            f"WorldFloods source tile mismatch: expected {expected_tile_size}, "
            f"LMDB declares {dataset.tile_size}"
        )
    return dataset


def _validate_dataset_splits(
    datasets: dict[str, WorldFloodsSegmentationDataset],
    expected_counts: dict,
) -> None:
    """Fail before training if the benchmark split identity is not exact."""
    for split, dataset in datasets.items():
        expected = expected_counts.get(split)
        if expected is not None and len(dataset) != int(expected):
            raise RuntimeError(
                f"WorldFloods {split} count mismatch: expected {expected}, got {len(dataset)}"
            )
        sample_ids = [str(value) for value in dataset.sample_ids]
        if len(sample_ids) != len(set(sample_ids)):
            raise RuntimeError(f"WorldFloods {split} contains duplicate sample IDs")

    split_names = list(datasets)
    for left_index, left_name in enumerate(split_names):
        left = datasets[left_name]
        for right_name in split_names[left_index + 1 :]:
            right = datasets[right_name]
            sample_overlap = set(map(str, left.sample_ids)) & set(map(str, right.sample_ids))
            if sample_overlap:
                raise RuntimeError(
                    f"WorldFloods {left_name}/{right_name} sample IDs overlap "
                    f"({len(sample_overlap)} IDs)"
                )
            source_overlap = set(map(str, left.source_ids)) & set(map(str, right.source_ids))
            if source_overlap:
                raise RuntimeError(
                    f"WorldFloods {left_name}/{right_name} source-scene IDs overlap "
                    f"({len(source_overlap)} IDs)"
                )


def _loader(
    dataset,
    training_cfg: dict,
    sampler=None,
    shuffle=False,
    batch_size: int | None = None,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader:
    workers = int(training_cfg.get("num_workers", 8))
    return DataLoader(
        dataset,
        batch_size=(
            int(batch_size)
            if batch_size is not None
            else int(training_cfg.get("per_device_batch_size", 8))
        ),
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers,
        pin_memory=bool(training_cfg.get("pin_memory", True)),
        persistent_workers=bool(training_cfg.get("persistent_workers", True)) and workers > 0,
        drop_last=drop_last,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def _evaluation_batch_size(config: dict) -> int:
    return int(
        config.get("evaluation", {}).get(
            "sliding_window_batch_size",
            config["optimization"].get(
                "eval_per_device_batch_size",
                config["optimization"].get("per_device_batch_size", 8),
            ),
        )
    )


def _optimizer(model: torch.nn.Module, opt_cfg: dict) -> torch.optim.Optimizer:
    """Build named parameter groups without an empty frozen-encoder group."""
    groups = []
    encoder_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    if encoder_parameters:
        groups.append(
            {
                "name": "encoder",
                "params": encoder_parameters,
                "lr": float(opt_cfg.get("encoder_learning_rate", 1.0e-5)),
            }
        )
    decoder_parameters = [
        parameter for parameter in model.decoder.parameters() if parameter.requires_grad
    ]
    if not decoder_parameters:
        raise RuntimeError("Segmentation decoder has no trainable parameters")
    groups.append(
        {
            "name": "decoder",
            "params": decoder_parameters,
            "lr": float(opt_cfg.get("decoder_learning_rate", 1.0e-4)),
        }
    )
    return torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("epsilon", 1.0e-8)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    opt_cfg: dict,
    epochs: int,
    steps_per_epoch: int,
):
    scheduler_name = str(opt_cfg.get("scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        minimum_lr = float(opt_cfg.get("minimum_learning_rate", 1.0e-6))
        warmup_epochs = int(opt_cfg.get("warmup_epochs", 0))
        if warmup_epochs < 0 or warmup_epochs >= epochs:
            raise ValueError("optimization.warmup_epochs must be in [0, epochs)")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if any(minimum_lr > base_lr for base_lr in base_lrs):
            raise ValueError("minimum_learning_rate cannot exceed a parameter-group LR")
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        def schedule_for(base_lr: float):
            minimum_ratio = minimum_lr / base_lr

            def schedule(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(
                    1,
                    total_steps - warmup_steps - 1,
                )
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                return minimum_ratio + (1.0 - minimum_ratio) * cosine

            return schedule

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[schedule_for(base_lr) for base_lr in base_lrs],
        )
    if scheduler_name == "multistep":
        milestones = [int(value) for value in opt_cfg.get("lr_milestones", [])]
        if not milestones or any(value <= 0 or value >= epochs for value in milestones):
            raise ValueError(
                "optimization.lr_milestones must be non-empty and within the training horizon"
            )
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(opt_cfg.get("lr_gamma", 0.1)),
        )
    raise ValueError("optimization.scheduler must be 'cosine' or 'multistep'")


def _learning_rate(optimizer: torch.optim.Optimizer, group_name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group["lr"])
    return 0.0


def _clip_gradients(model: torch.nn.Module, gradient_clip_norm) -> None:
    if gradient_clip_norm is None:
        return
    maximum_norm = float(gradient_clip_norm)
    if maximum_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        maximum_norm,
    )


def _validate_checkpoint_compatibility(config: dict, checkpoint: dict) -> None:
    """Reject evaluation with a config that changes trained model semantics."""
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Segmentation checkpoint has no resolved training config")
    critical_paths = (
        ("data", "crop_size"),
        ("data", "input_bands"),
        ("data", "normalization"),
        ("data", "crop_attempts"),
        ("data", "minimum_supervised_fraction"),
        ("data", "augment_d4"),
        ("data", "d4_probability"),
        ("data", "rotate_probability"),
        ("data", "rotate_limit_degrees"),
        ("model", "encoder_source"),
        ("model", "decoder_type"),
        ("model", "decoder_dim"),
        ("model", "decoder_dropout"),
        ("model", "feature_layers"),
        ("model", "pangaea_seg_upernet_revision"),
        ("loss", "use_class_weights"),
        ("loss", "ce_weight"),
        ("loss", "dice_weight"),
        ("loss", "focal_weight"),
        ("loss", "focal_alpha"),
        ("loss", "focal_gamma"),
        ("loss", "ignore_index"),
        ("optimization", "decoder_learning_rate"),
        ("optimization", "weight_decay"),
        ("optimization", "scheduler"),
        ("optimization", "warmup_epochs"),
        ("optimization", "minimum_learning_rate"),
        ("optimization", "gradient_clip_norm"),
        ("optimization", "dtype"),
        ("optimization", "per_device_batch_size"),
        ("distributed", "required_world_size"),
        ("distributed", "global_batch_size"),
    )
    for section, key in critical_paths:
        current = config.get(section, {}).get(key)
        saved = saved_config.get(section, {}).get(key)
        if current != saved:
            raise ValueError(
                f"Evaluation config mismatch for {section}.{key}: "
                f"checkpoint used {saved!r}, current config uses {current!r}"
            )


def _validate_pretrain_identity(loaded: dict, saved: dict) -> None:
    """Ensure the external epoch checkpoint matches the segmenter provenance."""
    critical_paths = (
        ("data", "crop_size"),
        ("data", "input_bands"),
        ("model", "model_name"),
        ("model", "input_channels"),
        ("model", "patch_size"),
    )
    for section, key in critical_paths:
        current = loaded.get(section, {}).get(key)
        expected = saved.get(section, {}).get(key)
        if current != expected:
            raise ValueError(
                f"Pretraining provenance mismatch for {section}.{key}: "
                f"segmenter used {expected!r}, supplied checkpoint uses {current!r}"
            )


def _save(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_miou: float,
    best_epoch: int,
    config: dict,
    pretrain_config: dict,
) -> None:
    unwrapped = model.module if hasattr(model, "module") else model
    payload = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "seed": int(config["experiment"].get("seed", 42)),
        "config": config,
        "pretrain_config": pretrain_config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


def _write_provenance(
    config_dir: Path,
    config: dict,
    model: torch.nn.Module,
    train_dataset: WorldFloodsSegmentationDataset,
    val_dataset: WorldFloodsSegmentationDataset,
    test_dataset: WorldFloodsSegmentationDataset,
) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }
    (config_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n",
        encoding="utf-8",
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    (config_dir / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    parameters = {
        **_parameter_counts(model),
        "encoder_frozen": bool(config["model"].get("freeze_encoder", False)),
        "decoder": str(config["model"].get("decoder_type")),
        "decoder_channels": int(config["model"].get("decoder_dim", 512)),
    }
    (config_dir / "parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n",
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[3]
    upstream_revision_path = project_root / "UPSTREAM_VJEPA2_COMMIT"
    vjepa_revision = (
        upstream_revision_path.read_text(encoding="utf-8").strip()
        if upstream_revision_path.is_file()
        else "unknown"
    )
    (config_dir / "THIRD_PARTY_COMMITS.tsv").write_text(
        "component\trevision\n"
        f"facebookresearch/vjepa2\t{vjepa_revision}\n"
        "VMarsocci/pangaea-bench\tca19b837a16797cd809225225b671473c7c93768\n",
        encoding="utf-8",
    )
    (config_dir / "MODEL_SOURCES.tsv").write_text(
        "component\tsource\n"
        "Flood-JEPA\tlocal repository\n"
        "SegUPerNet\thttps://github.com/VMarsocci/pangaea-bench\n"
        "V-JEPA2\thttps://github.com/facebookresearch/vjepa2\n",
        encoding="utf-8",
    )
    manifest_source = Path(train_dataset.lmdb_path).parent / "worldfloods_manifest.json"
    manifest_destination = config_dir / "worldfloods_manifest.json"
    if manifest_source.is_file():
        manifest_destination.write_bytes(manifest_source.read_bytes())
    else:
        manifest_destination.write_text(
            json.dumps(
                {
                    "train": {"path": train_dataset.lmdb_path, "tiles": len(train_dataset)},
                    "validation": {"path": val_dataset.lmdb_path, "tiles": len(val_dataset)},
                    "test": {"path": test_dataset.lmdb_path, "tiles": len(test_dataset)},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _write_checksums(run_dir: Path) -> None:
    checksum_path = run_dir / "config" / "SHA256SUMS.generated"
    lines = []
    for path in sorted(run_dir.rglob("*")):
        if (
            not path.is_file()
            or path == checksum_path
            or path.suffix in {".tmp", ".log"}
        ):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(run_dir)}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_benchmark_summary(run_dir: Path, result: dict, config: dict) -> None:
    model_name = str(config["experiment"]["name"])
    row = {
        "model": model_name,
        "seed": int(config["experiment"].get("seed", 42)),
        "status": "complete",
        "global_miou": result["global_miou"],
        "global_iou_land": result["global_iou_land"],
        "global_iou_water": result["global_iou_water"],
        "overall_accuracy": result["overall_accuracy"],
        "macro_f1": result["macro_f1"],
        "precision_water": result["precision_water"],
        "recall_water": result["recall_water"],
        "f1_water": result["f1_water"],
        "patch_miou_mean": result["patch_miou_mean"],
        "patch_miou_std": result["patch_miou_std"],
        "source_miou_mean": result["source_miou_mean"],
        "source_miou_std": result["source_miou_std"],
        "event_miou_mean": result["event_miou_mean"],
        "event_miou_std": result["event_miou_std"],
        "samples": result["samples"],
        "selected_epoch": result["selected_epoch"],
        "elapsed_seconds": result["elapsed_seconds"],
        "tiles_per_second": result["tiles_per_second"],
        "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
        "peak_gpu_memory_gib": result["peak_gpu_memory_gib"],
    }
    results_root = run_dir.parent.parent
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "worldfloods_benchmark_summary.csv"
    rows = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    rows = [existing for existing in rows if existing.get("model") != model_name]
    rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    (results_root / "worldfloods_benchmark_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    status_path = results_root / "run_status.tsv"
    with status_path.open("w", encoding="utf-8") as stream:
        stream.write("model\tseed\tstatus\trun_directory\n")
        stream.write(f"{model_name}\t{row['seed']}\tcomplete\t{run_dir}\n")


def train(config: dict) -> None:
    require_sections(config, "experiment", "data", "model", "loss", "optimization")
    world_size, rank, local_rank = _distributed()
    distributed_cfg = config.get("distributed", {})
    required_world_size = distributed_cfg.get("required_world_size")
    if required_world_size is not None and world_size != int(required_world_size):
        raise RuntimeError(
            f"Benchmark requires {required_world_size} distributed workers, got {world_size}"
        )
    per_device_batch = int(config["optimization"].get("per_device_batch_size", 8))
    expected_global_batch = distributed_cfg.get("global_batch_size")
    actual_global_batch = world_size * per_device_batch
    if expected_global_batch is not None and actual_global_batch != int(expected_global_batch):
        raise RuntimeError(
            f"Benchmark global batch must be {expected_global_batch}, got "
            f"{world_size} x {per_device_batch} = {actual_global_batch}"
        )
    if int(config["loss"].get("ignore_index", IGNORE_INDEX)) != IGNORE_INDEX:
        raise ValueError(f"WorldFloods ignore_index must be {IGNORE_INDEX}")
    seed = int(config["experiment"].get("seed", 42))
    _seed(seed, rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    output_dir = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    config_dir = output_dir / "config"
    metrics_dir = output_dir / "metrics"
    logs_dir = output_dir / "logs"
    if rank == 0:
        for directory in (checkpoint_dir, config_dir, metrics_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    train_dataset = _dataset(config, "train")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    train_loader = _loader(
        train_dataset,
        config["optimization"],
        sampler=train_sampler,
        drop_last=True,
        seed=seed + rank,
    )
    val_dataset = _dataset(config, "val")
    test_dataset = _dataset(config, "test")
    expected_counts = config["data"].get("expected_tile_counts", {})
    _validate_dataset_splits(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset},
        expected_counts,
    )
    val_sampler = (
        DistributedEvalSampler(val_dataset, rank=rank, world_size=world_size)
        if world_size > 1
        else None
    )
    val_loader = _loader(
        val_dataset,
        config["optimization"],
        sampler=val_sampler,
        shuffle=False,
        batch_size=1,
        seed=seed + rank,
    )

    model, pretrain_config = _build_model(config, device)
    if world_size == 1:
        _replace_sync_batchnorm(model)
    if list(config["data"]["input_bands"]) != list(pretrain_config["data"]["input_bands"]):
        raise ValueError("Fine-tuning bands/order must exactly match JEPA pretraining")
    opt_cfg = config["optimization"]
    if rank == 0:
        _write_provenance(
            config_dir,
            config,
            model,
            train_dataset,
            val_dataset,
            test_dataset,
        )
    optimizer = _optimizer(model, opt_cfg)
    epochs = int(opt_cfg.get("epochs", 30))
    configured_steps = opt_cfg.get("max_steps_per_epoch")
    steps_per_epoch = len(train_loader)
    if configured_steps is not None:
        steps_per_epoch = min(steps_per_epoch, int(configured_steps))
        if steps_per_epoch <= 0:
            raise ValueError("optimization.max_steps_per_epoch must be positive or null")
    scheduler = _scheduler(optimizer, opt_cfg, epochs, steps_per_epoch)
    start_epoch = 0
    best_miou = -1.0
    best_epoch = 0
    resumed_scaler_state = None
    checkpoint_cfg = config.get("checkpoint", {})
    resume = checkpoint_cfg.get("resume")
    if not resume and bool(checkpoint_cfg.get("auto_resume", False)):
        automatic_checkpoint = checkpoint_dir / "last.pt"
        if automatic_checkpoint.is_file():
            resume = str(automatic_checkpoint)
    if resume:
        resumed = torch.load(
            Path(resume).expanduser(),
            map_location="cpu",
            weights_only=False,
        )
        _validate_checkpoint_compatibility(config, resumed)
        resumed_pretrain_config = resumed.get("pretrain_config")
        if not isinstance(resumed_pretrain_config, dict):
            raise ValueError("Resume checkpoint has no pretraining provenance")
        _validate_pretrain_identity(pretrain_config, resumed_pretrain_config)
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        resumed_scaler_state = resumed.get("scaler")
        start_epoch = int(resumed["epoch"]) + 1
        best_miou = float(resumed["best_miou"])
        best_epoch = int(resumed.get("best_epoch", start_epoch))

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    unwrapped = model.module if hasattr(model, "module") else model
    class_weights = None
    if bool(config["loss"].get("use_class_weights", True)):
        class_weights = torch.as_tensor(train_dataset.class_weights, device=device)
    dtype_name = str(opt_cfg.get("dtype", "float16")).lower()
    autocast_dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    mixed_precision = device.type == "cuda" and dtype_name in {"float16", "bfloat16"}
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and dtype_name == "float16")
    if resumed_scaler_state is not None:
        scaler.load_state_dict(resumed_scaler_state)
    log_path = logs_dir / "training.jsonl"
    validation_csv_path = metrics_dir / "training_history.csv"
    if rank == 0 and start_epoch > 0:
        if log_path.is_file():
            retained = [
                line
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line and int(json.loads(line)["epoch"]) <= start_epoch
            ]
            log_path.write_text(
                ("\n".join(retained) + "\n") if retained else "",
                encoding="utf-8",
            )
        if validation_csv_path.is_file():
            with validation_csv_path.open(newline="", encoding="utf-8") as stream:
                retained_rows = [
                    row
                    for row in csv.DictReader(stream)
                    if int(row["epoch"]) <= start_epoch
                ]
            if retained_rows:
                with validation_csv_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(retained_rows[0]))
                    writer.writeheader()
                    writer.writerows(retained_rows)
            else:
                validation_csv_path.unlink()
    training_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        if bool(config["model"].get("freeze_encoder", False)):
            # Preserve deterministic frozen features even while the decoder is
            # in training mode.
            unwrapped.backbone.eval()
        epoch_loss = torch.zeros(5, dtype=torch.float64, device=device)
        epoch_start = time.time()
        progress = tqdm(
            islice(train_loader, steps_per_epoch),
            total=steps_per_epoch,
            desc=(
                f"Decoder epoch {epoch + 1}/{epochs}"
                if bool(config["model"].get("freeze_encoder", False))
                else f"Fine-tune epoch {epoch + 1}/{epochs}"
            ),
            dynamic_ncols=True,
            disable=rank != 0,
            unit="step",
        )
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=mixed_precision,
            ):
                logits = model(images)
                loss, parts = _loss(logits, target, class_weights, config["loss"])
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                _clip_gradients(unwrapped, opt_cfg.get("gradient_clip_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                _clip_gradients(unwrapped, opt_cfg.get("gradient_clip_norm", 1.0))
                optimizer.step()
            scheduler.step()
            epoch_loss[0] += loss.detach().double()
            epoch_loss[1] += parts["ce"].double()
            epoch_loss[2] += parts["dice"].double()
            epoch_loss[3] += parts["focal"].double()
            epoch_loss[4] += 1
            if rank == 0:
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if world_size > 1:
            dist.all_reduce(epoch_loss)
            dist.barrier()

        validation_frequency = int(opt_cfg.get("validation_every_epochs", 1))
        if validation_frequency <= 0:
            raise ValueError("optimization.validation_every_epochs must be positive")
        should_validate = (epoch + 1) % validation_frequency == 0 or epoch + 1 == epochs
        metric_value = torch.tensor(-1.0, device=device)
        validation = None
        if should_validate:
            distributed_validation = evaluate(
                unwrapped,
                val_loader,
                device,
                autocast_dtype,
                mixed_precision,
                class_weights=class_weights,
                loss_cfg=config["loss"],
                crop_size=int(config["data"]["crop_size"]),
                window_batch_size=_evaluation_batch_size(config),
                progress_description=f"Validation epoch {epoch + 1}/{epochs}",
            )
            validation = distributed_validation if rank == 0 else None
            primary_metric = str(
                config.get("evaluation", {}).get("primary_metric", "global_miou")
            )
            if primary_metric not in distributed_validation:
                raise ValueError(f"Unknown evaluation.primary_metric: {primary_metric}")
            metric_value.fill_(distributed_validation[primary_metric])
        current_miou = float(metric_value)

        if rank == 0:
            record = {
                "epoch": epoch + 1,
                "train_loss": float(epoch_loss[0] / epoch_loss[4].clamp_min(1)),
                "train_ce_loss": float(epoch_loss[1] / epoch_loss[4].clamp_min(1)),
                "train_dice_loss": float(epoch_loss[2] / epoch_loss[4].clamp_min(1)),
                "train_focal_loss": float(epoch_loss[3] / epoch_loss[4].clamp_min(1)),
                "elapsed_seconds": time.time() - epoch_start,
                "validation": validation,
                "encoder_learning_rate": _learning_rate(optimizer, "encoder"),
                "decoder_learning_rate": _learning_rate(optimizer, "decoder"),
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            if validation is not None:
                validation_record = {
                    "epoch": epoch + 1,
                    "train_loss": record["train_loss"],
                    "train_ce_loss": record["train_ce_loss"],
                    "train_dice_loss": record["train_dice_loss"],
                    "train_focal_loss": record["train_focal_loss"],
                    "decoder_learning_rate": record["decoder_learning_rate"],
                    "epoch_elapsed_seconds": record["elapsed_seconds"],
                    **validation,
                }
                write_header = not validation_csv_path.is_file()
                with validation_csv_path.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=list(validation_record),
                    )
                    if write_header:
                        writer.writeheader()
                    writer.writerow(validation_record)
            print(record, flush=True)
            if validation is not None and current_miou > best_miou:
                best_miou = current_miou
                best_epoch = epoch + 1
                _save(
                    checkpoint_dir / "best.pt",
                    unwrapped,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_miou,
                    best_epoch,
                    config,
                    pretrain_config,
                )
                (metrics_dir / "best_validation.json").write_text(
                    json.dumps(
                        {"selected_epoch": best_epoch, **validation},
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            _save(
                checkpoint_dir / "last.pt",
                unwrapped,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_miou,
                best_epoch,
                config,
                pretrain_config,
            )
            every = int(config.get("checkpoint", {}).get("save_every_epochs", 0))
            if every > 0 and (epoch + 1) % every == 0:
                _save(
                    checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
                    unwrapped,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_miou,
                    best_epoch,
                    config,
                    pretrain_config,
                )
        else:
            best_miou = max(best_miou, current_miou)
        if world_size > 1:
            dist.barrier()

    elapsed = time.time() - training_start
    local_peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    peak_tensor = torch.tensor(local_peak_memory, dtype=torch.int64, device=device)
    if world_size > 1:
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    peak_memory = int(peak_tensor.item())
    if rank == 0:
        runtime = {
            "elapsed_seconds": elapsed,
            "epochs": epochs,
            "updates_per_epoch": steps_per_epoch,
            "total_updates": epochs * steps_per_epoch,
            "peak_gpu_memory_bytes": peak_memory,
            "peak_gpu_memory_gib": peak_memory / (1024**3),
            "resumed": start_epoch > 0,
        }
        (metrics_dir / "training_runtime.json").write_text(
            json.dumps(runtime, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_checksums(output_dir)
        (output_dir / "TRAINING_COMPLETE.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "best_epoch": best_epoch,
                    "best_global_miou": best_miou,
                    "last_epoch": epochs,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: Path,
    split: str,
    metrics_output: Path | None = None,
    per_sample_output: Path | None = None,
) -> dict[str, float]:
    world_size, rank, local_rank = _distributed()
    required_world_size = config.get("distributed", {}).get("required_world_size")
    if required_world_size is not None and world_size != int(required_world_size):
        raise RuntimeError(
            f"Benchmark evaluation requires {required_world_size} distributed workers, "
            f"got {world_size}"
        )
    seed = int(config["experiment"].get("seed", 42))
    _seed(seed, rank)
    allow_repeat_test = bool(config.get("evaluation", {}).get("allow_repeat_test", False))
    if split == "test" and metrics_output is not None and not allow_repeat_test:
        metrics_destination = metrics_output.expanduser().resolve()
        prospective_run_directory = (
            metrics_destination.parent.parent
            if metrics_destination.parent.name == "metrics"
            else metrics_destination.parent
        )
        if (prospective_run_directory / "TESTING_COMPLETE.json").is_file():
            raise RuntimeError(
                "WorldFloods test has already been completed for this run; refusing "
                "a second held-out evaluation"
            )
    device = (
        torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    evaluation_start = time.time()
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    _validate_checkpoint_compatibility(config, checkpoint)
    model, loaded_pretrain_config = _build_model(config, device)
    saved_pretrain_config = checkpoint.get("pretrain_config")
    if not isinstance(saved_pretrain_config, dict):
        raise ValueError("Segmentation checkpoint has no pretraining provenance")
    _validate_pretrain_identity(loaded_pretrain_config, saved_pretrain_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    eval_dataset = _dataset(config, split)
    expected_count = config["data"].get("expected_tile_counts", {}).get(split)
    if expected_count is not None and len(eval_dataset) != int(expected_count):
        raise RuntimeError(
            f"WorldFloods {split} count mismatch: expected {expected_count}, got {len(eval_dataset)}"
        )
    eval_sampler = (
        DistributedEvalSampler(eval_dataset, rank=rank, world_size=world_size)
        if world_size > 1
        else None
    )
    loader = _loader(
        eval_dataset,
        config["optimization"],
        sampler=eval_sampler,
        shuffle=False,
        batch_size=1,
        seed=seed + rank,
    )
    dtype_name = str(config["optimization"].get("dtype", "float16")).lower()
    autocast_dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    class_weights = None
    if bool(config["loss"].get("use_class_weights", True)):
        train_weight_source = _dataset(config, "train")
        class_weights = torch.as_tensor(train_weight_source.class_weights, device=device)
    per_sample_records = (
        [] if bool(config.get("evaluation", {}).get("save_per_sample_metrics", False)) else None
    )
    metrics = evaluate(
        model,
        loader,
        device,
        autocast_dtype,
        device.type == "cuda" and dtype_name in {"float16", "bfloat16"},
        class_weights=class_weights,
        loss_cfg=config["loss"],
        per_sample_records=per_sample_records,
        crop_size=int(config["data"]["crop_size"]),
        window_batch_size=_evaluation_batch_size(config),
        progress_description=f"{split.capitalize()} evaluation",
    )
    if world_size > 1 and per_sample_records is not None:
        gathered_records = [None] * world_size if rank == 0 else None
        dist.gather_object(per_sample_records, gathered_records, dst=0)
        if rank == 0:
            per_sample_records = [
                record
                for rank_records in gathered_records
                for record in rank_records
            ]
            per_sample_records.sort(key=lambda record: str(record["sample_id"]))
        else:
            per_sample_records = None
    if per_sample_records is not None:
        metrics.update(_grouped_metric_summary(per_sample_records))
    elapsed_seconds = time.time() - evaluation_start
    local_peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    peak_tensor = torch.tensor(local_peak_memory, dtype=torch.int64, device=device)
    if world_size > 1:
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    peak_memory = int(peak_tensor.item())
    result = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "selected_epoch": int(checkpoint["epoch"]) + 1,
        **metrics,
        "elapsed_seconds": elapsed_seconds,
        "tiles_per_second": len(eval_dataset) / max(elapsed_seconds, 1.0e-12),
        "peak_gpu_memory_bytes": peak_memory,
        "peak_gpu_memory_gib": peak_memory / (1024**3),
    }
    if rank == 0:
        print(json.dumps(result, indent=2))
    if rank == 0 and metrics_output is not None:
        destination = metrics_output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if rank == 0 and per_sample_records is not None:
        if per_sample_output is None:
            per_sample_output = checkpoint_path.parent / f"{split}_per_sample_metrics.csv"
        per_sample_destination = per_sample_output.expanduser().resolve()
        per_sample_destination.parent.mkdir(parents=True, exist_ok=True)
        if per_sample_records:
            with per_sample_destination.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(per_sample_records[0]))
                writer.writeheader()
                writer.writerows(per_sample_records)
        confusion = np.zeros((2, 2), dtype=np.int64)
        for record in per_sample_records:
            confusion += np.asarray(
                [
                    [record["target_land_pred_land"], record["target_land_pred_water"]],
                    [record["target_water_pred_land"], record["target_water_pred_water"]],
                ],
                dtype=np.int64,
            )
        if metrics_output is not None:
            resolved_metrics_output = metrics_output.expanduser().resolve()
            confusion_stem = resolved_metrics_output.stem
            if confusion_stem.endswith("_metrics"):
                confusion_stem = confusion_stem[: -len("_metrics")]
            confusion_destination = (
                resolved_metrics_output.parent / f"{confusion_stem}_confusion_matrix.csv"
            )
        else:
            confusion_destination = per_sample_destination.parent / f"{split}_confusion_matrix.csv"
        with confusion_destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["target", "predicted_land", "predicted_water"])
            writer.writerow(["land", int(confusion[0, 0]), int(confusion[0, 1])])
            writer.writerow(["water", int(confusion[1, 0]), int(confusion[1, 1])])
    if rank == 0 and split == "test" and metrics_output is not None:
        metrics_destination = metrics_output.expanduser().resolve()
        run_directory = (
            metrics_destination.parent.parent
            if metrics_destination.parent.name == "metrics"
            else metrics_destination.parent
        )
        _write_benchmark_summary(run_directory, result, config)
        _write_checksums(run_directory)
        (run_directory / "TESTING_COMPLETE.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "split": "test",
                    "selected_epoch": int(checkpoint["epoch"]) + 1,
                    "metrics": str(metrics_destination),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluate", choices=["val", "test"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optionally write evaluation metrics as JSON",
    )
    parser.add_argument(
        "--per-sample-output",
        type=Path,
        help="Optionally write per-tile evaluation metrics as CSV",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a config value; may be specified more than once",
    )
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    if args.evaluate:
        if args.checkpoint is None:
            parser.error("--checkpoint is required with --evaluate")
        evaluate_checkpoint(
            config,
            args.checkpoint,
            args.evaluate,
            metrics_output=args.metrics_output,
            per_sample_output=args.per_sample_output,
        )
    else:
        train(config)


if __name__ == "__main__":
    main()
