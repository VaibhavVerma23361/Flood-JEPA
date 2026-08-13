"""Fine-tune and evaluate the Flood-JEPA encoder on WorldFloods segmentation."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from flood_jepa.config import apply_overrides, load_config, require_sections
from flood_jepa.data.supervised import IGNORE_INDEX, WorldFloodsSegmentationDataset
from flood_jepa.models.segmentation import FloodSegmenter, LinearFloodSegmenter
from flood_jepa.models.vjepa21 import build_flood_jepa


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return world_size, rank, local_rank


def _seed(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _build_model(config: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    model_cfg = config["model"]
    pretrain_path = Path(model_cfg["pretrained_checkpoint"]).expanduser().resolve()
    if not pretrain_path.is_file():
        raise FileNotFoundError(
            f"Missing JEPA pretraining checkpoint: {pretrain_path}. "
            "Set FLOOD_JEPA_PRETRAINED or update the fine-tuning config."
        )
    checkpoint = torch.load(pretrain_path, map_location="cpu")
    pretrain_config = checkpoint["config"]
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
    else:
        raise ValueError("model.decoder_type must be 'linear' or 'fusion'")
    del jepa

    freeze_encoder = bool(model_cfg.get("freeze_encoder", False))
    for parameter in segmenter.backbone.parameters():
        parameter.requires_grad = not freeze_encoder
    return segmenter.to(device), pretrain_config


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, num_classes=2).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=logits.dtype)
    valid_float = valid.unsqueeze(1).to(dtype=logits.dtype)
    probabilities = logits.softmax(dim=1) * valid_float
    one_hot = one_hot * valid_float
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = denominator > 0
    dice = (2.0 * intersection[present] + 1.0e-6) / (
        denominator[present] + 1.0e-6
    )
    return 1.0 - dice.mean()


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
    dice = _dice_loss(logits, target)
    total = float(loss_cfg.get("ce_weight", 0.5)) * ce + float(
        loss_cfg.get("dice_weight", 0.5)
    ) * dice
    return total, {"ce": ce.detach(), "dice": dice.detach()}


def _confusion(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = logits.argmax(dim=1)
    valid = target != IGNORE_INDEX
    encoded = target[valid] * 2 + prediction[valid]
    return torch.bincount(encoded, minlength=4).reshape(2, 2).to(torch.float64)


def _metrics(confusion: torch.Tensor) -> dict[str, float]:
    true_positive = confusion.diag()
    actual = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = actual + predicted - true_positive
    iou = true_positive / union.clamp_min(1.0)
    precision = true_positive[1] / predicted[1].clamp_min(1.0)
    recall = true_positive[1] / actual[1].clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    return {
        "miou": float(iou.mean()),
        "land_iou": float(iou[0]),
        "water_iou": float(iou[1]),
        "water_precision": float(precision),
        "water_recall": float(recall),
        "water_f1": float(f1),
        "valid_pixels": int(confusion.sum().item()),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    mixed_precision: bool,
) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros((2, 2), dtype=torch.float64, device=device)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=mixed_precision,
        ):
            logits = model(images)
        confusion += _confusion(logits, target)
    return _metrics(confusion.cpu())


def _dataset(config: dict, split: str) -> WorldFloodsSegmentationDataset:
    data_cfg = config["data"]
    return WorldFloodsSegmentationDataset(
        lmdb_path=data_cfg[f"{split}_lmdb"],
        split=split,
        crop_size=int(data_cfg["crop_size"]),
        input_bands=list(data_cfg["input_bands"]),
        normalization=str(data_cfg.get("normalization", "worldfloods")),
        augment_d4=split == "train" and bool(data_cfg.get("augment_d4", True)),
        minimum_supervised_fraction=float(
            data_cfg.get("minimum_supervised_fraction", 0.25)
        ),
        sample_retry_attempts=int(data_cfg.get("sample_retry_attempts", 20)),
    )


def _loader(dataset, training_cfg: dict, sampler=None, shuffle=False) -> DataLoader:
    workers = int(training_cfg.get("num_workers", 8))
    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get("per_device_batch_size", 8)),
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers,
        pin_memory=bool(training_cfg.get("pin_memory", True)),
        persistent_workers=bool(training_cfg.get("persistent_workers", True)) and workers > 0,
        drop_last=False,
    )


def _save(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_miou: float,
    config: dict,
    pretrain_config: dict,
) -> None:
    unwrapped = model.module if hasattr(model, "module") else model
    payload = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
        "config": config,
        "pretrain_config": pretrain_config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config: dict) -> None:
    require_sections(config, "experiment", "data", "model", "loss", "optimization")
    world_size, rank, local_rank = _distributed()
    seed = int(config["experiment"].get("seed", 42))
    _seed(seed, rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    output_dir = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()

    train_dataset = _dataset(config, "train")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    train_loader = _loader(train_dataset, config["optimization"], sampler=train_sampler)
    val_loader = None
    if rank == 0:
        val_loader = _loader(_dataset(config, "val"), config["optimization"], shuffle=False)

    model, pretrain_config = _build_model(config, device)
    if list(config["data"]["input_bands"]) != list(pretrain_config["data"]["input_bands"]):
        raise ValueError("Fine-tuning bands/order must exactly match JEPA pretraining")
    opt_cfg = config["optimization"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": float(opt_cfg.get("encoder_learning_rate", 1.0e-5)),
            },
            {
                "params": model.decoder.parameters(),
                "lr": float(opt_cfg.get("decoder_learning_rate", 1.0e-4)),
            },
        ],
        weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
    )
    epochs = int(opt_cfg.get("epochs", 50))
    configured_steps = opt_cfg.get("max_steps_per_epoch")
    steps_per_epoch = len(train_loader)
    if configured_steps is not None:
        steps_per_epoch = min(steps_per_epoch, int(configured_steps))
        if steps_per_epoch <= 0:
            raise ValueError("optimization.max_steps_per_epoch must be positive or null")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(opt_cfg.get("minimum_learning_rate", 1.0e-6)),
    )
    start_epoch = 0
    best_miou = -1.0
    resume = config.get("checkpoint", {}).get("resume")
    if resume:
        resumed = torch.load(Path(resume).expanduser(), map_location="cpu")
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        start_epoch = int(resumed["epoch"]) + 1
        best_miou = float(resumed["best_miou"])

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
    log_path = output_dir / "training.jsonl"

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        if bool(config["model"].get("freeze_encoder", False)):
            # Preserve deterministic frozen features even while the decoder is
            # in training mode.
            unwrapped.backbone.eval()
        epoch_loss = torch.zeros(2, dtype=torch.float64, device=device)
        epoch_start = time.time()
        progress = tqdm(
            islice(train_loader, steps_per_epoch),
            total=steps_per_epoch,
            desc=f"Probe epoch {epoch + 1}/{epochs}",
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
                loss, _parts = _loss(logits, target, class_weights, config["loss"])
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in unwrapped.parameters() if parameter.requires_grad],
                    float(opt_cfg.get("gradient_clip_norm", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in unwrapped.parameters() if parameter.requires_grad],
                    float(opt_cfg.get("gradient_clip_norm", 1.0)),
                )
                optimizer.step()
            epoch_loss[0] += loss.detach().double()
            epoch_loss[1] += 1
            if rank == 0:
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        scheduler.step()
        if world_size > 1:
            dist.all_reduce(epoch_loss)
            dist.barrier()

        metric_value = torch.tensor(-1.0, device=device)
        validation = None
        if rank == 0:
            validation = evaluate(
                unwrapped,
                val_loader,
                device,
                autocast_dtype,
                mixed_precision,
            )
            metric_value.fill_(validation["miou"])
        if world_size > 1:
            dist.broadcast(metric_value, src=0)
        current_miou = float(metric_value)

        if rank == 0:
            record = {
                "epoch": epoch + 1,
                "train_loss": float(epoch_loss[0] / epoch_loss[1].clamp_min(1)),
                "elapsed_seconds": time.time() - epoch_start,
                "validation": validation,
                "encoder_learning_rate": optimizer.param_groups[0]["lr"],
                "decoder_learning_rate": optimizer.param_groups[1]["lr"],
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(record, flush=True)
            _save(
                output_dir / "latest.pt",
                unwrapped,
                optimizer,
                scheduler,
                epoch,
                max(best_miou, current_miou),
                config,
                pretrain_config,
            )
            if current_miou > best_miou:
                best_miou = current_miou
                _save(
                    output_dir / "best.pt",
                    unwrapped,
                    optimizer,
                    scheduler,
                    epoch,
                    best_miou,
                    config,
                    pretrain_config,
                )
        else:
            best_miou = max(best_miou, current_miou)
        if world_size > 1:
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: Path,
    split: str,
    metrics_output: Path | None = None,
) -> dict[str, float]:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Checkpoint evaluation is single-process; launch it with python")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model, _pretrain_config = _build_model(config, device)
    checkpoint = torch.load(checkpoint_path.expanduser().resolve(), map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = _loader(_dataset(config, split), config["optimization"], shuffle=False)
    dtype_name = str(config["optimization"].get("dtype", "float16")).lower()
    autocast_dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    metrics = evaluate(
        model,
        loader,
        device,
        autocast_dtype,
        device.type == "cuda" and dtype_name in {"float16", "bfloat16"},
    )
    result = {"split": split, "checkpoint": str(checkpoint_path), **metrics}
    print(json.dumps(result, indent=2))
    if metrics_output is not None:
        destination = metrics_output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
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
        )
    else:
        train(config)


if __name__ == "__main__":
    main()
