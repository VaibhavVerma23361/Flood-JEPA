"""Distributed WorldFloods pretraining for the V-JEPA 2.1 image architecture."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from flood_jepa.config import apply_overrides, load_config, require_sections
from flood_jepa.data.masks import WorldFloodsMaskCollator
from flood_jepa.data.worldfloods import WorldFloodsPretrainDataset
from flood_jepa.models.vjepa21 import build_flood_jepa
from flood_jepa.training.losses import (
    SketchedIsotropicGaussianRegularizer,
    dense_jepa_loss,
    normalize_deep_targets,
)
from flood_jepa.vendor import upstream_root


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return world_size, rank, local_rank


def _seed_everything(seed: int, rank: int) -> None:
    rank_seed = seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def _cosine(start: float, end: float, progress: float) -> float:
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def _lr_at_step(step: int, total_steps: int, warmup_steps: int, cfg: dict) -> float:
    start_lr = float(cfg["start_learning_rate"])
    peak_lr = float(cfg["learning_rate"])
    final_lr = float(cfg["final_learning_rate"])
    if warmup_steps > 0 and step < warmup_steps:
        return start_lr + (peak_lr - start_lr) * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return _cosine(peak_lr, final_lr, min(max(progress, 0.0), 1.0))


def _context_weight(step: int, total_steps: int, cfg: dict) -> float:
    target = float(cfg.get("context_weight", 0.5))
    start = int(total_steps * float(cfg.get("context_warmup_start_fraction", 0.05)))
    end = int(total_steps * float(cfg.get("context_warmup_end_fraction", 0.10)))
    if step < start:
        return 0.0
    if step >= end:
        return target
    return target * (step - start) / max(1, end - start)


def _ema_at_step(step: int, total_steps: int, cfg: dict) -> float:
    start = float(cfg.get("ema_start", 0.99925))
    end = float(cfg.get("ema_end", start))
    return start + (end - start) * step / max(1, total_steps - 1)


def _set_optimizer_step(optimizer, step: int, total_steps: int, warmup_steps: int, cfg: dict):
    lr = _lr_at_step(step, total_steps, warmup_steps, cfg)
    wd_progress = step / max(1, total_steps - 1)
    weight_decay = _cosine(
        float(cfg["weight_decay"]),
        float(cfg["final_weight_decay"]),
        wd_progress,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["weight_decay"] = 0.0 if group.get("exclude_from_weight_decay") else weight_decay
    return lr, weight_decay


def _optimizer(model, cfg: dict):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or "mask_tokens" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "exclude_from_weight_decay": False},
            {"params": no_decay, "exclude_from_weight_decay": True, "weight_decay": 0.0},
        ],
        lr=float(cfg["start_learning_rate"]),
        betas=(0.9, 0.999),
    )


def _git_revision(path: Path) -> str:
    head = path / ".git" / "HEAD"
    if not head.is_file():
        return "unknown"
    head_value = head.read_text(encoding="utf-8").strip()
    if head_value.startswith("ref: "):
        ref_path = path / ".git" / head_value[5:]
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
    return head_value


def _save_checkpoint(path: Path, model, optimizer, epoch: int, global_step: int, config: dict):
    unwrapped = model.module if hasattr(model, "module") else model
    payload = {
        "online_encoder": unwrapped.online_encoder.state_dict(),
        "target_encoder": unwrapped.target_encoder.state_dict(),
        "predictor": unwrapped.predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "upstream_revision": _git_revision(upstream_root()),
    }
    if unwrapped.sigreg_projector is not None:
        payload["sigreg_projector"] = unwrapped.sigreg_projector.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path, model, optimizer) -> tuple[int, int]:
    # Resume checkpoints contain optimizer/scheduler/config metadata in
    # addition to tensors. Explicitly allow that trusted local payload under
    # the PyTorch 2.6+ weights-only default.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.online_encoder.load_state_dict(checkpoint["online_encoder"])
    model.target_encoder.load_state_dict(checkpoint["target_encoder"])
    model.predictor.load_state_dict(checkpoint["predictor"])
    if model.sigreg_projector is not None:
        if "sigreg_projector" not in checkpoint:
            raise ValueError(
                "SIGReg is enabled, but the resume checkpoint has no SIGReg projector state"
            )
        model.sigreg_projector.load_state_dict(checkpoint["sigreg_projector"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


def train(config: dict) -> None:
    require_sections(
        config,
        "experiment",
        "data",
        "model",
        "loss",
        "optimization",
        "checkpoint",
    )
    world_size, rank, local_rank = _distributed()
    experiment = config["experiment"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    sigreg_cfg = loss_cfg.get("sigreg", {})
    opt_cfg = config["optimization"]
    checkpoint_cfg = config["checkpoint"]
    seed = int(experiment.get("seed", 42))
    _seed_everything(seed, rank)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    output_dir = Path(experiment["output_dir"]).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()

    dataset = WorldFloodsPretrainDataset(
        lmdb_path=data_cfg["lmdb_path"],
        crop_size=int(data_cfg["crop_size"]),
        input_bands=list(data_cfg["input_bands"]),
        normalization=data_cfg.get("normalization", "worldfloods"),
        augment_d4=bool(data_cfg.get("augment_d4", True)),
        minimum_nonpadding_fraction=float(data_cfg.get("minimum_nonpadding_fraction", 0.25)),
        sample_retry_attempts=int(data_cfg.get("sample_retry_attempts", 20)),
    )
    if int(model_cfg["input_channels"]) != len(dataset.input_bands):
        raise ValueError("model.input_channels must equal the number of data.input_bands")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    collator = WorldFloodsMaskCollator(
        mask_config=config["mask"],
        crop_size=int(data_cfg["crop_size"]),
        patch_size=int(model_cfg["patch_size"]),
    )
    workers = int(opt_cfg.get("num_workers", 8))
    loader = DataLoader(
        dataset,
        batch_size=int(opt_cfg["per_device_batch_size"]),
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(opt_cfg.get("pin_memory", True)),
        persistent_workers=bool(opt_cfg.get("persistent_workers", True)) and workers > 0,
        drop_last=True,
        collate_fn=collator,
    )

    model = build_flood_jepa(
        model_cfg,
        image_size=int(data_cfg["crop_size"]),
        mask_count=len(config["mask"]),
        sigreg_config=sigreg_cfg,
    ).to(device)
    sigreg = None
    if bool(sigreg_cfg.get("enabled", False)):
        if int(opt_cfg["per_device_batch_size"]) < 2:
            raise ValueError("SIGReg requires optimization.per_device_batch_size >= 2")
        sigreg = SketchedIsotropicGaussianRegularizer(
            num_projections=int(sigreg_cfg.get("num_projections", 256)),
            num_knots=int(sigreg_cfg.get("num_knots", 17)),
            t_max=float(sigreg_cfg.get("t_max", 3.0)),
            seed=int(sigreg_cfg.get("seed", seed)),
        ).to(device)
    optimizer = _optimizer(model, opt_cfg)

    start_epoch = 0
    global_step = 0
    resume = checkpoint_cfg.get("resume")
    if resume:
        start_epoch, global_step = _load_checkpoint(Path(resume), model, optimizer)

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    epochs = int(opt_cfg["epochs"])
    configured_steps = opt_cfg.get("max_steps_per_epoch")
    steps_per_epoch = len(loader)
    if configured_steps is not None:
        steps_per_epoch = min(steps_per_epoch, int(configured_steps))
        if steps_per_epoch <= 0:
            raise ValueError("optimization.max_steps_per_epoch must be positive or null")
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(opt_cfg.get("warmup_epochs", 0)) * steps_per_epoch
    dtype_name = str(opt_cfg.get("dtype", "float16")).lower()
    autocast_dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    mixed_precision = device.type == "cuda" and dtype_name in {"float16", "bfloat16"}
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and autocast_dtype == torch.float16)
    log_path = output_dir / "training_rank0.jsonl"

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        progress = tqdm(
            islice(loader, steps_per_epoch),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{epochs}",
            dynamic_ncols=True,
            disable=rank != 0,
            unit="step",
        )
        for iteration, (batch, masks_encoder, masks_predictor) in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True)
            masks_encoder = [mask.to(device, non_blocking=True) for mask in masks_encoder]
            masks_predictor = [mask.to(device, non_blocking=True) for mask in masks_predictor]
            lr, weight_decay = _set_optimizer_step(
                optimizer, global_step, total_steps, warmup_steps, opt_cfg
            )
            context_weight = _context_weight(global_step, total_steps, loss_cfg)
            optimizer.zero_grad(set_to_none=True)
            unwrapped = model.module if hasattr(model, "module") else model

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=mixed_precision,
            ):
                target, predicted, predicted_context, sigreg_embeddings = model(
                    images,
                    masks_encoder,
                    masks_predictor,
                )
                with torch.no_grad():
                    target = normalize_deep_targets(
                        target,
                        encoder_dim=unwrapped.info.encoder_dim,
                        levels=int(model_cfg.get("deep_self_supervision_levels", 4)),
                    )
                jepa_loss, loss_parts = dense_jepa_loss(
                    predicted,
                    predicted_context,
                    target,
                    masks_encoder,
                    masks_predictor,
                    grid_size=int(data_cfg["crop_size"]) // int(model_cfg["patch_size"]),
                    exponent=float(loss_cfg.get("exponent", 1.0)),
                    context_weight=context_weight if loss_cfg.get("predict_all", True) else 0.0,
                    distance_weighted_context=bool(
                        loss_cfg.get("distance_weighted_context", True)
                    ),
                )
                if sigreg is not None:
                    if sigreg_embeddings is None:
                        raise RuntimeError("SIGReg is enabled but the model returned no embeddings")
                    sigreg_loss = sigreg(sigreg_embeddings, step=global_step)
                else:
                    sigreg_loss = jepa_loss.new_zeros(())
                sigreg_weight = float(sigreg_cfg.get("weight", 0.05))
                loss = jepa_loss + sigreg_weight * sigreg_loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(unwrapped.trainable_parameters()),
                    float(opt_cfg.get("gradient_clip_norm", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(unwrapped.trainable_parameters()),
                    float(opt_cfg.get("gradient_clip_norm", 1.0)),
                )
                optimizer.step()

            momentum = _ema_at_step(global_step, total_steps, opt_cfg)
            unwrapped.update_target(momentum)
            epoch_loss += float(loss.detach())
            global_step += 1

            if rank == 0:
                progress.set_postfix(
                    loss=f"{float(loss.detach()):.4f}",
                    pred=f"{float(loss_parts['prediction']):.4f}",
                    context=f"{float(loss_parts['context']):.4f}",
                    sigreg=f"{float(sigreg_loss.detach()):.4f}",
                    lr=f"{lr:.2e}",
                    ema=f"{momentum:.6f}",
                )

            if rank == 0 and (iteration % 20 == 0 or iteration + 1 == steps_per_epoch):
                record = {
                    "epoch": epoch + 1,
                    "iteration": iteration + 1,
                    "global_step": global_step,
                    "loss": float(loss.detach()),
                    "jepa_loss": float(jepa_loss.detach()),
                    "prediction_loss": float(loss_parts["prediction"]),
                    "context_loss": float(loss_parts["context"]),
                    "context_weight": context_weight,
                    "sigreg_loss": float(sigreg_loss.detach()),
                    "sigreg_weight": sigreg_weight,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "ema": momentum,
                }
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")

        if rank == 0:
            summary = {
                "epoch": epoch + 1,
                "mean_loss": epoch_loss / max(1, steps_per_epoch),
                "elapsed_seconds": time.time() - epoch_start,
            }
            tqdm.write(json.dumps(summary))
            _save_checkpoint(
                checkpoint_dir / "latest.pt",
                unwrapped,
                optimizer,
                epoch,
                global_step,
                config,
            )
            every = int(checkpoint_cfg.get("save_every_epochs", 10))
            if every > 0 and (epoch + 1) % every == 0:
                _save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
                    unwrapped,
                    optimizer,
                    epoch,
                    global_step,
                    config,
                )
        if world_size > 1:
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a config value; may be specified more than once",
    )
    args = parser.parse_args()
    train(apply_overrides(load_config(args.config), args.override))


if __name__ == "__main__":
    main()
