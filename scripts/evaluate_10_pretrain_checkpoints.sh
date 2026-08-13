#!/usr/bin/env bash
set -euo pipefail

# Frozen-encoder WorldFloods segmentation probe for pretraining epochs 10..100.
# The target encoder is never updated; only a fresh, identical decoder is fit.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WORLDFLOODS_LMDB_DIR:?Set WORLDFLOODS_LMDB_DIR to the WorldFloods LMDB directory}"

GPU_IDS="${GPU_IDS:-0,2,3,4,5,6}"
IFS=',' read -r -a SELECTED_GPUS <<< "${GPU_IDS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#SELECTED_GPUS[@]}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PRETRAIN_DIR="${PRETRAIN_DIR:-${PROJECT_ROOT}/results/vjepa21_vitb16_worldfloods_10band_256/seed_42/checkpoints}"
PROBE_ROOT="${PROBE_ROOT:-${PROJECT_ROOT}/results/worldfloods_10_checkpoint_frozen_probe}"
PROBE_CONFIG="${PROBE_CONFIG:-${PROJECT_ROOT}/configs/probe_worldfloods_10band.yaml}"
PROBE_EPOCHS="${PROBE_EPOCHS:-20}"
PROBE_BATCH_PER_GPU="${PROBE_BATCH_PER_GPU:-8}"
PROBE_WORKERS="${PROBE_WORKERS:-8}"
DECODER_LR="${DECODER_LR:-0.0001}"
SEED="${SEED:-42}"

mkdir -p "${PROBE_ROOT}"

for epoch in 010 020 030 040 050 060 070 080 090 100; do
  pretrain_checkpoint="${PRETRAIN_DIR}/epoch_${epoch}.pt"
  probe_output="${PROBE_ROOT}/epoch_${epoch}/seed_${SEED}"
  best_probe="${probe_output}/best.pt"
  metrics_json="${probe_output}/test_metrics.json"

  if [[ ! -f "${pretrain_checkpoint}" ]]; then
    echo "Missing pretraining checkpoint: ${pretrain_checkpoint}" >&2
    exit 1
  fi

  echo "=== Epoch ${epoch}: frozen decoder-probe training ==="
  if [[ ! -f "${best_probe}" ]]; then
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      -m flood_jepa.training.segmentation \
      --config "${PROBE_CONFIG}" \
      --override "experiment.seed=${SEED}" \
      --override "experiment.output_dir=${probe_output}" \
      --override "model.pretrained_checkpoint=${pretrain_checkpoint}" \
      --override "model.encoder_source=target" \
      --override "model.freeze_encoder=true" \
      --override "model.use_activation_checkpointing=false" \
      --override "optimization.epochs=${PROBE_EPOCHS}" \
      --override "optimization.per_device_batch_size=${PROBE_BATCH_PER_GPU}" \
      --override "optimization.num_workers=${PROBE_WORKERS}" \
      --override "optimization.encoder_learning_rate=0.0" \
      --override "optimization.decoder_learning_rate=${DECODER_LR}"
  else
    echo "Using existing probe: ${best_probe}"
  fi

  echo "=== Epoch ${epoch}: official test evaluation ==="
  CUDA_VISIBLE_DEVICES="${SELECTED_GPUS[0]}" python \
    -m flood_jepa.training.segmentation \
    --config "${PROBE_CONFIG}" \
    --evaluate test \
    --checkpoint "${best_probe}" \
    --metrics-output "${metrics_json}" \
    --override "experiment.output_dir=${probe_output}" \
    --override "model.pretrained_checkpoint=${pretrain_checkpoint}" \
    --override "model.encoder_source=target" \
    --override "model.freeze_encoder=true" \
    --override "model.use_activation_checkpointing=false" \
    --override "optimization.per_device_batch_size=${PROBE_BATCH_PER_GPU}" \
    --override "optimization.num_workers=${PROBE_WORKERS}"
done

PROBE_ROOT="${PROBE_ROOT}" SEED="${SEED}" python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["PROBE_ROOT"])
seed = int(os.environ["SEED"])
rows = []
for epoch in range(10, 101, 10):
    path = root / f"epoch_{epoch:03d}" / f"seed_{seed}" / "test_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"pretrain_epoch": epoch, **metrics})

destination = root / "test_checkpoint_comparison.csv"
with destination.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {destination}")
PY
