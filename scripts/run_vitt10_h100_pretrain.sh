#!/usr/bin/env bash
# ViT-Tiny/16 WorldFloods pretraining on one H100.
# Keeps the reported recipe: global batch 32, 100 epochs, AdamW 2e-4, BF16.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-pretrain}"
GPU_IDS="${GPU_IDS:-0}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"
GLOBAL_BATCH_SIZE="${H100_PRETRAIN_GLOBAL_BATCH_SIZE:-32}"

if [[ "${PHASE}" != "smoke" && "${WORLD_SIZE}" -ne 1 ]]; then
  echo "The reported H100 pretraining run uses one GPU and global batch ${GLOBAL_BATCH_SIZE}. GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % WORLD_SIZE != 0 )); then
  echo "H100_PRETRAIN_GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by ${WORLD_SIZE} GPUs" >&2
  exit 2
fi

export GPU_IDS
export WORLDFLOODS_LMDB_DIR="${WORLDFLOODS_LMDB_DIR:-${PROJECT_ROOT}/datasets/WorldFloodsv2/lmdb}"
if [[ ! -f "${WORLDFLOODS_LMDB_DIR}/worldfloods_train.lmdb" ]]; then
  echo "Missing packed train LMDB: ${WORLDFLOODS_LMDB_DIR}/worldfloods_train.lmdb" >&2
  echo "Flood-JEPA reads 512px tiles (21,364 train), not the raw WorldFloodsv2 GeoTIFFs." >&2
  exit 2
fi
export PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-${PROJECT_ROOT}/configs/pretrain_vitt16_256_10band_h100.yaml}"
export MODEL_NAME=vit_tiny
export ENCODER_DIM=192
export PREDICTOR_DEPTH=4
export PREDICTOR_DIM=192
export PREDICTOR_HEADS=6
export PRETRAIN_BATCH_PER_GPU="$((GLOBAL_BATCH_SIZE / WORLD_SIZE))"
export PRETRAIN_DTYPE="${PRETRAIN_DTYPE:-bfloat16}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-false}"
export PRETRAIN_OUTPUT="${PRETRAIN_OUTPUT:-${PROJECT_ROOT}/results/h100/vjepa21_vitt16_worldfloods_10band_256/pretrain/seed_${SEED:-42}}"
export SMOKE_OUTPUT="${SMOKE_OUTPUT:-${PROJECT_ROOT}/results/h100/smoke_vitt16_worldfloods_10band_256/seed_${SEED:-42}}"

exec "${PROJECT_ROOT}/scripts/run_vitb10_pipeline.sh" "${PHASE}"
