#!/usr/bin/env bash
# Evaluate the H100 ViT-Tiny + SegUPerNet-512 checkpoint.
# EVAL_SPLIT=val rechecks selection. EVAL_SPLIT=test is the held-out report.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-0}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"
RUN_DIR="${H100_FINETUNE_RUN_DIR:-${PROJECT_ROOT}/results/h100/vjepa21_vitt16_worldfloods_10band_256/finetune/seed_${SEED:-42}}"

export GPU_IDS
export WORLDFLOODS_LMDB_DIR="${WORLDFLOODS_LMDB_DIR:-${PROJECT_ROOT}/datasets/WorldFloodsv2/lmdb}"
export EVAL_WORLD_SIZE="${WORLD_SIZE}"
export EVAL_GLOBAL_BATCH_SIZE="${H100_FINETUNE_GLOBAL_BATCH_SIZE:-14}"
export EVAL_EXPERIMENT_NAME="flood_jepa_vitt10_256_seg_upernet512_h100"
export FLOOD_JEPA_FINETUNE_CONFIG="${PROJECT_ROOT}/configs/finetune_worldfloods_vitt10_h100.yaml"
export FLOOD_JEPA_PRETRAINED="${FLOOD_JEPA_PRETRAINED:-${PROJECT_ROOT}/results/h100/vjepa21_vitt16_worldfloods_10band_256/pretrain/seed_${SEED:-42}/checkpoints/epoch_100.pt}"
export FLOOD_JEPA_SEGMENTER="${FLOOD_JEPA_SEGMENTER:-${RUN_DIR}/checkpoints/best.pt}"
export METRICS_OUTPUT="${METRICS_OUTPUT:-${RUN_DIR}/metrics/${EVAL_SPLIT:-val}_metrics.json}"
export PER_SAMPLE_OUTPUT="${PER_SAMPLE_OUTPUT:-${RUN_DIR}/metrics/${EVAL_SPLIT:-val}_per_sample.csv}"
export LOG_OUTPUT="${LOG_OUTPUT:-${RUN_DIR}/logs/${EVAL_SPLIT:-val}_evaluation.log}"

exec "${PROJECT_ROOT}/scripts/evaluate_worldfloods.sh"
