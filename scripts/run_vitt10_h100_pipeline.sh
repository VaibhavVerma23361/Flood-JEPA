#!/usr/bin/env bash
# Recreate the reported Flood-JEPA ViT-Tiny WorldFloods V2 run on one H100.
# Usage: ./scripts/run_vitt10_h100_pipeline.sh {smoke|pretrain|finetune|train-test|all}
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-help}"
SEED="${SEED:-42}"
export SEED
PRETRAIN_RUN="${PROJECT_ROOT}/results/h100/vjepa21_vitt16_worldfloods_10band_256/pretrain/seed_${SEED}"
SMOKE_PRETRAIN_RUN="${PROJECT_ROOT}/results/h100/smoke_vitt16_worldfloods_10band_256/seed_${SEED}"

case "${PHASE}" in
  smoke)
    "${PROJECT_ROOT}/scripts/run_vitt10_h100_pretrain.sh" smoke
    FLOOD_JEPA_PRETRAINED="${SMOKE_PRETRAIN_RUN}/checkpoints/latest.pt" \
      "${PROJECT_ROOT}/scripts/run_vitt10_h100_finetune.sh" smoke
    ;;
  pretrain)
    "${PROJECT_ROOT}/scripts/run_vitt10_h100_pretrain.sh" pretrain
    ;;
  finetune)
    FLOOD_JEPA_PRETRAINED="${FLOOD_JEPA_PRETRAINED:-${PRETRAIN_RUN}/checkpoints/epoch_100.pt}" \
      "${PROJECT_ROOT}/scripts/run_vitt10_h100_finetune.sh" train
    ;;
  train-test)
    FLOOD_JEPA_PRETRAINED="${FLOOD_JEPA_PRETRAINED:-${PRETRAIN_RUN}/checkpoints/epoch_100.pt}" \
      "${PROJECT_ROOT}/scripts/run_vitt10_h100_finetune.sh" train-test
    ;;
  all)
    "${PROJECT_ROOT}/scripts/run_vitt10_h100_pretrain.sh" pretrain
    FLOOD_JEPA_PRETRAINED="${PRETRAIN_RUN}/checkpoints/epoch_100.pt" \
      "${PROJECT_ROOT}/scripts/run_vitt10_h100_finetune.sh" train-test
    ;;
  *)
    echo "Usage: $0 {smoke|pretrain|finetune|train-test|all}" >&2
    exit 2
    ;;
esac
