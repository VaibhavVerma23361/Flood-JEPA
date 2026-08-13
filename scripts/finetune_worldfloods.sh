#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WORLDFLOODS_LMDB_DIR:?Set WORLDFLOODS_LMDB_DIR to the WorldFloods LMDB directory}"
: "${FLOOD_JEPA_PRETRAINED:?Set FLOOD_JEPA_PRETRAINED to a Flood-JEPA pretraining checkpoint}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m flood_jepa.training.segmentation \
  --config "${FLOOD_JEPA_FINETUNE_CONFIG:-${PROJECT_ROOT}/configs/finetune_worldfloods_10band.yaml}"
