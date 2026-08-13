#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WORLDFLOODS_LMDB_DIR:?Set WORLDFLOODS_LMDB_DIR to the existing WorldFloods lmdb directory}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m flood_jepa.training.pretrain \
  --config "${FLOOD_JEPA_CONFIG:-${PROJECT_ROOT}/configs/pretrain_vitb16_256_10band.yaml}"
