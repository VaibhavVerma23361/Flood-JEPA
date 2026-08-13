#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WORLDFLOODS_LMDB_DIR:?Set WORLDFLOODS_LMDB_DIR to the WorldFloods LMDB directory}"
: "${FLOOD_JEPA_PRETRAINED:?Set FLOOD_JEPA_PRETRAINED to the Flood-JEPA pretraining checkpoint used for fine-tuning}"
: "${FLOOD_JEPA_SEGMENTER:?Set FLOOD_JEPA_SEGMENTER to the selected best.pt segmentation checkpoint}"

python -m flood_jepa.training.segmentation \
  --config "${FLOOD_JEPA_FINETUNE_CONFIG:-${PROJECT_ROOT}/configs/finetune_worldfloods_10band.yaml}" \
  --evaluate test \
  --checkpoint "${FLOOD_JEPA_SEGMENTER}"
