#!/usr/bin/env bash
set -euo pipefail

: "${WORLDFLOODS_LMDB_DIR:?Set WORLDFLOODS_LMDB_DIR to the existing WorldFloods lmdb directory}"
python -m flood_jepa.data.worldfloods \
  "${WORLDFLOODS_LMDB_DIR}/worldfloods_train.lmdb" \
  --crop-size 256

