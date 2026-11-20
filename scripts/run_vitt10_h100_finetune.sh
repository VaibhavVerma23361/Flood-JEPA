#!/usr/bin/env bash
# Frozen ViT-Tiny + SegUPerNet-512 on one H100.
# Global batch 14 matches the 7x2 benchmark update count (1,526 steps/epoch).
# Validation global mIoU selects best.pt. The reported test number used epoch 10.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-train}"
GPU_IDS="${GPU_IDS:-0}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"
GLOBAL_BATCH_SIZE="${H100_FINETUNE_GLOBAL_BATCH_SIZE:-14}"

if (( GLOBAL_BATCH_SIZE % WORLD_SIZE != 0 )); then
  echo "H100_FINETUNE_GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by ${WORLD_SIZE} GPUs" >&2
  exit 2
fi
if [[ "${PHASE}" != "smoke" && "${WORLD_SIZE}" -ne 1 ]]; then
  echo "The reported H100 run uses one GPU and global batch ${GLOBAL_BATCH_SIZE}. GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi

export GPU_IDS
export WORLDFLOODS_LMDB_DIR="${WORLDFLOODS_LMDB_DIR:-${PROJECT_ROOT}/datasets/WorldFloodsv2/lmdb}"
for split in train val test; do
  if [[ ! -f "${WORLDFLOODS_LMDB_DIR}/worldfloods_${split}.lmdb" ]]; then
    echo "Missing packed ${split} LMDB: ${WORLDFLOODS_LMDB_DIR}/worldfloods_${split}.lmdb" >&2
    echo "Flood-JEPA reads 512px tiles (21,364/751/841), not the raw WorldFloodsv2 GeoTIFFs." >&2
    exit 2
  fi
done
export FINETUNE_WORLD_SIZE="${WORLD_SIZE}"
export FINETUNE_BATCH_PER_GPU="$((GLOBAL_BATCH_SIZE / WORLD_SIZE))"
export FLOOD_JEPA_FINETUNE_CONFIG="${FLOOD_JEPA_FINETUNE_CONFIG:-${PROJECT_ROOT}/configs/finetune_worldfloods_vitt10_h100.yaml}"
export FLOOD_JEPA_PRETRAINED="${FLOOD_JEPA_PRETRAINED:-${PROJECT_ROOT}/results/h100/vjepa21_vitt16_worldfloods_10band_256/pretrain/seed_${SEED:-42}/checkpoints/epoch_100.pt}"
if [[ "${PHASE}" == "smoke" ]]; then
  export FINETUNE_EXPERIMENT_NAME="${FINETUNE_EXPERIMENT_NAME:-smoke_flood_jepa_vitt10_h100}"
  export FINETUNE_OUTPUT="${FINETUNE_OUTPUT:-${PROJECT_ROOT}/results/h100/smoke_vitt16_worldfloods_10band_256/finetune/seed_${SEED:-42}}"
else
  export FINETUNE_EXPERIMENT_NAME="${FINETUNE_EXPERIMENT_NAME:-flood_jepa_vitt10_256_seg_upernet512_h100}"
  export FINETUNE_OUTPUT="${FINETUNE_OUTPUT:-${PROJECT_ROOT}/results/h100/vjepa21_vitt16_worldfloods_10band_256/finetune/seed_${SEED:-42}}"
fi
export FINETUNE_DTYPE="${FINETUNE_DTYPE:-bfloat16}"
export FINETUNE_DECODER_LABEL="SegUPerNet-512"
export FINETUNE_DECODER_DIM=512

exec "${PROJECT_ROOT}/scripts/finetune_worldfloods.sh" "${PHASE}"
