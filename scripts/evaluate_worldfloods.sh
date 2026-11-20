#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WORLDFLOODS_LMDB_DIR:?Set WORLDFLOODS_LMDB_DIR to the WorldFloods LMDB directory}"
: "${FLOOD_JEPA_PRETRAINED:?Set FLOOD_JEPA_PRETRAINED to the Flood-JEPA pretraining checkpoint used for fine-tuning}"
: "${FLOOD_JEPA_SEGMENTER:?Set FLOOD_JEPA_SEGMENTER to the selected best.pt segmentation checkpoint}"

EVAL_SPLIT="${EVAL_SPLIT:-val}"
if [[ "${EVAL_SPLIT}" != "val" && "${EVAL_SPLIT}" != "test" ]]; then
  echo "EVAL_SPLIT must be 'val' or 'test'" >&2
  exit 2
fi
GPU_IDS="${GPU_IDS:-0,1,2,4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
EVAL_WORLD_SIZE="${EVAL_WORLD_SIZE:-7}"
EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-$((EVAL_WORLD_SIZE * 2))}"
EVAL_EXPERIMENT_NAME="${EVAL_EXPERIMENT_NAME:-flood_jepa_vitb10_256_frozen}"
ALLOW_REPEAT_TEST="${ALLOW_REPEAT_TEST:-false}"
if [[ "${#GPU_ARRAY[@]}" -ne "${EVAL_WORLD_SIZE}" ]]; then
  echo "Evaluation requires ${EVAL_WORLD_SIZE} GPUs; GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi
if [[ "${EVAL_WORLD_SIZE}" -ne 7 ]]; then
  echo "WARNING: evaluating a non-benchmark ${EVAL_WORLD_SIZE}-GPU run." >&2
fi
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/vjepa2${PYTHONPATH:+:${PYTHONPATH}}"
METRICS_OUTPUT="${METRICS_OUTPUT:-$(dirname "${FLOOD_JEPA_SEGMENTER}")/${EVAL_SPLIT}_metrics.json}"
PER_SAMPLE_OUTPUT="${PER_SAMPLE_OUTPUT:-$(dirname "${FLOOD_JEPA_SEGMENTER}")/${EVAL_SPLIT}_per_sample_metrics.csv}"
LOG_OUTPUT="${LOG_OUTPUT:-$(dirname "${FLOOD_JEPA_SEGMENTER}")/${EVAL_SPLIT}_evaluation.log}"
mkdir -p "$(dirname "${METRICS_OUTPUT}")" "$(dirname "${PER_SAMPLE_OUTPUT}")" "$(dirname "${LOG_OUTPUT}")"

torchrun --standalone --nproc_per_node="${EVAL_WORLD_SIZE}" -m flood_jepa.training.segmentation \
  --config "${FLOOD_JEPA_FINETUNE_CONFIG:-${PROJECT_ROOT}/configs/finetune_worldfloods_10band.yaml}" \
  --evaluate "${EVAL_SPLIT}" \
  --checkpoint "${FLOOD_JEPA_SEGMENTER}" \
  --metrics-output "${METRICS_OUTPUT}" \
  --per-sample-output "${PER_SAMPLE_OUTPUT}" \
  --override "experiment.name=${EVAL_EXPERIMENT_NAME}" \
  --override "distributed.required_world_size=${EVAL_WORLD_SIZE}" \
  --override "distributed.global_batch_size=${EVAL_GLOBAL_BATCH_SIZE}" \
  --override "evaluation.allow_repeat_test=${ALLOW_REPEAT_TEST}" 2>&1 | tee "${LOG_OUTPUT}"
