#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WORLDFLOODS_LMDB_DIR:?Set WORLDFLOODS_LMDB_DIR to the WorldFloods LMDB directory}"
: "${FLOOD_JEPA_PRETRAINED:?Set FLOOD_JEPA_PRETRAINED to a Flood-JEPA pretraining checkpoint}"

PHASE="${1:-train}"
GPU_IDS="${GPU_IDS:-0,1,2,4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
FINETUNE_WORLD_SIZE="${FINETUNE_WORLD_SIZE:-7}"
CONFIG="${FLOOD_JEPA_FINETUNE_CONFIG:-${PROJECT_ROOT}/configs/finetune_worldfloods_10band.yaml}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/vjepa2${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${TMPDIR:-/tmp/flood-jepa-${USER:-worker}}"
mkdir -p "${TMPDIR}"

case "${PHASE}" in
  train|train-test)
    EPOCHS="${FINETUNE_EPOCHS:-30}"
    MAX_STEPS="${FINETUNE_MAX_STEPS_PER_EPOCH:-null}"
    SCHEDULER="${FINETUNE_SCHEDULER:-cosine}"
    WARMUP_EPOCHS="${FINETUNE_WARMUP_EPOCHS:-3}"
    AUTO_RESUME="${FINETUNE_AUTO_RESUME:-true}"
    ;;
  smoke)
    EPOCHS="${FINETUNE_EPOCHS:-1}"
    MAX_STEPS="${FINETUNE_MAX_STEPS_PER_EPOCH:-5}"
    SCHEDULER="cosine"
    WARMUP_EPOCHS=0
    AUTO_RESUME=false
    ;;
  *)
    echo "Usage: $0 [smoke|train|train-test]" >&2
    exit 2
    ;;
esac
FINETUNE_BATCH_PER_GPU="${FINETUNE_BATCH_PER_GPU:-2}"
FINETUNE_NUM_WORKERS="${FINETUNE_NUM_WORKERS:-4}"
FINETUNE_DTYPE="${FINETUNE_DTYPE:-float16}"
FINETUNE_DECODER_LABEL="${FINETUNE_DECODER_LABEL:-benchmark SegUPerNet}"
FINETUNE_DECODER_DIM="${FINETUNE_DECODER_DIM:-512}"
GLOBAL_BATCH_SIZE=$((NPROC_PER_NODE * FINETUNE_BATCH_PER_GPU))
if [[ "${PHASE}" == "smoke" ]]; then
  DEFAULT_EXPERIMENT_NAME="smoke_worldfloods_flood_jepa_seg_upernet"
  DEFAULT_OUTPUT="${PROJECT_ROOT}/results/${DEFAULT_EXPERIMENT_NAME}"
elif [[ "${NPROC_PER_NODE}" -eq 7 && "${GLOBAL_BATCH_SIZE}" -eq 14 ]]; then
  DEFAULT_EXPERIMENT_NAME="flood_jepa_vitb10_256_frozen"
  DEFAULT_OUTPUT="${PROJECT_ROOT}/results/${DEFAULT_EXPERIMENT_NAME}/seed_42"
else
  DEFAULT_EXPERIMENT_NAME="flood_jepa_vitb10_256_frozen_${NPROC_PER_NODE}gpu_global${GLOBAL_BATCH_SIZE}"
  DEFAULT_OUTPUT="${PROJECT_ROOT}/results/${DEFAULT_EXPERIMENT_NAME}/seed_42"
fi
FINETUNE_EXPERIMENT_NAME="${FINETUNE_EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
FINETUNE_OUTPUT="${FINETUNE_OUTPUT:-${DEFAULT_OUTPUT}}"
if [[ "${NPROC_PER_NODE}" -ne "${#GPU_ARRAY[@]}" ]]; then
  echo "NPROC_PER_NODE (${NPROC_PER_NODE}) must match GPU_IDS count (${#GPU_ARRAY[@]})" >&2
  exit 2
fi
if [[ "${PHASE}" != "smoke" && "${NPROC_PER_NODE}" -ne "${FINETUNE_WORLD_SIZE}" ]]; then
  echo "Expected ${FINETUNE_WORLD_SIZE} GPUs, but GPU_IDS contains ${NPROC_PER_NODE}" >&2
  exit 2
fi
if [[ "${PHASE}" != "smoke" && "${NPROC_PER_NODE}" -ne 7 ]]; then
  echo "WARNING: this is not the locked 7-GPU/global-14 benchmark configuration." >&2
fi
mkdir -p \
  "${FINETUNE_OUTPUT}/checkpoints" \
  "${FINETUNE_OUTPUT}/config" \
  "${FINETUNE_OUTPUT}/logs" \
  "${FINETUNE_OUTPUT}/metrics"

OVERRIDES=(
  --override "experiment.name=${FINETUNE_EXPERIMENT_NAME}"
  --override "experiment.output_dir=${FINETUNE_OUTPUT}"
  --override "experiment.seed=${SEED:-42}"
  --override "optimization.epochs=${EPOCHS}"
  --override "optimization.scheduler=${SCHEDULER}"
  --override "optimization.warmup_epochs=${WARMUP_EPOCHS}"
  --override "optimization.max_steps_per_epoch=${MAX_STEPS}"
  --override "optimization.per_device_batch_size=${FINETUNE_BATCH_PER_GPU}"
  --override "optimization.num_workers=${FINETUNE_NUM_WORKERS}"
  --override "optimization.dtype=${FINETUNE_DTYPE}"
  --override "optimization.encoder_learning_rate=${ENCODER_LR:-0.0}"
  --override "optimization.decoder_learning_rate=${DECODER_LR:-0.00002}"
  --override "distributed.required_world_size=${NPROC_PER_NODE}"
  --override "distributed.global_batch_size=${GLOBAL_BATCH_SIZE}"
  --override "checkpoint.auto_resume=${AUTO_RESUME}"
  --override "optimization.validation_every_epochs=${FINETUNE_VALIDATION_EVERY:-1}"
)
if [[ -n "${FINETUNE_TRAIN_SOURCE_MANIFEST:-}" ]]; then
  OVERRIDES+=(--override "data.train_source_manifest=${FINETUNE_TRAIN_SOURCE_MANIFEST}" --override "loss.use_class_weights=false")
fi
if [[ -n "${FINETUNE_EXPECTED_TRAIN_TILES:-}" ]]; then
  OVERRIDES+=(--override "data.expected_tile_counts.train=${FINETUNE_EXPECTED_TRAIN_TILES}")
fi
if [[ -n "${FINETUNE_RESUME:-}" ]]; then
  OVERRIDES+=(--override "checkpoint.resume=${FINETUNE_RESUME}")
fi

echo "Phase: ${PHASE}"
echo "GPUs: ${GPU_IDS} (${NPROC_PER_NODE} workers)"
echo "Encoder: frozen EMA target encoder"
echo "Decoder: ${FINETUNE_DECODER_LABEL}, width ${FINETUNE_DECODER_DIM}, dropout 0.1"
echo "Training: ${EPOCHS} epochs, batch/GPU ${FINETUNE_BATCH_PER_GPU}, global batch ${GLOBAL_BATCH_SIZE}"
echo "Loss: 1.0 CE + 1.0 Dice + 0.5 focal (class weighted); precision: ${FINETUNE_DTYPE} AMP"
echo "Schedule: cosine, ${WARMUP_EPOCHS}-epoch warm-up; decoder LR ${DECODER_LR:-0.00002}"
echo "Pretrained encoder: ${FLOOD_JEPA_PRETRAINED}"
echo "Output: ${FINETUNE_OUTPUT}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m flood_jepa.training.segmentation \
  --config "${CONFIG}" \
  "${OVERRIDES[@]}" 2>&1 | tee -a "${FINETUNE_OUTPUT}/logs/training.log"

if [[ "${PHASE}" == "train-test" ]]; then
  SEGMENTER_CHECKPOINT="${FINETUNE_OUTPUT}/checkpoints/best.pt"
  if [[ ! -f "${SEGMENTER_CHECKPOINT}" ]]; then
    echo "Training completed without the expected best checkpoint: ${SEGMENTER_CHECKPOINT}" >&2
    exit 1
  fi
  echo "Training complete. Evaluating validation-selected best.pt on validation and test."
  FLOOD_JEPA_SEGMENTER="${SEGMENTER_CHECKPOINT}" \
  EVAL_SPLIT=val \
  GPU_IDS="${GPU_IDS}" \
  EVAL_WORLD_SIZE="${NPROC_PER_NODE}" \
  EVAL_GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
  EVAL_EXPERIMENT_NAME="${FINETUNE_EXPERIMENT_NAME}" \
  METRICS_OUTPUT="${FINETUNE_OUTPUT}/metrics/val_best_metrics.json" \
  PER_SAMPLE_OUTPUT="${FINETUNE_OUTPUT}/metrics/val_best_per_sample.csv" \
  LOG_OUTPUT="${FINETUNE_OUTPUT}/logs/validation_best.log" \
    "${PROJECT_ROOT}/scripts/evaluate_worldfloods.sh"
  FLOOD_JEPA_SEGMENTER="${FINETUNE_OUTPUT}/checkpoints/last.pt" \
  EVAL_SPLIT=val \
  GPU_IDS="${GPU_IDS}" \
  EVAL_WORLD_SIZE="${NPROC_PER_NODE}" \
  EVAL_GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
  EVAL_EXPERIMENT_NAME="${FINETUNE_EXPERIMENT_NAME}" \
  METRICS_OUTPUT="${FINETUNE_OUTPUT}/metrics/val_latest_metrics.json" \
  PER_SAMPLE_OUTPUT="${FINETUNE_OUTPUT}/metrics/val_latest_per_sample.csv" \
  LOG_OUTPUT="${FINETUNE_OUTPUT}/logs/validation_latest.log" \
    "${PROJECT_ROOT}/scripts/evaluate_worldfloods.sh"
  FLOOD_JEPA_SEGMENTER="${SEGMENTER_CHECKPOINT}" \
  EVAL_SPLIT=test \
  GPU_IDS="${GPU_IDS}" \
  EVAL_WORLD_SIZE="${NPROC_PER_NODE}" \
  EVAL_GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
  EVAL_EXPERIMENT_NAME="${FINETUNE_EXPERIMENT_NAME}" \
  METRICS_OUTPUT="${FINETUNE_OUTPUT}/metrics/test_metrics.json" \
  PER_SAMPLE_OUTPUT="${FINETUNE_OUTPUT}/metrics/test_per_sample.csv" \
  LOG_OUTPUT="${FINETUNE_OUTPUT}/logs/testing.log" \
    "${PROJECT_ROOT}/scripts/evaluate_worldfloods.sh"
fi
