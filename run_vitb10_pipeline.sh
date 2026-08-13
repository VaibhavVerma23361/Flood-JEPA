#!/usr/bin/env bash
set -euo pipefail

# Flood-JEPA ViT-B/16 self-supervised pretraining using all ten WorldFloods bands.
# Usage: ./scripts/run_vitb10_pipeline.sh {smoke|pretrain}
# Every hyperparameter can be overridden through an environment variable.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-pretrain}"

# Hardware and paths.
GPU_IDS="${GPU_IDS:-0,2,3,4,5,6}"
IFS=',' read -r -a SELECTED_GPUS <<< "${GPU_IDS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#SELECTED_GPUS[@]}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
: "${WORLDFLOODS_LMDB_DIR:?Set WORLDFLOODS_LMDB_DIR to the WorldFloods LMDB directory}"

PRETRAIN_CONFIG="${PROJECT_ROOT}/configs/pretrain_vitb16_256_10band.yaml"

# Shared data hyperparameters.
SEED="${SEED:-42}"
CROP_SIZE="${CROP_SIZE:-256}"
INPUT_BANDS="${INPUT_BANDS:-[B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]}"
INPUT_CHANNELS="${INPUT_CHANNELS:-10}"
NORMALIZATION="${NORMALIZATION:-worldfloods}"
AUGMENT_D4="${AUGMENT_D4:-true}"

# JEPA architecture hyperparameters.
MODEL_NAME="${MODEL_NAME:-vit_base}"
PATCH_SIZE="${PATCH_SIZE:-16}"
ENCODER_DIM="${ENCODER_DIM:-768}"
PREDICTOR_DEPTH="${PREDICTOR_DEPTH:-12}"
PREDICTOR_DIM="${PREDICTOR_DIM:-384}"
PREDICTOR_HEADS="${PREDICTOR_HEADS:-12}"
DEEP_SUPERVISION_LEVELS="${DEEP_SUPERVISION_LEVELS:-4}"
USE_ROPE="${USE_ROPE:-true}"
USE_SDPA="${USE_SDPA:-true}"
UNIFORM_POWER="${UNIFORM_POWER:-true}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
MODALITY_EMBEDDING="${MODALITY_EMBEDDING:-false}"

# Masking and latent-loss hyperparameters.
MASK_ASPECT_RATIO="${MASK_ASPECT_RATIO:-[0.75,1.5]}"
MASK_NUM_BLOCKS="${MASK_NUM_BLOCKS:-10}"
MASK_SPATIAL_SCALE="${MASK_SPATIAL_SCALE:-[0.15,0.15]}"
MASK_TEMPORAL_SCALE="${MASK_TEMPORAL_SCALE:-[1.0,1.0]}"
MASK_FULL_COMPLEMENT="${MASK_FULL_COMPLEMENT:-false}"
MASK_MAX_TEMPORAL_KEEP="${MASK_MAX_TEMPORAL_KEEP:-1.0}"
MASK_MAX_KEEP="${MASK_MAX_KEEP:-null}"
LOSS_EXPONENT="${LOSS_EXPONENT:-1.0}"
PREDICT_ALL="${PREDICT_ALL:-true}"
DISTANCE_WEIGHTED_CONTEXT="${DISTANCE_WEIGHTED_CONTEXT:-true}"
CONTEXT_WEIGHT="${CONTEXT_WEIGHT:-0.5}"
CONTEXT_WARMUP_START="${CONTEXT_WARMUP_START:-0.05}"
CONTEXT_WARMUP_END="${CONTEXT_WARMUP_END:-0.10}"

# Pretraining optimization hyperparameters.
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_BATCH_PER_GPU="${PRETRAIN_BATCH_PER_GPU:-16}"
PRETRAIN_WORKERS="${PRETRAIN_WORKERS:-8}"
PRETRAIN_LR="${PRETRAIN_LR:-0.0002}"
PRETRAIN_START_LR="${PRETRAIN_START_LR:-0.000001}"
PRETRAIN_FINAL_LR="${PRETRAIN_FINAL_LR:-0.000001}"
PRETRAIN_WARMUP_EPOCHS="${PRETRAIN_WARMUP_EPOCHS:-10}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-0.04}"
PRETRAIN_FINAL_WEIGHT_DECAY="${PRETRAIN_FINAL_WEIGHT_DECAY:-0.4}"
EMA_START="${EMA_START:-0.99925}"
EMA_END="${EMA_END:-0.99925}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"
PRETRAIN_DTYPE="${PRETRAIN_DTYPE:-float16}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PRETRAIN_MAX_STEPS_PER_EPOCH="${PRETRAIN_MAX_STEPS_PER_EPOCH:-null}"
MINIMUM_NONPADDING_FRACTION="${MINIMUM_NONPADDING_FRACTION:-0.25}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-10}"
PRETRAIN_RESUME="${PRETRAIN_RESUME:-null}"
PRETRAIN_OUTPUT="${PRETRAIN_OUTPUT:-${PROJECT_ROOT}/results/vjepa21_vitb16_worldfloods_10band_256/seed_${SEED}}"

set_pretrain_overrides() {
  PRETRAIN_OVERRIDES=(
    --override "experiment.seed=${SEED}"
    --override "experiment.output_dir=${PRETRAIN_OUTPUT}"
    --override "data.crop_size=${CROP_SIZE}"
    --override "data.input_bands=${INPUT_BANDS}"
    --override "data.normalization=${NORMALIZATION}"
    --override "data.augment_d4=${AUGMENT_D4}"
    --override "data.minimum_nonpadding_fraction=${MINIMUM_NONPADDING_FRACTION}"
    --override "model.model_name=${MODEL_NAME}"
    --override "model.input_channels=${INPUT_CHANNELS}"
    --override "model.patch_size=${PATCH_SIZE}"
    --override "model.encoder_dim=${ENCODER_DIM}"
    --override "model.predictor_depth=${PREDICTOR_DEPTH}"
    --override "model.predictor_dim=${PREDICTOR_DIM}"
    --override "model.predictor_heads=${PREDICTOR_HEADS}"
    --override "model.deep_self_supervision_levels=${DEEP_SUPERVISION_LEVELS}"
    --override "model.use_rope=${USE_ROPE}"
    --override "model.use_sdpa=${USE_SDPA}"
    --override "model.uniform_power=${UNIFORM_POWER}"
    --override "model.use_activation_checkpointing=${ACTIVATION_CHECKPOINTING}"
    --override "model.modality_embedding=${MODALITY_EMBEDDING}"
    --override "mask.0.aspect_ratio=${MASK_ASPECT_RATIO}"
    --override "mask.0.full_complement=${MASK_FULL_COMPLEMENT}"
    --override "mask.0.max_temporal_keep=${MASK_MAX_TEMPORAL_KEEP}"
    --override "mask.0.max_keep=${MASK_MAX_KEEP}"
    --override "mask.0.num_blocks=${MASK_NUM_BLOCKS}"
    --override "mask.0.spatial_scale=${MASK_SPATIAL_SCALE}"
    --override "mask.0.temporal_scale=${MASK_TEMPORAL_SCALE}"
    --override "loss.exponent=${LOSS_EXPONENT}"
    --override "loss.predict_all=${PREDICT_ALL}"
    --override "loss.distance_weighted_context=${DISTANCE_WEIGHTED_CONTEXT}"
    --override "loss.context_weight=${CONTEXT_WEIGHT}"
    --override "loss.context_warmup_start_fraction=${CONTEXT_WARMUP_START}"
    --override "loss.context_warmup_end_fraction=${CONTEXT_WARMUP_END}"
    --override "optimization.epochs=${PRETRAIN_EPOCHS}"
    --override "optimization.per_device_batch_size=${PRETRAIN_BATCH_PER_GPU}"
    --override "optimization.num_workers=${PRETRAIN_WORKERS}"
    --override "optimization.learning_rate=${PRETRAIN_LR}"
    --override "optimization.start_learning_rate=${PRETRAIN_START_LR}"
    --override "optimization.final_learning_rate=${PRETRAIN_FINAL_LR}"
    --override "optimization.warmup_epochs=${PRETRAIN_WARMUP_EPOCHS}"
    --override "optimization.weight_decay=${PRETRAIN_WEIGHT_DECAY}"
    --override "optimization.final_weight_decay=${PRETRAIN_FINAL_WEIGHT_DECAY}"
    --override "optimization.ema_start=${EMA_START}"
    --override "optimization.ema_end=${EMA_END}"
    --override "optimization.gradient_clip_norm=${GRADIENT_CLIP_NORM}"
    --override "optimization.dtype=${PRETRAIN_DTYPE}"
    --override "optimization.pin_memory=${PIN_MEMORY}"
    --override "optimization.persistent_workers=${PERSISTENT_WORKERS}"
    --override "optimization.max_steps_per_epoch=${PRETRAIN_MAX_STEPS_PER_EPOCH}"
    --override "checkpoint.save_every_epochs=${SAVE_EVERY_EPOCHS}"
    --override "checkpoint.resume=${PRETRAIN_RESUME}"
  )
}

print_run() {
  echo "Phase: ${PHASE}"
  echo "Project: ${PROJECT_ROOT}"
  echo "Conda environment: ${CONDA_DEFAULT_ENV:-none}"
  echo "Physical GPUs: ${GPU_IDS} (${NPROC_PER_NODE} workers)"
  echo "WorldFloods LMDB: ${WORLDFLOODS_LMDB_DIR}"
  echo "Model: ${MODEL_NAME}, ${INPUT_CHANNELS} bands, ${CROP_SIZE}px, patch ${PATCH_SIZE}"
  echo "Predictor: depth ${PREDICTOR_DEPTH}, width ${PREDICTOR_DIM}, heads ${PREDICTOR_HEADS}"
  echo "Pretrain: ${PRETRAIN_EPOCHS} epochs, batch/GPU ${PRETRAIN_BATCH_PER_GPU}, global batch $((PRETRAIN_BATCH_PER_GPU * NPROC_PER_NODE)), LR ${PRETRAIN_LR}, ${PRETRAIN_DTYPE}"
}

verify_pretrain_installation() {
  if ! grep -q -- '"--override"' "${PROJECT_ROOT}/src/flood_jepa/training/pretrain.py" || \
     ! grep -q -- 'def apply_overrides' "${PROJECT_ROOT}/src/flood_jepa/config.py"; then
    echo "The launcher and Python sources are from different Flood-JEPA versions." >&2
    echo "Install the complete pretraining update archive before launching." >&2
    exit 2
  fi
}

run_pretrain() {
  verify_pretrain_installation
  set_pretrain_overrides
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m flood_jepa.training.pretrain \
    --config "${PRETRAIN_CONFIG}" \
    "${PRETRAIN_OVERRIDES[@]}"
}

cd "${PROJECT_ROOT}"

case "${PHASE}" in
  smoke)
    PRETRAIN_EPOCHS=1
    PRETRAIN_BATCH_PER_GPU="${SMOKE_BATCH_PER_GPU:-2}"
    PRETRAIN_WORKERS="${SMOKE_WORKERS:-2}"
    PRETRAIN_WARMUP_EPOCHS=0
    PRETRAIN_MAX_STEPS_PER_EPOCH="${SMOKE_STEPS:-5}"
    PRETRAIN_OUTPUT="${PROJECT_ROOT}/results/smoke_vitb16_worldfloods_10band/seed_${SEED}"
    GPU_IDS="${SELECTED_GPUS[0]}"
    CUDA_VISIBLE_DEVICES="${SELECTED_GPUS[0]}"
    NPROC_PER_NODE=1
    print_run
    run_pretrain
    ;;
  pretrain)
    print_run
    run_pretrain
    ;;
  *)
    echo "Usage: $0 {smoke|pretrain}" >&2
    exit 2
    ;;
esac
