#!/usr/bin/env bash
set -euo pipefail

# Rank epochs 10..100 using identical frozen, final-layer 1x1 linear probes.
# This selection stage uses WorldFloods train and validation only—never test.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WORLDFLOODS_LMDB_DIR:?Set WORLDFLOODS_LMDB_DIR to the WorldFloods LMDB directory}"

GPU_IDS="${GPU_IDS:-0,2,3,4,5,6}"
IFS=',' read -r -a SELECTED_GPUS <<< "${GPU_IDS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#SELECTED_GPUS[@]}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PRETRAIN_DIR="${PRETRAIN_DIR:-${PROJECT_ROOT}/results/vjepa21_vitb16_worldfloods_10band_256/seed_42/checkpoints}"
SELECTION_ROOT="${SELECTION_ROOT:-${PROJECT_ROOT}/results/worldfloods_pretrain_weight_selection}"
PROBE_CONFIG="${PROBE_CONFIG:-${PROJECT_ROOT}/configs/select_pretrain_worldfloods_linear_probe.yaml}"
PROBE_EPOCHS="${PROBE_EPOCHS:-15}"
PROBE_BATCH_PER_GPU="${PROBE_BATCH_PER_GPU:-8}"
PROBE_WORKERS="${PROBE_WORKERS:-8}"
PROBE_LR="${PROBE_LR:-0.0001}"
PROBE_MAX_STEPS_PER_EPOCH="${PROBE_MAX_STEPS_PER_EPOCH:-null}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-010 020 030 040 050 060 070 080 090 100}"
SEED="${SEED:-42}"

mkdir -p "${SELECTION_ROOT}"

for epoch in ${CHECKPOINT_EPOCHS}; do
  pretrain_checkpoint="${PRETRAIN_DIR}/epoch_${epoch}.pt"
  probe_output="${SELECTION_ROOT}/epoch_${epoch}/seed_${SEED}"
  best_probe="${probe_output}/best.pt"
  completion_marker="${probe_output}/selection_complete.marker"

  if [[ ! -f "${pretrain_checkpoint}" ]]; then
    echo "Missing pretraining checkpoint: ${pretrain_checkpoint}" >&2
    exit 1
  fi

  echo "=== Pretraining epoch ${epoch}: frozen linear probe ==="
  if [[ -f "${best_probe}" && -f "${completion_marker}" ]]; then
    echo "Already complete; using ${best_probe}"
    continue
  fi

  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m flood_jepa.training.segmentation \
    --config "${PROBE_CONFIG}" \
    --override "experiment.seed=${SEED}" \
    --override "experiment.output_dir=${probe_output}" \
    --override "model.pretrained_checkpoint=${pretrain_checkpoint}" \
    --override "model.encoder_source=target" \
    --override "model.freeze_encoder=true" \
    --override "model.decoder_type=linear" \
    --override "model.use_activation_checkpointing=false" \
    --override "optimization.epochs=${PROBE_EPOCHS}" \
    --override "optimization.per_device_batch_size=${PROBE_BATCH_PER_GPU}" \
      --override "optimization.num_workers=${PROBE_WORKERS}" \
      --override "optimization.max_steps_per_epoch=${PROBE_MAX_STEPS_PER_EPOCH}" \
      --override "optimization.encoder_learning_rate=0.0" \
      --override "optimization.decoder_learning_rate=${PROBE_LR}"
  touch "${completion_marker}"
done

SELECTION_ROOT="${SELECTION_ROOT}" PRETRAIN_DIR="${PRETRAIN_DIR}" SEED="${SEED}" CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS}" python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["SELECTION_ROOT"])
pretrain_dir = Path(os.environ["PRETRAIN_DIR"]).resolve()
seed = int(os.environ["SEED"])
checkpoint_epochs = [int(value) for value in os.environ["CHECKPOINT_EPOCHS"].split()]
rows = []
for pretrain_epoch in checkpoint_epochs:
    run_dir = root / f"epoch_{pretrain_epoch:03d}" / f"seed_{seed}"
    records = [json.loads(line) for line in (run_dir / "training.jsonl").read_text().splitlines() if line]
    best = max(records, key=lambda item: float(item["validation"]["miou"]))
    validation = best["validation"]
    rows.append(
        {
            "pretrain_epoch": pretrain_epoch,
            "probe_best_epoch": int(best["epoch"]),
            "validation_miou": float(validation["miou"]),
            "validation_land_iou": float(validation["land_iou"]),
            "validation_water_iou": float(validation["water_iou"]),
            "validation_water_precision": float(validation["water_precision"]),
            "validation_water_recall": float(validation["water_recall"]),
            "validation_water_f1": float(validation["water_f1"]),
            "pretrain_checkpoint": str(pretrain_dir / f"epoch_{pretrain_epoch:03d}.pt"),
            "probe_checkpoint": str((run_dir / "best.pt").resolve()),
        }
    )

rows.sort(key=lambda item: item["validation_miou"], reverse=True)
destination = root / "validation_checkpoint_ranking.csv"
with destination.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

winner = rows[0]
(root / "best_pretrain_checkpoint.txt").write_text(
    winner["pretrain_checkpoint"] + "\n", encoding="utf-8"
)
(root / "selection_summary.json").write_text(
    json.dumps(
        {
            "selection_split": "validation",
            "encoder_source": "target",
            "encoder_frozen": True,
            "probe": "final-layer 1x1 linear classifier",
            "winner": winner,
            "ranking": rows,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print("\nValidation ranking:")
for rank, row in enumerate(rows, 1):
    print(
        f"{rank:2d}. epoch_{row['pretrain_epoch']:03d}.pt  "
        f"mIoU={row['validation_miou']:.6f}  "
        f"water-IoU={row['validation_water_iou']:.6f}"
    )
print(f"\nSelected checkpoint: {winner['pretrain_checkpoint']}")
print(f"Ranking CSV: {destination}")
PY
