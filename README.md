# Flood-JEPA

Flood-JEPA adapts Meta's V-JEPA 2.1 dense self-supervised recipe to multispectral
flood imagery. The first experiment pretrains on label-free WorldFloods V2 train
imagery at 256 x 256 pixels. It uses all ten stored Sentinel-2 channels: B2, B3,
B4, B5, B6, B7, B8, B8A, B11, and B12.

The upstream implementation is cloned at `third_party/vjepa2` and pinned to
commit `204698b45b3712590f06245fbfba32d3be539812`. Flood-JEPA imports its encoder,
predictor, mask generator, and mask utilities rather than maintaining rewritten
copies.

## Initial architecture

```text
WorldFloods [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12], 10 x 256 x 256
                   |
      shared initialization, then EMA updates
          +--------+--------+
          |                 |
  context ViT-B/16     target ViT-B/16
    visible tokens       all tokens, no grad
          |                 |
          +---- predictor --+
              latent L1

downstream only:
  context encoder levels 3/6/9/12 -> feature-fusion decoder -> land/water logits
```

- Context encoder: official V-JEPA 2.1 image ViT-B/16, 12 layers, 768 dimensions,
  12 heads, RoPE, and a ten-channel `Conv2d` patch projection. A 256-pixel input
  produces a 16 x 16 grid (256 tokens).
- Target encoder: an initially identical ViT-B/16 with gradients disabled and
  weights updated by EMA (`0.99925`). Four normalized layer outputs are the
  prediction targets.
- Predictor: official V-JEPA 2.1 ViT predictor, 12 layers, 384 dimensions, 12
  heads, learned mask tokens, deep self-supervision, and prediction of both
  masked and visible-context tokens.
- Segmentation head: kept outside pretraining. It reshapes features from encoder
  layers 3, 6, 9, and 12 into 16 x 16 maps, projects each to 256 channels, fuses
  them, and upsamples two-class logits to 256 x 256.

See [the architecture decision](docs/architecture.md) for the band-embedding
analysis and the WorldFloods-to-UFO protocol.

## Setup

Use a dedicated Python 3.11 or 3.12 cloud environment, as required by the cloned
upstream package, then run:

```bash
./scripts/setup_environment.sh
export WORLDFLOODS_LMDB_DIR=/path/to/worldfloods_lmdb
./scripts/verify_worldfloods.sh
```

The LMDB reader verifies that the input is the `train` split and deliberately
does not decode or return segmentation labels during self-supervised pretraining.

Launch on the eight-GPU server:

```bash
NPROC_PER_NODE=8 ./scripts/pretrain.sh
```

For a one-GPU smoke run, set `NPROC_PER_NODE=1` and temporarily lower the batch
size, worker count, and epoch count in a copied config. Checkpoints and JSONL
training logs are written below `results/`.

After pretraining, fit the segmentation encoder/decoder on WorldFloods train
labels and select by validation mIoU:

```bash
export FLOOD_JEPA_PRETRAINED="$PWD/results/vjepa21_vitb16_worldfloods_10band_256/seed_42/checkpoints/latest.pt"
NPROC_PER_NODE=8 ./scripts/finetune_worldfloods.sh
```

Only after selection, evaluate the chosen checkpoint on the official test split:

```bash
export FLOOD_JEPA_SEGMENTER="$PWD/results/flood_jepa_worldfloods_segmentation_10band/seed_42/best.pt"
./scripts/evaluate_worldfloods.sh
```

Validation and test tiles are deterministically partitioned into four 256 x 256
quadrants, so evaluation uses the requested input size without dropping parts of
the original 512 x 512 tiles. Pixels labeled 255 are excluded from every metric.

## Experiment variants

- `configs/pretrain_vitb16_256_10band.yaml` is the primary WorldFloods experiment
  and the default used by `scripts/pretrain.sh`.
- `configs/pretrain_vitb16_256_4band.yaml` is retained as the common-band
  WorldFloods-to-UFO transfer baseline.

Select a different config without editing the launcher:

```bash
FLOOD_JEPA_CONFIG="$PWD/configs/pretrain_vitb16_256_4band.yaml" ./scripts/pretrain.sh
```

## Split policy

1. Self-supervised pretraining: WorldFloods official train split, images only.
2. Segmentation fitting/model selection: WorldFloods train labels and validation
   split after pretraining. The predictor and EMA target are discarded.
3. In-domain report: WorldFloods test, evaluated once after model selection.
4. Cross-sensor report: use either a separately pretrained common four-band
   model or a frozen, train-only sensor adapter/distilled four-band student. A
   ten-channel model cannot directly consume four-channel UFO tensors.

The future UFO evaluation must not use UFO labels to train the bridge. It should
report both a 10 m/px harmonized condition and a native-resolution stress test.
A 256-pixel crop covers different physical areas at Sentinel-2 and PlanetScope
native resolutions, so silently mixing native scales would confound sensor
generalization with scale shift.

## Local verification

```bash
python -m compileall -q src tests
bash -n scripts/*.sh
pytest -q
```

The final command needs the project dependencies installed. Dataset integration
checks additionally need access to the cloud LMDBs.
