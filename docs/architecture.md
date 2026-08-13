# Flood-JEPA architecture decision

## Decision

The first model is **Flood-JEPA ViT-B/16-256**, not the earlier proposed
Flood-JEPA-S/8. It retains the V-JEPA 2.1 ViT-B encoder and predictor topology and
changes only the image patch projection from three to ten input channels. The
channels are `[B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]`, in the exact order
stored in the WorldFloods V2 LMDB.

At 256 x 256 with patch size 16, each image becomes 256 spatial tokens. Both
encoders use 12 transformer blocks, width 768, 12 heads, and rotary positional
embeddings. The target encoder receives all tokens and is updated only by EMA.
The context encoder receives the visible subset. The 12-block, width-384,
12-head predictor consumes concatenated context features from encoder blocks
3, 6, 9, and 12 and predicts the similarly concatenated target features.

The V-JEPA 2.1 additions are retained:

- dense predictive loss over masked tokens and visible context tokens;
- deep self-supervision from four encoder levels;
- distance weighting for the visible-context loss;
- learned predictor mask tokens.

The segmentation decoder is downstream-only. It is not attached during JEPA
pretraining and therefore cannot leak WorldFloods labels into the representation
objective.

## Why there is no per-band embedding in version 1

V-JEPA 2.1's `e_img` and `e_video` identify different tokenizers/modalities. Each
is one learned vector broadcast across every token of an image or video stream.
They do not identify RGB channels.

In Flood-JEPA v1, a patch is produced by one convolution with shape
`[768, 10, 16, 16]`. Its weights are already distinct for every Sentinel-2 band,
so band identity is encoded by fixed channel position before the predictor ever
sees a token. After that projection, every token is a fusion of all ten bands;
there is no individual "B2 token" to which a B2 embedding could be added.

Potential advantages of explicit band embeddings in a later spectral-token
architecture are:

- clear band identity when bands are tokenized separately;
- easier missing-band masking and flexible sensor subsets;
- a clean place to attach wavelength or spectral-response metadata;
- better extensibility to SAR, elevation, weather, and temporal sensors.

The disadvantages in the current channel-stacked architecture are:

- adding a constant per channel before a linear convolution is largely
  equivalent to changing a bias and may be removed by normalization;
- adding ten embeddings after patch fusion has no well-defined band-token
  correspondence;
- it departs from the requested V-JEPA 2.1 topology without a testable need;
- it can encourage sensor-specific shortcuts and hurt WorldFloods-to-UFO
  transfer.

Therefore `modality_embedding: false` is enforced for this experiment. When SAR
or other modalities are added, the recommended extension is one tokenizer or
small adapter per modality, followed by a modality/sensor embedding and a common
latent width. Band tokens with wavelength encodings should be evaluated as a
separate ablation, not silently added to this baseline.

## Sensor and scale protocol

The ten-band model maximizes the spectral information available during the
WorldFloods experiment, but its input shape is not compatible with four-band
PlanetScope UFO. Sentinel-2 and PlanetScope also differ in ground sampling
distance, spectral response, radiometry, atmosphere/cloud behavior, and scene
composition.

Before the UFO experiment, choose and freeze one explicit bridge using only
WorldFloods or other non-UFO training data:

- train the provided common four-band Flood-JEPA baseline;
- distill the ten-band encoder into a four-band student; or
- learn a four-band sensor adapter feeding the frozen ten-band representation.

The primary UFO result should then:

1. reorder PlanetScope channels to blue, green, red, NIR;
2. convert both datasets to a documented reflectance representation;
3. resample UFO to the same 10 m/px grid used by WorldFloods before creating
   256 x 256 crops;
4. apply WorldFloods-derived normalization without fitting anything on UFO;
5. evaluate the frozen selected model with no UFO labels used for tuning.

Also report native-resolution UFO as a stress test. Keeping these two conditions
separate makes the result interpretable: the harmonized score emphasizes sensor
and geography transfer, while the native score also contains a strong scale
shift.

## Comparisons and ablations

Use the same band set, crop policy, supervised labels, decoder capacity, and
model-selection split wherever the backbone permits it. At minimum report:

| Experiment | Purpose |
| --- | --- |
| Randomly initialized ViT-B/16 + decoder | Measures the value of JEPA pretraining |
| ImageNet-supervised ViT-B/16 + decoder | Conventional representation baseline |
| MAE ViT-B/16 + decoder | Pixel-reconstruction SSL baseline |
| DINO-family ViT + decoder | Discriminative feature SSL baseline |
| Flood-JEPA ten-band | Main WorldFloods model using all available channels |
| Flood-JEPA four-band | Common-channel UFO transfer baseline |
| Flood-JEPA without dense context loss | V-JEPA 2.1 objective ablation |
| Flood-JEPA final layer only | Deep-self-supervision ablation |

For flood segmentation, report water IoU, macro IoU, F1/Dice, precision, recall,
and per-scene statistics. Treat label 255 as ignore. Select checkpoints only on
WorldFloods validation; keep WorldFloods test and UFO untouched until the
protocol is fixed.

## Upstream references

- V-JEPA 2.1 paper: <https://arxiv.org/abs/2603.14482>
- Official code: <https://github.com/facebookresearch/vjepa2>
- Original I-JEPA paper: <https://arxiv.org/abs/2301.08243>
- Original V-JEPA paper: <https://arxiv.org/abs/2404.08471>
