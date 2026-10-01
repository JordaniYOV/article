# Frozen encoder probe on orbital crater imagery

Status: input preparation and admission audit, 30 September 2026. All 120
selected native WAC rasters have passed the recorded source SHA-256 checks.
All 120 decoded TIFFs have shape 512×512×5 on the preparation host.
No encoder weights have been loaded and no head has been trained. This protocol supersedes the
complete-system comparison as the **proposed primary research design**. The
complete-system comparison remains a separate secondary analysis.

## Research claim

The experiment measures the utility of frozen spatial visual features for a
specified downstream task under a fixed probe protocol. It does not measure
the full SpaceLLaVA's language reasoning, and a score difference alone cannot
isolate the causal effect of lunar pretraining because architecture, training
corpus and native input modalities differ. The Remyx configuration names
`openai/clip-vit-large-patch14-336` as its vision tower and sets
`unfreeze_mm_vision_tower=false`. The actual tower tensors must be compared
with the corresponding base CLIP revision before describing it as unchanged.
The checkpoint index locates all 391 visual-tower tensors in shard 6, so the
prepared extractor targets the **SpaceLLaVA visual tower itself** and excludes
the language model and multimodal projector. Its loading and outputs still need
a GPU smoke test.

Sources: [SpaceLLaVA config](https://huggingface.co/remyxai/SpaceLLaVA/blob/main/config.json),
[IBM LFM card](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model).

## Initial task and assets

Use the selected SomBench WAC crater catalog polygons as a binary spatial
target. The active collection has 50 publisher-train images from
`data_orbital/orbital_300_v3/moon/train/`, 39 publisher-val images and 31
publisher-test images. The two heads are trained **only** on the 50 train
images and their crater masks. The val/test set combines 20 selected WAC
segmentation tiles with 50 distinct WAC elevation-track tiles. Publisher COCO
polygons are rasterized for the latter; the elevation maps are never used as
image-only inputs. Every image has a source scene/product group and image/mask
hashes. `linear_head_approach.prepare` checks these assets and writes
separate model-visible input and target manifests beneath
`linear_head_approach/outputs/encoder_probe_wac_v1/`. The input manifest contains no mask or answer.
Elevation arrays are not inputs for this image-only probe.

Preparation command (using the project's Python environment):

```powershell
& .\.venv-benchmark\Scripts\python.exe -m linear_head_approach.prepare
```

`linear_head_approach.acquire_wac` retrieves only selected native WAC
TIFFs from the pinned revision and checks each against the source-raster hash.
It writes `data_orbital/encoder_probe_wac_v1/native_vis/selected_rasters.jsonl`.
Do not use the COCO annotation files as model inputs. After feature extractors
are smoke tested, each extractor should write a `features.jsonl` with exactly
one row per probe sample: `sample_id`, `split`, `scene_group_id`,
`rgb_image_sha256`, `feature`, `feature_sha256`, `extractor_id`,
`model_revision`, and `preprocessing_id`. Each `feature` must be a finite
H×W×C NumPy array under `linear_head_approach/outputs/`, using spatial encoder tokens before any
task head. The shared ridge probe in `linear_head_approach.fit_head` fits
on train and selects its threshold on val; a separate `score-test` action opens
test masks only after fitting. Its synthetic unit test is a software check,
not a scientific model result. After both test reports exist,
`linear_head_approach.compare` computes the **paired** per-image IoU
difference with scene-group bootstrap intervals; separate confidence intervals
for each model are insufficient to assess their difference.

For CLIP, record the layer specified by SpaceLLaVA's
`mm_vision_select_layer=-2`, remove its class token and verify the resulting
patch grid. For IBM, use the official TerraTorch backbone with `vis` only,
record the selected encoder block and patch size, and remove register tokens
using the upstream interface before reshaping spatial features. The published
WAC crater configuration uses 512×512 input, patch size 8, and a 64×64 token
grid; its selected backbone indices are `[2, 5, 8, 11]`. A first protocol
choice is the final encoder block (`11`) for IBM and `-2` for CLIP, with a
layer-matched sensitivity analysis if resources allow. Refuse any unexpected
token count rather than guessing grid dimensions.

For IBM's native WAC arm, follow the exact WAC data adapter: read the five-band
TIFF as float32, replace NaNs with zero, resize each channel to 512×512 using
PIL bilinear interpolation, then normalize with means
`[0.018106, 0.025262, 0.027113, 0.028616, 0.030757]` and standard deviations
`[0.010051, 0.013646, 0.014491, 0.015267, 0.016555]`. Record the upstream
source commit and checkpoint hash. This is based on the
[published WAC configuration](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/main/terratorch_integration/configs/wac_craters/full_data/ni_lfm_ps8_frozen.yaml)
and [data adapter](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/main/terratorch_integration/data_adapter.py),
not the repository's generic example settings. Check these operations with a
one-image smoke test before full feature extraction.

The existing three-channel PNGs are rendered from bands 0–2 with a per-tile
percentile stretch. They are suitable for the CLIP arm. IBM's published WAC
`vis` tokenizer expects five bands. Recover only the selected source WAC rasters
at the pinned revision and verify their SHA-256 against the original raster
hashes in the input manifest. Do not duplicate RGB channels or substitute
height maps. Confirm the exact official IBM preprocessing and feature tensor
with a one-image smoke test. Run the CLIP tower smoke test separately.

## Comparison contract

1. Extract spatial tokens **before any task head** with both encoders frozen.
   Pin checkpoint and processor revisions, feature layer, image transform,
   numeric precision, and feature-grid geometry. Check whether the SpaceLLaVA
   tower differs from base CLIP. No LLM output enters the probe.
2. Fit the same simple spatial head family to each feature set using only
   publisher-train images. Predefine feature normalization, class weighting,
   regularization, interpolation and a common 24×24 output grid. The feature widths
   differ, so report trainable parameter counts and include a capacity-matched
   sensitivity analysis if needed.
3. Select any head hyperparameters and mask threshold with publisher-val only.
   Freeze the entire rule before opening publisher-test predictions. Keep
   `scene_group_id` intact and audit geographic and pretraining overlap.
4. Report per-image crater IoU and Dice, an all-foreground baseline, failure
   examples, and uncertainty grouped by product/scene. Do not treat individual
   pixels as independent samples. Report train, val and test separately.

The 31 selected test images permit a preliminary grouped estimate, but a
stable article-level ranking requires uncertainty analysis, model-training
membership audit and preferably a larger independent WAC test set or an
independent NAC replication. Native five-band IBM input versus three-band CLIP
input also changes information available to each model. Report this explicitly
as a native-modality comparison; a strict same-information encoder claim needs
a validated shared-input protocol. The existing IBM crater-detection head is
not part of this frozen-encoder probe.

Mars MMLSv2 train masks and the selected Mars segmentation evaluation masks
have different label semantics. They are not pooled into this crater task.
The route and measured-geometry experiments remain gated on appropriate truth.
