# Orbital Mars/Moon dataset v1

Collected 29 September 2026 for an orbital-imagery research track. This directory
is independent of `data_rover/`; no rover asset was copied, converted, or relabeled.
The dataset files are local and ignored by Git. Source revisions, original split
membership, file sizes, SHA-256 hashes, and license claims are recorded in
`raw/<source>/acquisition.json`. Rebuild with:

```powershell
& .\.venv-benchmark\Scripts\python.exe scripts\acquire_orbital.py
& .\.venv-benchmark\Scripts\python.exe scripts\prepare_orbital.py
& .\.venv-benchmark\Scripts\python.exe scripts\audit_orbital.py
```

`prepare_orbital.py` needs NumPy, Pillow and PyArrow. Acquisition needs `curl.exe`
on PATH. Both scripts are explicit; normal package imports and tests do not fetch
data. The source files are pinned to specific Hugging Face repository commits.
An optional `--include-wac` source is defined in the acquisition script but is
not part of this completed collection.

| Local source | Original source and label | Downloaded splits | Usable images | Test images |
|---|---|---:|---:|---:|
| `mars_crater_binary_seg` | [Mars-Bench binary crater segmentation](https://huggingface.co/datasets/Mirali33/mb-crater_binary_seg), THEMIS daytime IR; pixel mask | val + test | 1,800 | 900 |
| `mars_conequest_seg` | [Mars-Bench ConeQuest segmentation](https://huggingface.co/datasets/Mirali33/mb-conequest_seg), CTX orbital mosaic; pixel mask | val + test | 962 | 643 |
| `moon_imp_seg` | [SomBench IMP segmentation](https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-IMP-Segmentation), LROC NAC; pixel mask | train + val + test | 130 | 10 |
| `moon_nac_crater_det` | [SomBench NAC crater detection](https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-NAC-Crater-Detection), LROC NAC; COCO boxes | train + val + test | 408 | 63 |

Layout:

- `raw/` holds original Parquet, GeoTIFF, NumPy and COCO files, source README
  files, and acquisition manifests. Mars train files were not fetched because this
  stage does not train models. Moon sources are small enough to retain in full.
- `processed/` holds extracted Mars image/mask pairs and image-only PNG renditions
  of lunar NAC arrays and IMP reflectance tiles.
- `source_index/` holds provenance indices, including label paths. **Never pass a
  source index to model inference.**
- `manifests/asset_audit.json` reports mask values and label balance;
  `manifests/split_group_audit.json` reports a first-pass scene-group overlap.
  No model-visible request or answer manifest is generated until tasks/prompts
  have been frozen.
- `candidates/moon_wac_crater_det_metadata_only/` contains SomBench WAC crater
  COCO metadata obtained during source inspection. Its 200 val/test images were
  not downloaded after the additional network request was rejected, so it is
  **not admitted** to this dataset or any count above.

The lunar IMP PNG uses a fixed reflectance display interval `[0, 0.12]`; invalid
float nodata is rendered black. The source float32 GeoTIFF is retained and each
index row records this transformation. Lunar NAC crater boxes are **not** pixel
masks. Mars image and mask bytes are extracted unchanged from the published
Parquet. No missing label channel is synthesized.

## Scientific admission notes

- Mars crater test has 886 foreground and only 14 all-background tiles. It is
  suited to segmentation; an image-level false-positive analysis is weak. The
  filename heuristic yields only four test region groups, so uncertainty must
  be estimated at the source-region level rather than treating 900 tiles as
  independent observations.
- Mars ConeQuest test has 333 foreground and 310 background tiles, but validation
  and test share all eight coarse source-region IDs. Treat it as exploratory until
  geographic/parent-tile overlap has been resolved. Do not tune prompts on that
  validation set and call the test independent.
- Moon IMP test has only 10 positive tiles. It cannot support a stable binary
  presence or false-positive estimate. Source labels may be spatially noisy.
- Moon NAC crater has 63 test tiles, grouped by 6 study boxes. `large` boxes are
  annotated only for craters >= 10 m; apply the publisher's label-completeness
  rule before scoring detections. Its COCO boxes are unsuitable as ground-truth
  segmentation masks. The current dataset card lists 289/56/63 train/val/test
  tiles (408 total), but the published
  [IBM detection checkpoint card](https://huggingface.co/nasa-ibm-ai4science/Crater-Detection-NASA-IBM-Lunar-Foundation-Model)
  describes 645/59/62 (766 total). Reconcile exact image and study-box
  membership before using this as an independent IBM test or reproducing mAP.
- The selected models are the
  [NASA–IBM Lunar Foundation Model](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model)
  and [remyxai/SpaceLLaVA](https://huggingface.co/remyxai/SpaceLLaVA).
  Both checkpoints are public; neither has been downloaded or smoke tested.
  IBM is a lunar remote-sensing encoder with separately published task heads,
  not an open-ended QA VLM, and its card does not validate Mars inputs.
  Remyx SpaceLLaVA is a 13.4B LLaVA-derived spatial-reasoning VLM trained on
  synthetic VQA; it is distinct from the planetary model of
  [Foutter et al.](https://arxiv.org/abs/2408.05924). Its FP16 weights exceed
  the available 16 GB GPU memory before inference overhead. A common paired
  task and local inference path remain to be validated.
- Orbital [Mars-Bench boulder masks](https://huggingface.co/datasets/Mirali33/mb-boulder_seg)
  exist but are not in this collection. The published test split has only four
  images; rock questions are a candidate exploratory task, not a supported
  score on the current collection. No image-aligned measured DEM/DTM or rover
  cost model is present, so flatness and physically optimal-route targets
  cannot be derived from these masks.

The mask and split checks are mechanical checks, not proof of label correctness,
geographic independence, or model compatibility. See `research_plan.md` for the
experiment gates and adapted E1–E5 protocol in
`docs/protocols/orbital_study.md`. Source dataset cards claim CC BY 4.0; retain attribution to
Mars-Bench, SomBench, original imagery instruments, and original label sources.
