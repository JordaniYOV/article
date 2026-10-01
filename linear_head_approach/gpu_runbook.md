# GPU runbook: frozen SpaceLLaVA and IBM WAC encoders

Status: **prepared, not smoke tested** (30 September 2026). The current host
has no NVIDIA GPU or model runtime. Run the commands below on the computer with
the RTX 5070 Ti, **one encoder at a time**. A successful one-image smoke test
is required before extracting 120 images. All paths below are relative to the
repository root, and commands use PowerShell.

## What produces the article result

The smoke test checks only whether one image passes through each frozen encoder
and whether spatial feature tensors have the expected shapes. Its logs cannot
support a quality claim. The experiment then uses **existing**
`data_orbital/orbital_300_v3/` assets in this order:

| Stage | Source in the curated collection | Use |
|---|---|---|
| Head fitting | 50 lunar WAC images and masks from `moon/train/` | Fit one separate head per frozen encoder |
| Model selection | 39 lunar WAC images from publisher val | Select the mask threshold without test data |
| Final evaluation | 31 lunar WAC images from publisher test | Compute per-image metrics and paired scene-group uncertainty |

The 39 val and 31 test images are the selected WAC examples in
`moon/segmentation/` and `moon/height/`; the latter receive crater masks from
the publisher's COCO polygons. Elevation arrays are not encoder inputs. The
remaining 180 images in `orbital_300_v3` are Mars or other lunar annotation
tasks and cannot be added to this **single WAC crater** metric without new
task-specific targets and a separate experimental track.

## Transfer the exact input selection

Transfer this working copy's `linear_head_approach/` folder, including its
ignored `outputs/encoder_probe_wac_v1/` manifests, plus these data assets to
the GPU computer:

- `data_orbital/orbital_300_v3/moon/` (selected RGB tiles and source masks)
- `data_orbital/encoder_probe_wac_v1/` (120 verified five-band TIFFs, derived masks, COCO indices)
- `linear_head_approach/outputs/encoder_probe_wac_v1/` (separate input/target manifests and audit)

Do not transfer or inspect `data_rover/`. On the GPU computer, rerun
`python -m linear_head_approach.prepare` and verify `native_vis_verified_count=120`,
`split_counts={train:50,val:39,test:31}`, and `unique_scene_groups=106`.
This hashes all selected source images, masks, and native TIFFs.

## Environment and selected weights

Use Python 3.11 or 3.12. Install a CUDA-enabled PyTorch build that supports
the GPU using the [official PyTorch installer](https://pytorch.org/get-started/locally/),
then the pinned IBM package dependencies in a dedicated environment. Verify
`torch.cuda.is_available()` before a model run. `external/ibm-lfm` must be the
[official source](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model)
at commit `d54c67aad513cb9daca444afa425cfb278e4fbf8`.

```powershell
git clone https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model.git external/ibm-lfm
git -C external/ibm-lfm checkout d54c67aad513cb9daca444afa425cfb278e4fbf8
python -m venv .venv-encoder
& .\.venv-encoder\Scripts\python.exe -m pip install --upgrade pip
# Install CUDA PyTorch from the official selector before the next command.
& .\.venv-encoder\Scripts\python.exe -m pip install -e .
& .\.venv-encoder\Scripts\python.exe -m pip install -e external/ibm-lfm
& .\.venv-encoder\Scripts\python.exe -m pip install "transformers>=4.49,<5" safetensors
& .\.venv-encoder\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
& .\.venv-encoder\Scripts\python.exe -m pip freeze > linear_head_approach/outputs/encoder_probe_wac_v1/gpu_requirements_freeze.txt
```

Download **only** the files used by these encoders. SpaceLLaVA's indexed
visual tower has 391 tensors, all in shard 6; the other five 13B language-model
shards are unnecessary. The SpaceLLaVA config fixes
`mm_vision_select_layer=-2` and names CLIP-L/14@336. IBM uses its 2.43 GB
backbone checkpoint and config. The `hf download` command is provided by
`huggingface_hub` (installed via the IBM package dependencies). The revisions
below were resolved on 30 September 2026.

```powershell
& .\.venv-encoder\Scripts\hf.exe download remyxai/SpaceLLaVA config.json preprocessor_config.json model.safetensors.index.json model-00006-of-00006.safetensors --revision 5a69f71bda9dea3e4d8113c1047c07fd3da483f8 --local-dir external/space-weights
& .\.venv-encoder\Scripts\hf.exe download openai/clip-vit-large-patch14-336 config.json --revision ce19dc912ca5cd21c8a653c79e251e808ccabcd1 --local-dir external/clip-config
& .\.venv-encoder\Scripts\hf.exe download nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model backbone/checkpoint.pt backbone/config.yaml --revision b657bdfe6fc4c0445d918f77a2d7b919805f3bce --local-dir external/ibm-weights
```

The first extraction uses the **actual SpaceLLaVA tower tensors** from its
shard, excluding the multimodal projector and LLM. The CLIP repository supplies
only the architecture config. This avoids assuming that SpaceLLaVA's tower is
identical to the public CLIP checkpoint.

## Smoke tests, then feature extraction

```powershell
& .\.venv-encoder\Scripts\python.exe -m linear_head_approach.prepare
& .\.venv-encoder\Scripts\python.exe -m linear_head_approach.extract_features --encoder spacellava --model-dir external/space-weights --base-clip-dir external/clip-config --model-revision 5a69f71bda9dea3e4d8113c1047c07fd3da483f8 --output-dir linear_head_approach/outputs/encoder_probe_wac_v1/spacellava --limit 1
& .\.venv-encoder\Scripts\python.exe -m linear_head_approach.extract_features --encoder ibm --model-dir external/ibm-weights --ibm-source-dir external/ibm-lfm --model-revision b657bdfe6fc4c0445d918f77a2d7b919805f3bce --output-dir linear_head_approach/outputs/encoder_probe_wac_v1/ibm --limit 1
```

Check that the SpaceLLaVA feature is 24×24×1024 and the IBM feature is
64×64×768 (the final channel counts are smoke-test expectations, not confirmed
runtime facts). Each run checks input hashes, records the checkpoint hash, and refuses an
unexpected token count. If the smoke test fails, record the traceback, Python
and package versions, GPU memory, and which encoder failed; do not launch the
full extraction until the adapter is corrected.

After both smoke tests pass, rerun the same two extraction commands **without**
`--limit 1`, one after the other. Each writes 120 `.npy` spatial feature maps
and a checked `features.jsonl` in its output directory. No mask enters either
extractor.

## Train, freeze, and compare the heads

```powershell
& .\.venv-encoder\Scripts\python.exe -m linear_head_approach.fit_head fit --features linear_head_approach/outputs/encoder_probe_wac_v1/spacellava/features.jsonl --result-dir linear_head_approach/outputs/encoder_probe_wac_v1/spacellava/head
& .\.venv-encoder\Scripts\python.exe -m linear_head_approach.fit_head fit --features linear_head_approach/outputs/encoder_probe_wac_v1/ibm/features.jsonl --result-dir linear_head_approach/outputs/encoder_probe_wac_v1/ibm/head
```

Review both validation scores, feature widths, and head parameter counts.
With the protocol fixed, score the sealed test split once:

```powershell
& .\.venv-encoder\Scripts\python.exe -m linear_head_approach.fit_head score-test --features linear_head_approach/outputs/encoder_probe_wac_v1/spacellava/features.jsonl --result-dir linear_head_approach/outputs/encoder_probe_wac_v1/spacellava/head
& .\.venv-encoder\Scripts\python.exe -m linear_head_approach.fit_head score-test --features linear_head_approach/outputs/encoder_probe_wac_v1/ibm/features.jsonl --result-dir linear_head_approach/outputs/encoder_probe_wac_v1/ibm/head
& .\.venv-encoder\Scripts\python.exe -m linear_head_approach.compare --first-report linear_head_approach/outputs/encoder_probe_wac_v1/spacellava/head/test_report.json --second-report linear_head_approach/outputs/encoder_probe_wac_v1/ibm/head/test_report.json --output linear_head_approach/outputs/encoder_probe_wac_v1/paired_comparison.json
```

Send the two smoke-test logs, `audit.json`, both head metadata JSON files,
both test reports, and the paired comparison JSON back for interpretation.
Keep source TIFFs, weights, and feature arrays out of Git. The comparison
measures frozen spatial-feature utility under a shared probe protocol; IBM's
native five-channel input and SpaceLLaVA's RGB input remain a modality
confound, and pretraining-scene overlap must be audited before an article-level
claim.
