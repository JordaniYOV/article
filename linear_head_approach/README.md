# Linear head approach

This folder contains the proposed frozen-encoder comparison and its spatial
linear head. The scientific design is in [protocol.md](protocol.md); the
commands for the separate GPU computer are in [gpu_runbook.md](gpu_runbook.md).

| File | Purpose |
|---|---|
| `encoder_probe.py`, `prepare.py`, `acquire_wac.py` | Select and verify 50 train, 39 val, and 31 test lunar WAC images from `data_orbital/orbital_300_v3` |
| `extract_features.py` | Extract frozen SpaceLLaVA or IBM spatial features without target masks |
| `linear_probe.py`, `fit_head.py` | Fit a separate matched spatial ridge head on train, select its threshold on val, and score test |
| `compare.py` | Compute the paired test difference with scene-group bootstrap uncertainty |
| `config.toml`, `protocol.md`, `gpu_runbook.md` | Pin the experiment and document its limits and commands |
| `tests/` | Synthetic software and leakage checks |

All generated manifests, features, head weights, and reports go to
`linear_head_approach/outputs/` and are ignored by Git. Source imagery and
derived WAC masks stay under `data_orbital/`; this folder does not read or
modify `data_rover/`.

From the repository root, check the preparation and tests with:

```powershell
& .\.venv-benchmark\Scripts\python.exe -m linear_head_approach.prepare
& .\.venv-benchmark\Scripts\python.exe -m unittest discover -s linear_head_approach/tests -v
```

No model weights have been loaded or heads trained on this host. A one-image
GPU smoke test comes before full feature extraction; article conclusions
require training on the WAC train split and evaluating the sealed test split.
