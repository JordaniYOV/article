# Installation profiles

These are **installation specifications**, not GPU-tested lockfiles. Preparing
these files does not install packages, download weights, or confirm RTX 5070 Ti
compatibility. Model profiles target **Windows x64 and CPython 3.11/3.12**.
Keep API and its worker on the same interpreter. Do not reuse a copied venv from
another machine, and do not combine the two model profiles in one environment.

| File | Purpose |
|---|---|
| `../requirements.txt` | Default full SpaceLLaVA profile, including API and orbital data I/O |
| `app.txt` | API, database, worker orchestration and image transformations; no Torch |
| `orbital.txt` | Parquet/HDF5/TIFF data restoration; no model runtime |
| `spacellava.txt` | Full native SpaceLLaVA inference and NF4 dependencies |
| `ibm-imp.txt` | Separate NASA–IBM/TerraTorch IMP environment |
| `dev.txt` | CPU software tests, including httpx and requirement syntax checks |
| `core-windows-py312.txt`, `stereo-pilot.txt` | Historical CPU-pilot pins; not full model environments |

## SpaceLLaVA (default)

From the repository root, use a new environment so obsolete packages do not
survive an in-place upgrade. Do not create over an existing environment.

```powershell
py -3.12 -m venv .venv-spacellava
& .\.venv-spacellava\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-spacellava\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv-spacellava\Scripts\python.exe -m pip install --no-deps -e .
& .\.venv-spacellava\Scripts\python.exe -m pip check
& .\.venv-spacellava\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
& .\.venv-spacellava\Scripts\python.exe src/api/worker.py --help
```

Only after the import checks pass, run API and worker in separate terminals
using that same interpreter. `--help` never processes the queue.

```powershell
& .\.venv-spacellava\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
& .\.venv-spacellava\Scripts\python.exe src/api/worker.py
```

The native [LLaVA source at the configured commit](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/pyproject.toml)
pins Transformers 4.37.2. Its forward/generation interface is not the generic
Qwen adapter's `hf` extra (`>=4.49`); do not install that extra over this profile.
Torch 2.1.2 from upstream predates Blackwell. This profile instead selects the
official [Torch 2.8.0 / torchvision 0.23.0 CUDA 12.8 wheels](https://pytorch.org/get-started/previous-versions/#v280)
and [bitsandbytes 0.48.2](https://huggingface.co/docs/bitsandbytes/v0.48.2/en/installation).
This is a candidate combination; full model loading and generation still need
a separately authorized one-image smoke test. Do not install upstream LLaVA
with dependencies: the adapter uses `external/LLaVA` directly and checks its
commit. The repository and checkpoint folders must already exist.

Weight download commands are **not** requirement entries. If downloading is
explicitly requested, these are the commands for the existing pinned config:

```powershell
& .\.venv-spacellava\Scripts\hf.exe download remyxai/SpaceLLaVA --revision 5a69f71bda9dea3e4d8113c1047c07fd3da483f8 --local-dir weights/SpaceLLaVA --include "*.safetensors" "*.json" "*.model" "*.txt"
& .\.venv-spacellava\Scripts\hf.exe download openai/clip-vit-large-patch14-336 config.json preprocessor_config.json pytorch_model.bin --revision ce19dc912ca5cd21c8a653c79e251e808ccabcd1 --local-dir weights/clip-vit-large-patch14-336
```

## NASA–IBM

Use a **different** environment:

```powershell
py -3.12 -m venv .venv-ibm
& .\.venv-ibm\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-ibm\Scripts\python.exe -m pip install -r requirements/ibm-imp.txt
& .\.venv-ibm\Scripts\python.exe -m pip install --no-deps -e .
& .\.venv-ibm\Scripts\python.exe -m pip check
& .\.venv-ibm\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The [configured NASA–IBM commit](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/d54c67aad513cb9daca444afa425cfb278e4fbf8/requirements.txt)
requires Torch >=2.12, NumPy >=2.4.4, tokenizers 0.22 and Hugging Face Hub >=1.15.
These conflict with the SpaceLLaVA profile. The IBM file mirrors those upstream
dependencies, adds API/orbital I/O and PyYAML, and selects official Torch 2.12.0 /
torchvision 0.27.0 CUDA 13.0 Windows wheels. Driver compatibility, TerraTorch
imports and inference are not locally tested.

The adapter imports `terratorch_integration` and `ni_lfm` from the checked local
`external/NASA-IBM-Lunar-Foundation-Model` repository. No runtime source or
weights are implicitly downloaded by pip. Use the CLI's `full-systems --model
NASA-IBM-Lunar-Foundation-Model` in this environment for a separately authorized
smoke test. The API worker does not filter its queue by model: do not queue IBM
runs to a SpaceLLaVA-only worker, or run workers from both environments together.

## Windows DLL errors and reproducibility

`c10.dll` initialization can fail before a checkpoint is read. Pip cannot
install the Windows C++ runtime or the NVIDIA driver. Install/repair the
[Microsoft Visual C++ x64 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
and check `nvidia-smi` on the GPU machine. Do not download individual DLLs from
third-party sites, disable TLS checks, or assume changing requirements fixes
every WinError 1114.

After changing libraries, restart the worker and create a **new run**; an old
run may refuse resume because its dependency metadata differs. Keep old logs.
Record `python -m pip freeze` after a successful smoke test; the range-based
requirements are not a substitute for that environment snapshot.

For CPU software tests, use `requirements/dev.txt`, install the project editable,
then run `python -m unittest discover -s tests -v`. Frontend dependencies remain
in `frontend/package.json` / its npm lockfile and are installed with `npm ci`.
