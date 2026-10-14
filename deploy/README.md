# API + two isolated model services

Implemented for **one host, one shared filesystem and one GPU**. The public API
and frontend endpoints stay unchanged. There is no Celery, Redis, automatic
weight download or GPU initialization at server startup. This is not a
multi-host/distributed database architecture: SQLite/WAL and OS locks require
the same local volume and identical workspace paths in all three services.

## Source layout

```text
src/
  api/                         # HTTP for the website, datasets, metrics, dispatcher
    main.py
    dispatcher.py              # Routes a particular queued job to the right service
    service_client.py          # Authenticated bounded HTTP, workspace identity checks
  benchmark_services/
    common/                    # Shared pipeline, not duplicate copies per model
      settings.py, store.py    # Workspace and the existing SQLite schema
      preparation.py           # Render once in the API environment, immutable input bundle
      execution.py             # Load/predict/save/resume one selected run
      http_app.py, controller.py # Private HTTP interface and one child process per run
      worker.py, server.py      # Shared process/server entry points
      locks.py                 # Reservation + crash-released GPU lock
      files.py, transforms.py, segmentation.py, local_runtime.py, fixtures.py
    spacellava/
      app.py, worker.py, adapter.py
    ibm/
      app.py, worker.py, adapter.py
  planetary_vlm/               # Existing CLI, dataset converters, contracts, evaluator
```

The old `src/api/worker.py` and model adapter locations were removed, not left as
duplicate implementations. Shared schema/settings/transforms were moved, with
imports and tests updated. Active historical CLI/converter modules remain
because existing workflows/tests still use them. `data_rover/`, weights, source
datasets and old results were not inspected, changed or cleaned.

`src/api/logging.py` is a lightweight, standalone structlog module shared by the
dispatcher and model workers. Model images copy only that module and API's empty
package initializer, **not** the public HTTP routes. Structlog is present in all
three dependency files. JSON operational events go to stdout; the controller
appends child output to each run's `service-worker.log` and mirrors new output to
that model service's console (including Docker logs). The worker writes directly
to the file, not a pipe; a broken console does not block its file output. Resume
starts mirroring at the file's existing end, without replaying old attempts.
No prompts, answers, GT,
credentials or exception messages are passed to these log events.
The same file also contains external libraries' stdout/stderr verbatim, which
does not pass through our JSON/field filter. Inspect full logs before sharing;
the privacy guarantee applies to our structured events, not all third-party output.

## Execution and consistency

The website submits a run to API as before. API persists it, snapshots the
renderer version, and its dispatcher checks the target service. API prepares a
single image-only bundle under `data_orbital/inference/api/<protocol_sha256>/`.
Both model environments consume the **same request/image paths and hashes**;
they never independently regenerate blur/noise/shear with different NumPy/OpenCV
versions. Model workers reverify the selection, manifests and image bytes before
loading an adapter. Masks/answers remain outside `ModelRequest` and the HTTP
submission, which contains only `run_id` and `protocol_sha256`.

Each model service spawns its selected-job worker using **its own
`sys.executable`**, without shell activation. The child uses the shared SQLite
journal to persist progress and responses. A global service reservation is held
until the child **exits**, even if the run already says `completed`; a second
GPU lock is held by the child, covering orphan workers after an API/service
crash. Two model services can be listening, but only one job occupies the GPU.

Duplicate delivery is idempotent. Offline services leave jobs in `waiting_service`
and do not stop the other model's queue. On restart, running jobs without a live
GPU owner become `interrupted`, not silently rerun. Manual resume revalidates
input hashes, dependencies, mock/real mode, shared bundle and pipeline code.
Changing any of those requires restoring the environment or a new run/protocol.
Cancellation is cooperative between requests; stopping a service or an explicit
job timeout terminates only its own child. Committed responses are retained.

`POST /runs/{run_id}/cancel` is the public cancellation route. Queued work is
atomically removed without model initialization. If a worker claims the run
concurrently, that same request flags the running job instead. Cancellation
during loading or inference waits for the current operation, then preserves the
committed response and stops; it does not forcibly interrupt CUDA. Cancelling
an already cancelled run is idempotent; unknown IDs return 404 and other stopped
or completed runs return 409. The website distinguishes **remove from queue**
from **stop running**, disables duplicate requests while cancelling and offers
resume only once the job has stopped.

`PLANETARY_API_SERVICE_MAX_JOB_SECONDS=0` disables the subprocess timeout. A
positive value is an optional loading+inference wall-clock limit, not an inference
latency metric. Automatic API-dispatcher startup is opt-in through
`PLANETARY_API_DISPATCHER_ENABLED=true`.

## Native Windows: first verify without a GPU

### Dependency files

`requirements/` contains exactly three **standalone installation specifications**:

- `api.txt`: public API, orbital Parquet/HDF5/TIFF restoration, rendering and
  CPU software/mock tests. No Torch or model runtime.
- `spacellava.txt`: private SpaceLLaVA HTTP service and native LLaVA runtime.
- `ibm-imp.txt`: private NASA–IBM HTTP service and published TerraTorch IMP runtime.

There are no nested `-r` includes, root umbrella, extra mock/dev or historical
pilot files. CPU fixture services reuse `api.txt`; real model services must use
their own profile. Orbital restoration happens in API, so its data I/O packages
are not included in SpaceLLaVA. IBM retains its published runtime's data packages.
These are **not GPU-tested lockfiles**. Frontend dependencies stay in
`frontend/package.json` and its npm lockfile; install them with `npm ci`.

Use Python **3.12 x64**. For CPU fixture testing, all three processes may use one
lightweight environment; separate model dependency stacks are needed only for
real inference. Do not create over an existing environment.

```powershell
py -3.12 -m venv .venv-api
& .\.venv-api\Scripts\python.exe -m pip install -r requirements/api.txt
& .\.venv-api\Scripts\python.exe -m pip install --no-deps -e .
```

If a root `.env` already exists, keep it and add the new fields from
`.env.example`. Otherwise copy the example. Generate a token once with
`python -c "import secrets; print(secrets.token_hex(32))"` and put that value in
`PLANETARY_API_SERVICE_TOKEN` for **all three services**. Do not commit it.
Keep `PLANETARY_API_SERVICE_MOCK=true` and `PLANETARY_API_DISPATCHER_ENABLED=true`.

Start the following in three terminals, all from the repository root:

```powershell
# Terminal 1: SpaceLLaVA service, explicit CPU fixture mode from .env
& .\.venv-api\Scripts\python.exe -m uvicorn --app-dir src benchmark_services.spacellava.app:app --host 127.0.0.1 --port 8001
# Terminal 2: IBM service, explicit CPU fixture mode from .env
& .\.venv-api\Scripts\python.exe -m uvicorn --app-dir src benchmark_services.ibm.app:app --host 127.0.0.1 --port 8002
# Terminal 3: website API, with its own lightweight dependencies
& .\.venv-api\Scripts\python.exe -m uvicorn --app-dir src api.main:app --host 127.0.0.1 --port 8000
```

Start the existing frontend with `npm run dev` from `frontend/`. Open API
`GET /services` to inspect service identity, availability, mock mode and shared
GPU reservation. Service health is **not** proof that weights/GPU are ready.
Do not use `--reload` or multiple uvicorn workers on the model services, because
their controller owns a subprocess and GPU reservation. API reload is optional.

For real inference, provision each model environment separately using
`requirements/spacellava.txt` and `requirements/ibm-imp.txt`, and install the
project editable in both. Start Terminal 1 with the SpaceLLaVA interpreter
(e.g. `.venv-spacellava-py312`) and Terminal 2 with `.venv-ibm`. API can stay in
`.venv-api`. Set `PLANETARY_API_SERVICE_MOCK=false` **only in the model service
terminals** for an explicitly authorized one-image GPU smoke test. Existing
runtime repos/checkpoints are still required at their configured revisions.

Current real adapters support SpaceLLaVA and the admitted **IBM IMP head**.
Service separation does not add new published IBM heads or make Q1–Q8 all
scoreable. Admission and scientific limitations of each task are unchanged.

### Real model environments and checks

Model profiles target **Windows x64, CPython 3.11/3.12**, not Python 3.13.
Do not reuse a copied venv, combine both model profiles or create over an existing
environment. From the repository root, provision each environment separately:

```powershell
py -3.12 -m venv .venv-spacellava
& .\.venv-spacellava\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-spacellava\Scripts\python.exe -m pip install --only-binary=tokenizers -r requirements/spacellava.txt
& .\.venv-spacellava\Scripts\python.exe -m pip install --no-deps -e .
& .\.venv-spacellava\Scripts\python.exe -m pip check
& .\.venv-spacellava\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
& .\.venv-spacellava\Scripts\python.exe src/benchmark_services/spacellava/worker.py --help

py -3.12 -m venv .venv-ibm
& .\.venv-ibm\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-ibm\Scripts\python.exe -m pip install -r requirements/ibm-imp.txt
& .\.venv-ibm\Scripts\python.exe -m pip install --no-deps -e .
& .\.venv-ibm\Scripts\python.exe -m pip check
& .\.venv-ibm\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`--help` does not run a job. Imports and `pip check` are prerequisites, not a
full model loading/generation smoke test. Source repos and checkpoints must
already exist at the configured revisions; pip does not download them.

The configured [native LLaVA commit](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/pyproject.toml)
uses Transformers 4.37.2. Do not install the generic Qwen adapter's `hf` extra
over this profile or install upstream LLaVA with dependencies: its Torch 2.1.2
pins predate Blackwell. Our candidate profile selects
[Torch 2.8.0 / torchvision 0.23.0 CUDA 12.8 wheels](https://pytorch.org/get-started/previous-versions/#v280)
and [bitsandbytes 0.48.2](https://huggingface.co/docs/bitsandbytes/v0.48.2/en/installation).

The [configured NASA–IBM commit](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/blob/d54c67aad513cb9daca444afa425cfb278e4fbf8/requirements.txt)
requires Torch >=2.12, NumPy >=2.4.4, tokenizers 0.22 and Hugging Face Hub >=1.15.
Those conflict with SpaceLLaVA; the IBM profile selects Torch 2.12.0 /
torchvision 0.27.0 CUDA 13.0 Windows wheels. Driver compatibility, TerraTorch
imports and inference still require a separately authorized GPU smoke test.

Weight download commands are **not** requirements entries. Only if downloading
is explicitly requested, the existing pinned SpaceLLaVA configuration uses:

```powershell
& .\.venv-spacellava\Scripts\hf.exe download remyxai/SpaceLLaVA --revision 5a69f71bda9dea3e4d8113c1047c07fd3da483f8 --local-dir weights/SpaceLLaVA --include "*.safetensors" "*.json" "*.model" "*.txt"
& .\.venv-spacellava\Scripts\hf.exe download openai/clip-vit-large-patch14-336 config.json preprocessor_config.json pytorch_model.bin --revision ce19dc912ca5cd21c8a653c79e251e808ccabcd1 --local-dir weights/clip-vit-large-patch14-336
```

For `c10.dll` / WinError 1114, pip cannot install the Windows C++ runtime or
NVIDIA driver. Install/repair the
[Microsoft Visual C++ x64 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
and check `nvidia-smi` on the GPU machine. Do not download individual DLLs from
third-party sites or disable TLS verification.

After changing libraries, restart the relevant service and create a **new run**;
old dependency metadata may correctly refuse resume. Keep old logs and record
`python -m pip freeze` after a successful smoke test. Requirements files are
not a substitute for that environment snapshot.

## Private model-service HTTP

All service endpoints require `Authorization: Bearer <shared token>`.
Missing token configuration fails closed (503); wrong/missing authorization is
401. Bind native model services to loopback; container model ports are not
published. Do not expose this single-user prototype API on the public internet.

- `GET /health`: service/protocol/store identity, active job, mock and GPU state.
- `POST /jobs`: `{ "run_id": "...", "protocol_sha256": "..." }`; 202, targeted and idempotent.
- `GET /jobs/{run_id}`: progress and whether the child process is still alive.
- `POST /jobs/{run_id}/cancel`: cooperative cancellation of this service's job.

Wrong backend/protocol and extra submission fields (including answers/masks)
are rejected. Services cannot accept arbitrary executables, environment paths,
checkpoint paths or SQL/database locations over HTTP. Those are server-side
configuration or frozen model configuration from API.

## Docker: optional CPU fixture stack

Dockerfiles/Compose are prepared but **have not been built or run here**.
They copy only explicit code/config/converter folders, never the workspace
root, datasets, `.env`, weights or the protected archive. Each model image
contains `common` plus only its own adapter package; the API image has neither
model adapter package. The default model profile is CPU fixture-only, with no
Torch installation, reusing `requirements/api.txt` for CPU fixtures. A shared
local volume supplies SQLite and both lock files.

After explicitly choosing to build images (package downloads are required):

```powershell
docker compose --env-file .env -f deploy/compose.yaml config --quiet
docker compose --env-file .env -f deploy/compose.yaml up --build
```

The frontend remains a host process and calls `http://localhost:8000`. Containers
use a **fresh named database/output volume**, not the native Windows SQLite
with `C:\...` paths. Existing `data_orbital` is mounted; no whole repository
or archive is mounted. Register/import configurations via API inside this stack.
Choose either native or container GPU execution; do not run both concurrently,
because their separate state volumes do not provide a shared cross-platform GPU lock.
Do not merge native Windows and Linux-container results by run ID or rewrite
their saved paths/metadata. `down` keeps volumes; do not use `down -v` unless you
explicitly intend to delete that stack's saved runs/results.

For a separately authorized GPU deployment, use the additional
`compose.gpu.yaml`. It explicitly builds Linux CUDA dependency profiles, switches
off fixtures, requests the GPU and mounts only that model's existing weights/code
read-only. It does not download checkpoints. Windows wheels are excluded by
their platform markers; CUDA PyTorch is installed from its official Linux index
before the remaining profile. Linux compatibility/driver/16 GB admission remain
unverified. Do **not** run this override before a GPU smoke-test task.

Reference: [Docker GPU on Windows uses WSL2](https://docs.docker.com/desktop/features/gpu/),
[Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/).
Containers do not partition VRAM; the shared service/GPU locks still serialize jobs.

## Tests and migration

`requirements/api.txt` also supplies software-test dependencies; then run
`python -m unittest discover -s tests -v`. New CPU tests cover both HTTP service
interfaces and real selected-job subprocesses, mock marking in responses/metrics,
same input bytes for both models, token/backend/GT rejection, duplicate delivery,
cross-service reservation, process failure, timeout, offline service and mode-change
resume refusal. A real-loopback test starts API and both HTTP service servers in
separate processes, runs CPU fixtures, and stops every test server afterward.
Synthetic fixtures are **not scientific results**.

The SQLite table names/columns are unchanged; shared storage moved in source only.
Old `api.worker` commands are removed: start the two service servers and enable
the API dispatcher instead. Old completed outputs remain readable. New renderer
and pipeline fingerprints intentionally prevent silently resuming old runs under
different implementations. Create a new run for the refactored pipeline; do not
delete historical outputs or edit `run.json` to bypass reproducibility checks.
