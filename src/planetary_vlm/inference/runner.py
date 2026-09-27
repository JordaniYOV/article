"""Sequential backend runner, stable resume keys, append-only responses."""

from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

from planetary_vlm.contracts import ModelResponse
from planetary_vlm.datasets.manifest import load_requests, check_assets
from planetary_vlm.inference.config import RunConfig
from planetary_vlm.io import read_jsonl, write_json, write_jsonl, records_fingerprint


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("planetary-vlm-benchmark", "Pillow", "numpy", "torch", "transformers",
                 "accelerate", "bitsandbytes", "peft"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def input_fingerprint(requests) -> str:
    """Detect changed input bytes even when file paths and config are unchanged."""
    assets = {}
    for request in requests:
        for value in (*request.image_paths, *(path for _, path in request.modality_paths)):
            path = Path(value)
            if path.is_file():
                assets[str(path)] = file_hash(path)
            elif path.is_dir():
                assets[str(path)] = {
                    str(child.relative_to(path)): file_hash(child)
                    for child in sorted(path.rglob("*")) if child.is_file()
                }
            else:
                raise ValueError(f"Missing input: {path}")
    blob = json.dumps({"requests": [asdict(item) for item in requests], "assets": assets},
                      sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def run(config: RunConfig, *, resume: bool = False) -> Path:
    # Imports are lazy so validation/evaluation never needs Torch/Transformers.
    from planetary_vlm.models.registry import create_adapter

    requests = load_requests(config.requests)
    check_assets(requests)
    fingerprint = input_fingerprint(requests)
    manifest_fingerprint = records_fingerprint(asdict(request) for request in requests)
    destination = config.output_dir / config.run_id
    metadata_path = destination / "run.json"
    predictions_path = destination / "predictions.jsonl"
    expected = {"schema_version": "1.0", "run_id": config.run_id,
                "model_id": config.model_id, "backend": config.backend,
                "config_sha256": config.config_sha256, "input_sha256": fingerprint,
                "request_manifest_sha256": manifest_fingerprint,
                "seed": config.seed, "mock": config.backend == "mock"}
    completed = set()
    if destination.exists():
        if not resume:
            raise ValueError(f"Run exists; use a new run_id or --resume: {destination}")
        if not metadata_path.is_file():
            raise ValueError("Existing run has no metadata; cannot resume safely")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError("Resume rejected: configuration or input fingerprint changed")
        saved = read_jsonl(predictions_path) if predictions_path.exists() else []
        for row in saved:
            request_id = row.get("request_id")
            if (request_id in completed or request_id not in {item.request_id for item in requests}
                    or row.get("model_id") != config.model_id or row.get("run_id") != config.run_id):
                raise ValueError("Resume rejected: duplicate/foreign prediction records")
            completed.add(request_id)
    else:
        if resume:
            raise ValueError("Cannot resume a run that does not exist")
        destination.mkdir(parents=True)
        write_json(metadata_path, {**expected, "python": platform.python_version(),
                                  "platform": platform.platform(), "options": config.options,
                                  "dependencies": dependency_versions(),
                                  "timing_protocol": "request_wall_time_including_first_lazy_load_no_warmup",
                                  "config_path": str(config.config_path)})
        write_jsonl(destination / "requests.jsonl", (asdict(item) for item in requests))
    pending = [request for request in requests if request.request_id not in completed]
    if not pending:
        return destination
    random.seed(config.seed)
    adapter = create_adapter(config.backend, {**config.options, "seed": config.seed})
    try:
        with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
            for request in pending:
                start = time.perf_counter()
                try:
                    response = adapter.predict(request)
                    if not isinstance(response, ModelResponse) or response.request_id != request.request_id:
                        raise ValueError("Adapter returned an invalid response ID/type")
                    if response.status not in {"ok", "error", "unsupported"}:
                        raise ValueError("Adapter returned an invalid status")
                    if not isinstance(response.raw_response, str):
                        raise ValueError("Adapter raw_response must be a string")
                    if (isinstance(response.elapsed_seconds, bool) or
                            not isinstance(response.elapsed_seconds, (int, float)) or
                            not math.isfinite(response.elapsed_seconds) or response.elapsed_seconds < 0):
                        raise ValueError("Adapter timing must be finite and nonnegative")
                except Exception as error:
                    response = ModelResponse(request.request_id, "", "error",
                                             time.perf_counter() - start,
                                             f"{type(error).__name__}: {error}")
                row = {**asdict(response), "run_id": config.run_id,
                       "model_id": config.model_id, "mock": config.backend == "mock",
                       "request_manifest_sha256": manifest_fingerprint}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        adapter.close()
    return destination
