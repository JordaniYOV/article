"""Sequential, one-request-per-process control. Never used implicitly."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter
import uuid

from benchmark_services.common.files import atomic_json
from planetary_vlm.datasets.manifest import load_requests, check_assets
from planetary_vlm.inference.config import RunConfig
from planetary_vlm.inference.runner import dependency_versions, file_hash, input_fingerprint, run
from planetary_vlm.io import read_jsonl, records_fingerprint, write_jsonl

SCHEMA = "fresh-process-control-v1"


def _guard(path):
    path = Path(path).resolve()
    if "data_rover" in {p.casefold() for p in path.parts}:
        raise ValueError("Protected archive cannot be used by a fresh-process control")
    return path


def _job_config(data):
    data = dict(data)
    for key in ("requests", "output_dir", "config_path"):
        data[key] = _guard(data[key])
    return RunConfig(**data)


def child(job_path):
    """Private worker: exactly one predict call and no inference warm-up."""
    job = json.loads(_guard(job_path).read_text(encoding="utf-8"))
    if job.get("schema") != SCHEMA:
        raise ValueError("Unknown fresh-process job schema")
    config = _job_config(job["config"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", config.run_id):
        raise ValueError("Unsafe fresh worker run ID")
    _guard(config.output_dir / config.run_id)
    requests = load_requests(config.requests)
    if len(requests) != 1:
        raise ValueError("A fresh worker accepts exactly one request")
    for path in (*requests[0].image_paths, *(v for _, v in requests[0].modality_paths)):
        _guard(path)
    if config.options.get("diagnostic_request_settings", {}).get(requests[0].request_id, {}).get("compare_first_forward") is True:
        raise ValueError("Fresh worker forbids preliminary direct-forward probes")
    proof = {"schema": SCHEMA, "pid": os.getpid(), "parent_pid": os.getppid(),
        "process_nonce": uuid.uuid4().hex, "job_nonce": job["job_nonce"],
        "python_executable": sys.executable, "request_id": requests[0].request_id,
        "prior_predict_calls": 0, "predict_calls_started": 0, "predict_calls_finished": 0,
        "preliminary_inference": False, "initialized_callbacks": 0,
        "worker_code_sha256": file_hash(Path(__file__))}
    def initialized(adapter, folder):
        # This hook only instruments predict; unlike ordinary spatial smoke it
        # never invokes diagnose(), forward(), generate(), or a warm-up request.
        proof["initialized_callbacks"] += 1
        if config.backend == "spacellava":
            loaded = json.loads((folder / "loading.json").read_text(encoding="utf-8"))["provenance"]
            proof["device_map"] = loaded.get("device_map", "")
            if "cpu" in proof["device_map"].lower() or "disk" in proof["device_map"].lower():
                raise RuntimeError("Cold SpaceLLaVA control requires a GPU-only load, not CPU/disk offload")
        original = adapter.predict
        def first_only(request):
            if proof["predict_calls_started"]:
                raise RuntimeError("Fresh worker must not run a second request")
            proof["predict_calls_started"] += 1
            atomic_json(folder / "isolation_proof.json", proof)
            try:
                return original(request)
            finally:
                proof["predict_calls_finished"] += 1
                atomic_json(folder / "isolation_proof.json", proof)
        adapter.predict = first_only
        atomic_json(folder / "isolation_proof.json", proof)
    result = run(config, on_initialized=initialized)
    rows = read_jsonl(result / "predictions.jsonl")
    return 0 if len(rows) == 1 and rows[0]["status"] == "ok" else 2


def run_fresh(config, *, on_response=None):
    """No adapter is loaded here. Every child exits before the next starts."""
    _guard(config.requests)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", config.run_id):
        raise ValueError("Unsafe fresh control run ID")
    destination = _guard(_guard(config.output_dir) / config.run_id)
    if destination.exists():
        raise ValueError("Fresh control exists; use a new run ID, never resume")
    requests = load_requests(config.requests)
    if not requests or len({r.request_id for r in requests}) != len(requests):
        raise ValueError("Fresh control requires distinct, nonempty requests")
    for request in requests:
        for path in (*request.image_paths, *(v for _, v in request.modality_paths)):
            _guard(path)
        setting = config.options.get("diagnostic_request_settings", {}).get(request.request_id, {})
        if setting.get("compare_first_forward") is True:
            raise ValueError("Disable preliminary direct-forward probes for a first-call control")
    check_assets(requests)
    fingerprint = records_fingerprint(map(asdict, requests))
    destination.mkdir(parents=True)
    atomic_json(destination / "run.json", {"schema_version": "1.0", "run_id": config.run_id,
        "model_id": config.model_id, "backend": config.backend, "mock": config.backend == "mock",
        "config_sha256": config.config_sha256, "input_sha256": input_fingerprint(requests),
        "request_manifest_sha256": fingerprint, "seed": config.seed, "options": config.options,
        "dependencies": dependency_versions(), "python": platform.python_version(),
        "isolation_mode": SCHEMA, "preliminary_inference": False,
        "timing_protocol": "response_time_excludes_child_model_load; child_wall_and_load_saved_separately"})
    write_jsonl(destination / "requests.jsonl", map(asdict, requests))
    jobs, logs = destination / "jobs", destination / "logs"
    jobs.mkdir(); logs.mkdir()
    traces, proofs, loading_provenance = {}, [], None
    child_env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])
                 + os.pathsep + os.environ.get("PYTHONPATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    for index, request in enumerate(requests):
        name = "q" + str(index).zfill(3)
        child_folder = destination / "children" / name
        manifest, job_path = jobs / (name + "_requests.jsonl"), jobs / (name + ".json")
        write_jsonl(manifest, [asdict(request)])
        options = dict(config.options)
        if options.get("generation_diagnostics"):
            options["diagnostic_output_dir"] = str(child_folder)
        # Only this request's settings go to its worker, with no reference labels.
        if "diagnostic_request_settings" in options:
            options["diagnostic_request_settings"] = {
                request.request_id: options["diagnostic_request_settings"].get(request.request_id, {})}
        child_config = replace(config, run_id=name, requests=manifest,
                               output_dir=destination / "children", options=options, config_path=job_path)
        record = {**asdict(child_config), "config_sha256": ""}
        payload = json.dumps(record, sort_keys=True, default=str).encode()
        import hashlib
        child_config = replace(child_config, config_sha256=hashlib.sha256(payload).hexdigest())
        nonce = uuid.uuid4().hex
        atomic_json(job_path, {"schema": SCHEMA, "job_nonce": nonce, "config": asdict(child_config)})
        command = [sys.executable, "-u", "-m", "planetary_vlm.inference.fresh_process", "--job", str(job_path)]
        start = perf_counter()
        print(json.dumps({"event": "fresh_child_starting", "request_id": request.request_id,
                          "index": index + 1, "total": len(requests)}), flush=True)
        with (logs / (name + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=child_env)
            try:
                for line in process.stdout:
                    log.write(line); log.flush()
                    print(line, end="", flush=True)
                exit_code = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=10)
                raise
            finally:
                process.stdout.close()
        if exit_code:
            raise RuntimeError(f"Fresh worker {name} exited {exit_code}; inspect logs/{name}.log")
        proof = json.loads((child_folder / "isolation_proof.json").read_text(encoding="utf-8"))
        # CPython's Windows venv redirector can launch the real interpreter as
        # its direct child. Record both PIDs; do not mistake it for model reuse.
        direct_process = proof["pid"] == process.pid
        windows_redirector = os.name == "nt" and proof["parent_pid"] == process.pid
        if (not (direct_process or windows_redirector) or proof["job_nonce"] != nonce or proof["request_id"] != request.request_id
                or proof["prior_predict_calls"] != 0 or proof["predict_calls_started"] != 1
                or proof["predict_calls_finished"] != 1 or proof["initialized_callbacks"] != 1
                or proof["preliminary_inference"] is not False or proof["schema"] != SCHEMA
                or proof["worker_code_sha256"] != file_hash(Path(__file__))):
            raise RuntimeError("Fresh worker isolation proof is inconsistent")
        proof.update(launcher_pid=process.pid,
                     launcher_relation="direct" if direct_process else "windows_venv_redirector")
        if proof["process_nonce"] in {p["process_nonce"] for p in proofs}:
            raise RuntimeError("Fresh worker process identity was reused")
        loading_path = child_folder / "loading.json"
        if loading_path.exists():
            current_loading = json.loads(loading_path.read_text(encoding="utf-8"))["provenance"]
            if loading_provenance is not None and current_loading != loading_provenance:
                raise RuntimeError("Fresh workers restored different model provenance")
            loading_provenance = current_loading
        rows = read_jsonl(child_folder / "predictions.jsonl")
        if len(rows) != 1 or rows[0]["request_id"] != request.request_id or rows[0]["status"] != "ok":
            raise RuntimeError("Fresh worker must produce one matching successful response")
        row = {**rows[0], "child_run_id": rows[0]["run_id"], "run_id": config.run_id,
            "child_request_manifest_sha256": rows[0]["request_manifest_sha256"],
            "request_manifest_sha256": fingerprint, "worker_pid": proof["pid"],
            "worker_process_nonce": proof["process_nonce"]}
        with (destination / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        trace_path = child_folder / "generation_diagnostics.json"
        if trace_path.exists():
            child_traces = json.loads(trace_path.read_text(encoding="utf-8"))
            if set(child_traces) != {request.request_id}:
                raise RuntimeError("Fresh worker has foreign generation traces")
            traces.update(child_traces)
        proofs.append({**proof, "child_wall_seconds": perf_counter() - start,
                       "child_result": "children/" + name})
        atomic_json(destination / "generation_diagnostics.json", traces)
        atomic_json(destination / "isolation_report.json", {"schema": SCHEMA,
            "parent_pid": os.getpid(), "completed": len(proofs), "total": len(requests),
            "workers": proofs, "scientific_result": False, "benchmark_admitted": False})
        if on_response:
            on_response(destination, index + 1, len(requests))
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    return child(parser.parse_args(argv).job)


if __name__ == "__main__":
    raise SystemExit(main())
