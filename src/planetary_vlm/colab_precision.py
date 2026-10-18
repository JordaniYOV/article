"""Sequential, same-GPU FP16/NF4 control; no training or scientific scoring."""

from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys

from benchmark_services.common.files import atomic_json
from planetary_vlm.colab_reference import read_baseline, make_inputs, checkpoint_identity, environment_differences
from planetary_vlm.inference.runner import file_hash
from planetary_vlm.io import read_jsonl, write_jsonl
from planetary_vlm.reference_spacellava import PRECISION_SCHEMA, precision_admission, GENERATION

ARMS = ("fp16", "nf4")  # Fail early on FP16 before spending a second model load.


def preflight(root):
    import torch
    return precision_admission(torch, root)


def persist_pair(local, durable):
    from planetary_vlm.colab import checkpoint
    checkpoint(local, durable)
    for arm in ARMS:
        if (local / arm).is_dir(): checkpoint(local / arm, durable / arm)


def summarize(folder, request_ids):
    """Keep incomplete/error records visible; equality is NOT accuracy."""
    rows, traces, loading = {}, {}, {}
    expected = set(request_ids)
    for arm in ARMS:
        path = folder / arm
        values = read_jsonl(path / "predictions.jsonl") if (path / "predictions.jsonl").is_file() else []
        rows[arm] = {r["request_id"]: r for r in values}
        if len(rows[arm]) != len(values) or set(rows[arm]) - expected:
            raise ValueError("Duplicate/foreign precision response IDs")
        traces[arm] = json.loads((path / "generation_diagnostics.json").read_text()) if (path / "generation_diagnostics.json").is_file() else {}
        loading[arm] = json.loads((path / "loading.json").read_text())["provenance"] if (path / "loading.json").is_file() else None
    comparisons = []
    for key in request_ids:
        pair = [rows[a].get(key) for a in ARMS]
        ok = all(r is not None and r["status"] == "ok" for r in pair)
        same_inputs = None
        if ok:
            a, b = (traces[arm][key] for arm in ARMS)
            same_inputs = (a["input_token_ids"] == b["input_token_ids"] and a["rendered_prompt"] == b["rendered_prompt"]
                and a["pixels"]["sha256_float32"] == b["pixels"]["sha256_float32"])
            if not same_inputs or a["generation_arguments"] != GENERATION or b["generation_arguments"] != GENERATION:
                raise ValueError("Unmatched precision inputs/generation settings")
        comparisons.append({"request_id": key, **{arm: rows[arm].get(key) for arm in ARMS},
            "same_inputs": same_inputs,
            "same_raw_response": pair[0]["raw_response"] == pair[1]["raw_response"] if ok else None,
            "same_generated_tokens": traces["fp16"][key]["generated_token_ids"] == traces["nf4"][key]["generated_token_ids"] if ok else None})
    ready = all(set(rows[a]) == expected and all(r["status"] == "ok" for r in rows[a].values()) for a in ARMS)
    visual_equal = None
    if ready:
        fp16, nf4 = (loading[a] for a in ARMS)
        if (fp16["precision"] != "fp16" or fp16["quantization"] != "none" or fp16["quantized_linear_modules"] != 0
                or not fp16["tensor_checks"]["all_language_weight_values_compared"]
                or nf4["precision"] != "nf4" or nf4["quantized_linear_modules"] != 280
                or fp16["checkpoint_sha256"] != nf4["checkpoint_sha256"]
                or fp16["code_revision"] != nf4["code_revision"] or fp16["source_sha256"] != nf4["source_sha256"]
                or fp16["processor_semantics"] != nf4["processor_semantics"]
                or fp16["memory_admission"]["gpu_identity"] != nf4["memory_admission"]["gpu_identity"]):
            raise ValueError("Precision/loading/GPU identity checks do not establish a matched pair")
        probes = [json.loads((folder / arm / "neutral_probes.json").read_text()) for arm in ARMS]
        if len(probes[0]) != 2 or [p["sample_id"] for p in probes[0]] != [p["sample_id"] for p in probes[1]]:
            raise ValueError("Precision probe source IDs differ")
        visual_equal = all(a[k]["sha256_float32"] == b[k]["sha256_float32"]
            for a, b in zip(*probes) for k in ("pixels", "vision", "projector"))
    return {"protocol": PRECISION_SCHEMA, "expected_per_arm": len(request_ids),
        "completed_per_arm": {a: len(rows[a]) for a in ARMS},
        "missing_per_arm": {a: sorted(expected - set(rows[a])) for a in ARMS},
        "errors_per_arm": {a: sum(r["status"] != "ok" for r in rows[a].values()) for a in ARMS},
        "comparison_available": ready, "matched_visual_hashes": visual_equal,
        "matched_control": ready and visual_equal is True,
        "same_raw_responses": sum(c["same_raw_response"] is True for c in comparisons),
        "different_raw_responses": sum(c["same_raw_response"] is False for c in comparisons),
        "same_generated_token_sequences": sum(c["same_generated_tokens"] is True for c in comparisons),
        "comparisons": comparisons, "scientific_score_available": False, "benchmark_admitted": False,
        "limitation": "Two development images only. Text differences are not correctness or proof of hallucination. "
            "FP16/NF4 share runtime, checkpoint, in-memory visual overrides and inputs; fixed arm order is not counterbalanced. "
            "FP16 is a numerical control, not ground truth. Visual hash differences withhold matched-control status."}


def launch_arm(root, local, durable, arm, job):
    """One OS process per arm; wait for exit before the next load."""
    path = local / arm
    path.mkdir()
    atomic_json(path / "reference_job.json", job)
    write_jsonl(path / "requests.jsonl", job["requests"])
    persist_pair(local, durable)
    worker = root / "src/planetary_vlm/reference_spacellava.py"
    with (path / "reference_worker.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", str(worker), "--job", str(path / "reference_job.json")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                log.write(line); log.flush(); print("[" + arm + "] " + line, end="", flush=True)
                if line.startswith('{"event": "reference_response_saved"'): persist_pair(local, durable)
            code = process.wait()
            if code != 0: raise RuntimeError(f"{arm} worker failed with exit code {code}; no precision fallback or second arm")
            proof = json.loads((path / "worker_started.json").read_text())
            lineage = proof["pid"] == process.pid or (os.name == "nt" and proof["parent_pid"] == process.pid)
            if (not lineage or proof["worker_sha256"] != job["worker_sha256"] or proof["schema"] != PRECISION_SCHEMA
                    or proof["precision"] != arm or proof["adapter_used"] is not False or proof["project_runner_used"] is not False):
                raise ValueError("Precision worker code/process/arm identity changed")
            atomic_json(path / "worker_attestation.json", {**proof, "launcher_pid": process.pid,
                "scope": "Logged process/code lineage, not remote security attestation"})
            predictions = read_jsonl(path / "predictions.jsonl")
            if (len(predictions) != 16 or len({r["request_id"] for r in predictions}) != 16
                    or {r["request_id"] for r in predictions} != {r["request_id"] for r in job["requests"]}
                    or any(r["status"] != "ok" or r.get("mock") is not False for r in predictions)):
                raise ValueError("Precision arm incomplete or mock; stop before next arm")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
            raise
        finally:
            process.stdout.close()
            persist_pair(local, durable)


def run_pair(root, persistent, run_id, session_id, baseline_path):
    from planetary_vlm.colab import verify_bundle, environment
    root, persistent = Path(root).resolve(), Path(persistent).resolve()
    if any("data_rover" in {p.casefold() for p in path.parts} for path in (root, persistent)):
        raise ValueError("Protected archive is out of scope")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", run_id) or not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise ValueError("Invalid precision run/session ID")
    name = run_id + "_precision_" + session_id[:8]
    local, durable = root / "outputs/SpaceLLaVA" / name, persistent / "runs/SpaceLLaVA" / name
    if local.exists() or durable.exists():
        raise ValueError("Precision run exists; use a new SESSION_ID, never overwrite or resume")
    bundle = verify_bundle(root)
    admission = preflight(root)  # Before creating a run or loading either model.
    baseline = read_baseline(baseline_path)
    requests, expected, sources = make_inputs(root, baseline)
    initial_environment = environment(root)
    worker_hash = file_hash(root / "src/planetary_vlm/reference_spacellava.py")
    execution = {"protocol": PRECISION_SCHEMA, "arm_order": list(ARMS), "session_id": session_id,
        "environment": initial_environment, "admission": admission, "bundle_sha256": bundle["bundle_sha256"],
        "worker_sha256": worker_hash, "sources": sources,
        "historical_environment_differences": environment_differences(baseline["execution"]["environment"], initial_environment),
        "historical_baseline_role": "Frozen INPUTS and checkpoint identity only; historical outputs are NOT used as current-GPU NF4 arm",
        "scientific_result": False, "benchmark_admitted": False}
    local.mkdir(parents=True); durable.mkdir(parents=True)
    atomic_json(local / "precision_execution.json", execution)
    write_jsonl(local / "requests.jsonl", requests)
    for source in sources:
        shutil.copyfile(source["image_path"], local / source["archived_image"])
        shutil.copyfile(source["image_path"], durable / source["archived_image"])
    persist_pair(local, durable)
    try:
        for arm in ARMS:
            current = environment(root)
            differences = environment_differences(initial_environment, current)
            current_admission = preflight(root)
            if differences or current_admission["gpu_identity"] != admission["gpu_identity"]:
                raise ValueError("GPU or core environment changed between precision arms")
            atomic_json(local / (arm + "_environment.json"), current)
            job = {"schema": PRECISION_SCHEMA, "precision": arm, "gpu_identity": admission["gpu_identity"],
                "workspace": str(root), "output_dir": str(local / arm), "requests": requests,
                "input_expectations": expected, "sources": [{"sample_id": s["sample_id"], "image_path": s["image_path"]} for s in sources],
                "checkpoint_sha256": checkpoint_identity(baseline["loading"]["provenance"]), "worker_sha256": worker_hash}
            launch_arm(root, local, durable, arm, job)
    except BaseException as error:
        atomic_json(local / "precision_error.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        atomic_json(local / "precision_comparison.json", summarize(local, [r["request_id"] for r in requests]))
        persist_pair(local, durable)
    report = json.loads((local / "precision_comparison.json").read_text())
    return {"success": report["matched_control"], "output_dir": str(local), "persistent_dir": str(durable),
        "comparison_available": report["comparison_available"], "requests_per_arm": 16,
        "scientific_score_available": False, "benchmark_admitted": False}
