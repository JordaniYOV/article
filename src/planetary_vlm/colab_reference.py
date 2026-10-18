"""Coordinator for standalone upstream LLaVA control; never loads an adapter."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from benchmark_services.common.files import atomic_json
from planetary_vlm.inference.runner import file_hash
from planetary_vlm.io import read_jsonl, write_jsonl
from planetary_vlm.reference_spacellava import SCHEMA, GENERATION

BASELINE_SHA256 = "7ec827c442e2d556fe25b7f3092f3cffc94188fc8de5334402f6e3bef00d403e"
BASELINE_ROOT = "spacellava_evidence_v10_evidence_55164967/"
CORE_DEPENDENCIES = ("python", "torch", "transformers", "accelerate", "bitsandbytes", "tokenizers",
    "safetensors", "sentencepiece", "Pillow", "numpy", "einops", "timm", "peft")


def read_baseline(path):
    path = Path(path).resolve()
    if "data_rover" in {p.casefold() for p in path.parts} or file_hash(path) != BASELINE_SHA256:
        raise ValueError("Use the exact audited v10 result ZIP, not another run or a modified archive")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != len({e.filename for e in entries}) or archive.testzip() is not None:
            raise ValueError("Baseline ZIP integrity failed")
        if any(not e.filename.startswith(BASELINE_ROOT) or e.file_size > 2_000_000 for e in entries):
            raise ValueError("Unexpected baseline ZIP entry")
        raw = {e.filename[len(BASELINE_ROOT):]: archive.read(e) for e in entries}
    return {"raw": raw, "execution": json.loads(raw["evidence_execution.json"]),
        "run": json.loads(raw["run.json"]), "loading": json.loads(raw["loading.json"]),
        "vision": json.loads(raw["vision_diagnostics.json"]), "traces": json.loads(raw["generation_diagnostics.json"]),
        "requests": list(map(json.loads, raw["requests.jsonl"].splitlines())),
        "predictions": list(map(json.loads, raw["predictions.jsonl"].splitlines()))}


def checkpoint_identity(provenance):
    result = {}
    for name, value in provenance["checkpoint_sha256"].items():
        key = name.split("/weights/", 1)[-1]
        if key.count("/") != 1 or key.split("/")[0] not in {"SpaceLLaVA", "clip-vit-large-patch14-336"} or key in result:
            raise ValueError("Ambiguous baseline checkpoint file")
        result[key] = value
    if len(result) != 15:
        raise ValueError("Incomplete baseline checkpoint identity")
    return result


def environment_differences(baseline, current):
    differences = {}
    for key in CORE_DEPENDENCIES:
        old, new = baseline["dependencies"].get(key), current["dependencies"].get(key)
        if old is None or new is None or old != new:
            differences["dependencies/" + key] = {"baseline": old, "current": new}
    for key in ("gpu", "gpu_memory_bytes", "compute_capability", "cuda", "driver"):
        if baseline.get(key) is None or baseline.get(key) != current.get(key):
            differences[key] = {"baseline": baseline.get(key), "current": current.get(key)}
    return differences


def make_inputs(root, baseline):
    from planetary_vlm.colab_evidence import prepare_requests
    from planetary_vlm.datasets.manifest import load_requests
    inputs = root / "data_orbital/colab_inputs"
    requests, settings, pairs, sources = prepare_requests(root,
        load_requests(inputs / "requests.jsonl", resolve_paths=False), read_jsonl(inputs / "provenance.jsonl"))
    rows = [asdict(r) for r in requests]
    # JSON-normalize tuples before comparing with the archived contract.
    rows = json.loads(json.dumps(rows))
    if pairs != baseline["execution"]["pairs"] or len(baseline["requests"]) != len(rows):
        raise ValueError("Baseline paired questions differ")
    expected = {}
    for row, previous in zip(rows, baseline["requests"]):
        if {k: v for k, v in row.items() if k != "image_paths"} != {k: v for k, v in previous.items() if k != "image_paths"}:
            raise ValueError("Reference prompt/order/generation contract changed")
        key = row["request_id"]
        trace = baseline["traces"][key]
        if trace["generation_arguments"] != GENERATION:
            raise ValueError("Baseline does not use the matched greedy settings")
        sid = key.split("__")[0]
        probe_index = next(i for i, s in enumerate(baseline["execution"]["sources"]) if s["sample_id"] == sid)
        expected[key] = {"input_token_ids": trace["input_token_ids"], "rendered_prompt": trace["rendered_prompt"],
            "pixels_sha256_float32": baseline["vision"]["traces"][probe_index]["pixels"]["sha256_float32"]}
    for source in sources:
        if (baseline["raw"][source["archived_image"]] != Path(source["image_path"]).read_bytes()
                or file_hash(Path(source["image_path"])) != source["image_sha256"]):
            raise ValueError("Reference source PNG differs from baseline")
    return rows, expected, sources


def compare_results(folder, baseline):
    rows = read_jsonl(folder / "predictions.jsonl") if (folder / "predictions.jsonl").is_file() else []
    trace_path = folder / "generation_diagnostics.json"
    traces = json.loads(trace_path.read_text()) if trace_path.is_file() else {}
    previous = {r["request_id"]: r for r in baseline["predictions"]}
    if len({r["request_id"] for r in rows}) != len(rows) or any(r["request_id"] not in previous for r in rows):
        raise ValueError("Duplicate/foreign reference response")
    comparisons = []
    for row in rows:
        key = row["request_id"]
        ok = row["status"] == "ok"
        comparisons.append({"request_id": key, "baseline_response": previous[key]["raw_response"],
            "reference_response": row["raw_response"], "status": row["status"],
            "same_raw_response": row["raw_response"] == previous[key]["raw_response"] if ok else None,
            "same_generated_tokens": traces[key]["generated_token_ids"] == baseline["traces"][key]["generated_token_ids"] if ok else None,
            "same_first_token_top5": traces[key].get("first_token_top5") == baseline["traces"][key].get("first_token_top5") if ok else None})
    probes_path = folder / "neutral_probes.json"
    neutral = []
    if probes_path.is_file():
        probes = json.loads(probes_path.read_text())
        if [r["sample_id"] for r in probes] != [s["sample_id"] for s in baseline["execution"]["sources"]]:
            raise ValueError("Reference neutral probe IDs/order changed")
        for row, old in zip(probes, baseline["vision"]["traces"]):
            neutral.append({"sample_id": row["sample_id"], **{key + "_hash_equal":
                row[key]["sha256_float32"] == old[key]["sha256_float32"]
                for key in ("pixels", "vision", "projector", "next_token_logits")}})
    return {"protocol": SCHEMA, "expected": len(previous), "completed": len(rows),
        "missing_request_ids": sorted(set(previous) - {r["request_id"] for r in rows}),
        "runtime_errors": sum(r["status"] != "ok" for r in rows),
        "same_raw_responses": sum(c["same_raw_response"] is True for c in comparisons),
        "same_generated_token_sequences": sum(c["same_generated_tokens"] is True for c in comparisons),
        "comparisons": comparisons, "neutral_probe_comparisons": neutral,
        "scientific_score_available": False, "benchmark_admitted": False,
        "scope": "Standalone upstream builder and preprocessing/generation glue, no project adapter/runner. "
            "Same LLaVA/Torch/Transformers, explicit merged-vision/NF4 compatibility overrides, not an independent engine or publisher oracle. "
            "Matching answers constrain the glue-bug hypothesis, not all runtime/model bugs. Differences alone do not establish a fix or its cause."}


def run_pilot(root, persistent, run_id, session_id, baseline_path):
    from planetary_vlm.colab import verify_bundle, environment, checkpoint
    root, persistent = Path(root).resolve(), Path(persistent).resolve()
    if any("data_rover" in {p.casefold() for p in pth.parts} for pth in (root, persistent)):
        raise ValueError("Protected archive is out of scope")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", run_id) or not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise ValueError("Invalid reference run/session ID")
    name = run_id + "_reference_" + session_id[:8]
    destination, durable = root / "outputs/SpaceLLaVA" / name, persistent / "runs/SpaceLLaVA" / name
    if destination.exists() or durable.exists():
        raise ValueError("Reference run exists; choose a new SESSION_ID, never overwrite/resume")
    bundle = verify_bundle(root)
    baseline = read_baseline(baseline_path)
    rows, expected, sources = make_inputs(root, baseline)
    current = environment(root)
    differences = environment_differences(baseline["execution"]["environment"], current)
    if differences:
        raise ValueError("Reference environment differs from baseline; do not treat it as a matched control: " + json.dumps(differences))
    worker = root / "src/planetary_vlm/reference_spacellava.py"
    job = {"schema": SCHEMA, "workspace": str(root), "output_dir": str(destination), "requests": rows,
        "input_expectations": expected, "sources": [{"sample_id": s["sample_id"], "image_path": s["image_path"]} for s in sources],
        "checkpoint_sha256": checkpoint_identity(baseline["loading"]["provenance"]), "worker_sha256": file_hash(worker)}
    destination.mkdir(parents=True)
    atomic_json(destination / "reference_job.json", job)
    write_jsonl(destination / "requests.jsonl", rows)
    atomic_json(destination / "reference_execution.json", {"protocol": SCHEMA, "baseline_sha256": BASELINE_SHA256,
        "session_id": session_id, "environment": current, "environment_differences": differences,
        "bundle_sha256": bundle["bundle_sha256"], "sources": sources, "worker_sha256": job["worker_sha256"],
        "scientific_result": False, "benchmark_admitted": False})
    durable.mkdir(parents=True)
    for source in sources:
        shutil.copyfile(source["image_path"], destination / source["archived_image"])
        shutil.copyfile(source["image_path"], durable / source["archived_image"])
    checkpoint(destination, durable)
    command = [sys.executable, "-u", str(worker), "--job", str(destination / "reference_job.json")]
    exit_code = None
    try:
        with (destination / "reference_worker.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end="", flush=True)
                    if line.startswith('{"event": "reference_response_saved"'):
                        checkpoint(destination, durable)
                exit_code = process.wait()
                if exit_code == 0:
                    proof = json.loads((destination / "worker_started.json").read_text())
                    relation = proof["pid"] == process.pid or (os.name == "nt" and proof["parent_pid"] == process.pid)
                    if (not relation or proof["worker_sha256"] != job["worker_sha256"]
                            or proof["adapter_used"] is not False or proof["project_runner_used"] is not False):
                        raise ValueError("Standalone worker identity/inference path is inconsistent")
                    atomic_json(destination / "worker_attestation.json", {**proof, "launcher_pid": process.pid,
                        "scope": "Logged process/code lineage, not remote attestation"})
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=20)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
                raise
            finally:
                process.stdout.close()
    finally:
        # The coordinator already has the archive in memory, but writes its
        # answers alongside model outputs only AFTER the worker exits.
        write_jsonl(destination / "baseline_predictions.jsonl", baseline["predictions"])
        atomic_json(destination / "reference_comparison.json", compare_results(destination, baseline))
        checkpoint(destination, durable)
    comparison = json.loads((destination / "reference_comparison.json").read_text())
    return {"success": exit_code == 0 and comparison["completed"] == 16 and comparison["runtime_errors"] == 0,
        "output_dir": str(destination), "persistent_dir": str(durable), "requests": 16,
        "scientific_score_available": False, "benchmark_admitted": False}
