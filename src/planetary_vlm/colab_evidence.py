"""Paired, greedy Q1-Q4 response-format pilot. No scientific scoring/admission."""

from dataclasses import asdict
from pathlib import Path
import json
import re
import shutil

from benchmark_services.common.files import atomic_json
from planetary_vlm.contracts import ModelRequest
from planetary_vlm.datasets.crater_reference import TASKS, ANSWERS
from planetary_vlm.datasets.manifest import load_requests
from planetary_vlm.evaluation.parser import parse_answer
from planetary_vlm.evaluation.structured import SCHEMA as RESPONSE_SCHEMA, parse_label_evidence
from planetary_vlm.inference.config import RunConfig
from planetary_vlm.inference.runner import file_hash, run
from planetary_vlm.io import read_jsonl, write_jsonl

PROTOCOL = "spacellava-evidence-pilot-v1"
MAX_TOKENS = 96
SOURCES = {
    "moon_seg_002": "4f8f93f91a302daf31476f7a2fb51ba3441c59ae1a6a4753fd81b2815f840238",
    "moon_seg_012": "bbc2bec1ca26cc62ae0da1ca0895b2d37819ccd07e92237f47ff8dedd90e5e36",
}
EVIDENCE_PREFIX = (
    "Answer the question using only visible evidence in the image. Do not assume that a crater is present. "
    "Distinguish visible observations from geological interpretations. "
    "Do not invent geological causes, materials or unseen features. "
)


def structured_prompt(original):
    suffix = "Reply with exactly one label: " + ", ".join(original.allowed_answers) + ". No explanation."
    if not original.prompt.endswith(suffix):
        raise ValueError("Frozen response instruction changed; do not guess how to replace it")
    # Preserve the entire task, eligibility, grid, tie and absence rules.
    body = original.prompt[:-len(suffix)]
    return (EVIDENCE_PREFIX + body + "Return exactly two lines, without a preamble or code fences:\n"
        "ANSWER: <one label from: " + ", ".join(original.allowed_answers) + ">\n"
        "EVIDENCE: <one short sentence describing visible cues supporting your answer; "
        "if uncertain, state the uncertainty rather than inventing evidence>")


def prepare_requests(root, originals, provenance):
    from planetary_vlm.colab import within, digest
    by_id = {r.request_id: r for r in originals}
    origins = {r["request_id"]: r for r in provenance}
    if len(by_id) != len(originals) or len(origins) != len(provenance):
        raise ValueError("Duplicate source request/provenance IDs")
    requests, settings, pairs, sources = [], {}, [], []
    for sample_index, (sid, checksum) in enumerate(SOURCES.items()):
        source_path = None
        for task_index, (task, labels) in enumerate(zip(TASKS, ANSWERS)):
            source_id = sid + "__" + task + "__clean"
            if source_id not in by_id or source_id not in origins:
                raise ValueError("Missing fixed evidence-pilot source: " + source_id)
            original, origin = by_id[source_id], origins[source_id]
            if (origin.get("sample_id") != sid or origin.get("source_split") != "val"
                    or origin.get("condition") != "clean" or origin.get("evaluation_eligible") is not True):
                raise ValueError("Evidence pilot requires fixed eligible clean val sources, never test/train")
            if (len(original.image_paths) != 1 or original.modality_paths or original.do_sample
                    or original.allowed_answers != labels):
                raise ValueError("Frozen source task/modalities/answer set changed")
            image = within(root, original.image_paths[0])
            if file_hash(image) != checksum or origin.get("image_sha256") != checksum:
                raise ValueError("Evidence source image SHA-256 changed")
            if source_path is not None and source_path != image:
                raise ValueError("Q1-Q4 must use the same source image")
            if source_path is None:
                sources.append({**origin, "image_path": str(image), "archived_image": sid + ".png"})
                source_path = image
            prompts = {"closed": original.prompt, "evidence": structured_prompt(original)}
            order = ("closed", "evidence") if (sample_index + task_index) % 2 == 0 else ("evidence", "closed")
            pair = {"source_request_id": source_id, "sample_id": sid, "task_id": task,
                "allowed_answers": list(labels), "original_prompt_sha256": digest(original.prompt),
                "original_max_new_tokens": original.max_new_tokens, "execution_order": list(order)}
            for mode in order:
                key = sid + "__" + task + "__" + mode + "_pilot_v1"
                requests.append(ModelRequest(key, (str(image),), prompts[mode], (), MAX_TOKENS))
                settings[key] = {"conversation_template": "llava_v1", "use_cache": True,
                    "compare_first_forward": False, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0}
                pair[mode + "_request_id"] = key
            pairs.append(pair)
    return requests, settings, pairs, sources


def summarize(rows, traces, pairs):
    by_id = {r["request_id"]: r for r in rows}
    expected = {p[m + "_request_id"] for p in pairs for m in ("closed", "evidence")}
    if len(by_id) != len(rows) or set(by_id) - expected or set(traces) - expected:
        raise ValueError("Duplicate/foreign pilot responses or traces")
    results = []
    counts = {mode: {"expected": len(pairs), "completed": 0, "runtime_errors": 0,
        "valid_labels": 0, "valid_formats": 0, "token_budget_hits": 0} for mode in ("closed", "evidence")}
    for pair in pairs:
        comparison = {"source_request_id": pair["source_request_id"], "sample_id": pair["sample_id"],
            "task_id": pair["task_id"], "allowed_answers": pair["allowed_answers"]}
        for mode in counts:
            key = pair[mode + "_request_id"]
            row = by_id.get(key)
            text = row["raw_response"] if row and row["status"] == "ok" else ""
            if mode == "evidence":
                parsed = parse_label_evidence(text, pair["allowed_answers"])
            else:
                label = parse_answer(text, pair["allowed_answers"])
                parsed = {"parsed_answer": label, "label_valid": label is not None,
                    "format_valid": label is not None, "evidence": None, "evidence_review_status": "NOT_REQUESTED"}
            trace = traces.get(key, {})
            comparison[mode] = {"request_id": key, "status": row["status"] if row else "missing",
                "raw_response": row["raw_response"] if row else None, **parsed,
                "hit_token_budget": trace.get("hit_token_budget"), "trace_success": trace.get("success")}
            counts[mode]["completed"] += row is not None
            counts[mode]["runtime_errors"] += row is not None and row["status"] != "ok"
            counts[mode]["valid_labels"] += parsed["label_valid"]
            counts[mode]["valid_formats"] += parsed["format_valid"]
            counts[mode]["token_budget_hits"] += trace.get("hit_token_budget") is True
        both = all(comparison[m]["label_valid"] for m in counts)
        comparison["same_label"] = (comparison["closed"]["parsed_answer"] == comparison["evidence"]["parsed_answer"]) if both else None
        results.append(comparison)
    return {"protocol": PROTOCOL, "response_schema": RESPONSE_SCHEMA, "expected": len(expected),
        "completed": len(rows), "runtime_errors": sum(r["status"] != "ok" for r in rows),
        "missing_request_ids": sorted(expected - set(by_id)), "by_format": counts, "comparisons": results,
        "comparable_label_pairs": sum(r["same_label"] is not None for r in results),
        "changed_label_pairs": sum(r["same_label"] is False for r in results),
        "evidence_truth_automatically_scored": False, "scientific_score_available": False,
        "benchmark_admitted": False, "interpretation": "Paired instruction/response-format diagnostic on two pre-used val scenes. "
            "Format validity and changed labels are not accuracy. All evidence awaits human review, including plausible or contradictory prose. "
            "No candidate answers are accepted as GT and no smoke gate is granted."}


def run_pilot(root, persistent, run_id, session_id):
    from planetary_vlm.colab import verify_bundle, environment, digest, model_options, checkpoint
    from api.logging import get_logger
    root, persistent = Path(root).resolve(), Path(persistent).resolve()
    if any("data_rover" in {p.casefold() for p in path.parts} for path in (root, persistent)):
        raise ValueError("Protected archive cannot be used for this pilot")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", run_id) or not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise ValueError("Invalid pilot run/session ID")
    actual_id = run_id + "_evidence_" + session_id[:8]
    destination, durable = root / "outputs/SpaceLLaVA" / actual_id, persistent / "runs/SpaceLLaVA" / actual_id
    if destination.exists() or durable.exists():
        raise ValueError("Evidence pilot exists; use a fresh SESSION_ID, never overwrite or resume")
    bundle = verify_bundle(root)
    if bundle["protocol"].get("question_type") != "candidate_Q1_Q4_clean_unscored":
        raise ValueError("Evidence pilot requires the frozen clean Q1-Q4 bundle")
    inputs = root / "data_orbital/colab_inputs"
    requests, settings, pairs, sources = prepare_requests(root,
        load_requests(inputs / "requests.jsonl", resolve_paths=False), read_jsonl(inputs / "provenance.jsonl"))
    env = environment(root)
    manifest = inputs / (actual_id + "_requests.jsonl")
    write_jsonl(manifest, map(asdict, requests))
    options = {**model_options(root), "generation_diagnostics": True, "diagnostic_output_dir": str(destination),
        "diagnostic_request_settings": settings}
    execution = {"protocol": PROTOCOL, "session_id": session_id, "environment": env,
        "bundle_sha256": bundle["bundle_sha256"], "pairs": pairs, "sources": sources, "settings": settings,
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_TOKENS,
            "use_cache": True, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
        "model_loads": 1, "scientific_result": False, "benchmark_admitted": False}
    config = RunConfig(actual_id, manifest, root / "outputs/SpaceLLaVA", "remyxai/SpaceLLaVA", "spacellava",
        options, 42, digest(execution), root / "colab_bundle.json")
    logger = get_logger("colab.evidence_pilot", run_id=actual_id)

    def save_context(folder):
        atomic_json(folder / "evidence_execution.json", execution)
        for source in sources:
            target = folder / source["archived_image"]
            if target.exists():
                if file_hash(target) != source["image_sha256"]:
                    raise ValueError("Archived evidence-pilot input changed")
            else:
                shutil.copyfile(source["image_path"], target)

    def initialized(adapter, folder):
        from benchmark_services.spacellava.diagnostics import diagnose
        save_context(folder)
        # Preflight on this same adapter; unlike v9, this is not a first-call test.
        diagnostic = diagnose(adapter, [s["image_path"] for s in sources])
        atomic_json(folder / "vision_diagnostics.json", diagnostic)
        if diagnostic.get("success") is not True:
            raise RuntimeError("Visual diagnostics failed before evidence pilot")

    def progress(folder, completed, total):
        logger.info("evidence_request_persisted", completed=completed, total=total)
        save_context(folder)
        checkpoint(folder, durable)

    # Publish the two immutable PNGs before model loading; checkpoint() only
    # transports text records, and a later load failure must keep the inputs.
    save_context(durable)
    logger.info("evidence_pilot_started", images=2, requests=len(requests), benchmark_admitted=False)
    try:
        result = run(config, on_initialized=initialized, on_response=progress)
    finally:
        if destination.is_dir():
            save_context(destination)
            rows = read_jsonl(destination / "predictions.jsonl") if (destination / "predictions.jsonl").is_file() else []
            trace_path = destination / "generation_diagnostics.json"
            traces = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.is_file() else {}
            atomic_json(destination / "evidence_report.json", summarize(rows, traces, pairs))
            checkpoint(destination, durable)
    return {"success": len(rows) == len(requests) and all(r["status"] == "ok" for r in rows),
        "output_dir": str(result), "persistent_dir": str(durable), "requests": len(requests),
        "scientific_score_available": False, "benchmark_admitted": False}
