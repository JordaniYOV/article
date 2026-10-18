"""Bounded, val-only language/generation debugging, never benchmark admission."""

from dataclasses import asdict
import json
from pathlib import Path
import re

from planetary_vlm.contracts import ModelRequest
from planetary_vlm.datasets.manifest import load_requests
from planetary_vlm.inference.config import RunConfig
from planetary_vlm.inference.runner import run
from planetary_vlm.io import read_jsonl, write_jsonl
from benchmark_services.common.files import atomic_json

PROTOCOL = "spacellava-language-pilot-v1"
SAMPLE_IDS = ("moon_seg_002", "moon_seg_012", "moon_seg_003", "mars_seg_017", "mars_seg_037")
PROMPTS = (
    ("describe", "Describe the visible contents of this image in one or two sentences.", 128),
    ("craters", "Are impact craters visible in this image? Briefly describe the visual evidence.", 96),
    ("location", "Describe the location of the most prominent surface feature using words such as left, center, or right.", 96),
)


def prepare_requests(root, originals, provenance, sample_ids=SAMPLE_IDS):
    from planetary_vlm.colab import within
    origins = {r["request_id"]: r for r in provenance}
    requests, settings, source_rows = [], {}, []
    if len(sample_ids) != 5 or len(set(sample_ids)) != 5:
        raise ValueError("Language pilot requires five distinct, fixed val samples")
    for index, sid in enumerate(sample_ids):
        matches = [r for r in originals if r.request_id.split("__")[0] == sid and r.request_id.endswith("__clean")]
        if not matches:
            raise ValueError("Missing fixed language-pilot sample: " + sid)
        original = matches[0]
        origin = origins[original.request_id]
        if origin.get("source_split") != "val" or origin.get("condition") != "clean" or origin.get("evaluation_eligible") is not True:
            raise ValueError("Language pilot must not use test/train/ineligible scenes")
        image = str(within(root, original.image_paths[0]))
        source_rows.append({**origin, "image_path": image})
        for name, prompt, budget in PROMPTS:
            request_id = sid + "__language_" + name + "__v1"
            requests.append(ModelRequest(request_id, (image,), prompt, (), budget))
            settings[request_id] = {"conversation_template":"llava_v1", "use_cache":True,
                "compare_first_forward": index < 2 and name == "describe"}
        if index < 2:
            # Predeclared checks. The runtime may override requested cache=False;
            # traces must confirm effective mode before treating it as an A/B.
            for suffix, template, cache in (("nocache", "llava_v1", False), ("v0", "llava_v0", True)):
                request_id = sid + "__language_describe_" + suffix + "__v1"
                requests.append(ModelRequest(request_id, (image,), PROMPTS[0][1], (), PROMPTS[0][2]))
                settings[request_id] = {"conversation_template":template, "use_cache":cache, "compare_first_forward":False}
    return requests, settings, source_rows


def summarize(rows, traces):
    by_id = {row["request_id"]: row for row in rows}
    comparisons = []
    for sid in SAMPLE_IDS[:2]:
        base = by_id.get(sid+"__language_describe__v1", {}).get("raw_response")
        cache_id = sid+"__language_describe_nocache__v1"
        cache_trace = traces.get(cache_id, {}).get("cache_trace", {})
        comparisons.append({"sample_id":sid, "base_description":base,
            "cache_off_requested_description":by_id.get(cache_id, {}).get("raw_response"),
            "cache_off_requested_trace":cache_trace,
            "cache_off_verified":cache_trace.get("observed_forward_calls", 0) > 0
                and cache_trace.get("unobserved_cache_calls") == 0 and cache_trace.get("effective_modes") == [False],
            "v0_description":by_id.get(sid+"__language_describe_v0__v1", {}).get("raw_response")})
    return {"protocol":PROTOCOL, "scientific_score_available":False, "benchmark_admitted":False,
        "completed":len(rows), "runtime_errors":sum(r["status"] != "ok" for r in rows),
        "token_budget_hits":sum(t.get("hit_token_budget") is True for t in traces.values()),
        "comparisons":comparisons,
        "answers":[{"request_id":r["request_id"], "raw_response":r["raw_response"], "status":r["status"]} for r in rows],
        "interpretation":"Human inspection required. Natural-language crater questions do not equal size-filtered Q1 GT. "
            "v0 is an upstream filename-heuristic fallback, not a proven correct SpaceLLaVA training template. "
            "Transformers may force cache on for inputs_embeds: a cache-off request is not a cache-off execution. "
            "Inspect effective forward modes; numeric differences alone are not evidence of a cache bug. "
            "Neither this pilot nor its completion grants smoke admission."}


def run_pilot(root, persistent, run_id, session_id):
    from planetary_vlm.colab import verify_bundle, environment, digest, model_options, checkpoint
    from api.logging import get_logger
    root, persistent = Path(root).resolve(), Path(persistent).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}",run_id) or not re.fullmatch(r"[a-f0-9]{32}",session_id):
        raise ValueError("Invalid pilot run/session ID")
    bundle = verify_bundle(root)
    env = environment(root)
    originals = load_requests(root/"data_orbital/colab_inputs/requests.jsonl", resolve_paths=False)
    provenance = read_jsonl(root/"data_orbital/colab_inputs/provenance.jsonl")
    requests, settings, sources = prepare_requests(root, originals, provenance)
    actual_id = run_id+"_language_"+session_id[:8]
    destination = root/"outputs/SpaceLLaVA"/actual_id
    durable = persistent/"runs/SpaceLLaVA"/actual_id
    if destination.exists() or durable.exists():
        raise ValueError("Pilot exists; use a fresh session/run ID, never overwrite or resume")
    manifest = root/"data_orbital/colab_inputs"/(actual_id+"_requests.jsonl")
    write_jsonl(manifest, map(asdict,requests))
    options = {**model_options(root), "generation_diagnostics":True, "diagnostic_output_dir":str(destination),
               "diagnostic_request_settings":settings}
    execution = {"protocol":PROTOCOL, "session_id":session_id, "environment":env,
        "bundle_sha256":bundle["bundle_sha256"], "settings":settings, "sources":sources,
        "scientific_result":False, "benchmark_admitted":False}
    config = RunConfig(actual_id,manifest,root/"outputs/SpaceLLaVA", "remyxai/SpaceLLaVA", "spacellava",
                       options,42,digest(execution),root/"colab_bundle.json")
    logger = get_logger("colab.language_pilot",run_id=actual_id)
    def initialized(adapter, folder):
        from benchmark_services.spacellava.diagnostics import diagnose
        report = diagnose(adapter,[row["image_path"] for row in sources[:2]])
        atomic_json(folder/"vision_diagnostics.json",report)
        if report["success"] is not True:
            raise RuntimeError("Visual diagnostics failed before language pilot")
    def progress(folder,completed,total):
        logger.info("language_request_persisted",completed=completed,total=total)
        atomic_json(folder/"language_execution.json",execution)
        checkpoint(folder,durable)
    logger.info("language_pilot_started",images=5,requests=len(requests),benchmark_admitted=False)
    try:
        result = run(config,on_initialized=initialized,on_response=progress)
    finally:
        if destination.is_dir():
            atomic_json(destination/"language_execution.json",execution)
            checkpoint(destination,durable)
    rows = read_jsonl(result/"predictions.jsonl")
    traces = json.loads((result/"generation_diagnostics.json").read_text(encoding="utf-8"))
    report = summarize(rows,traces)
    atomic_json(result/"language_report.json",report)
    checkpoint(result,durable)
    # Deliberately do not create/alter colab_smoke_passed.json.
    return {"success":len(rows)==19 and all(r["status"]=="ok" for r in rows),
        "output_dir":str(result), "persistent_dir":str(durable), "requests":19,
        "scientific_score_available":False, "benchmark_admitted":False}
