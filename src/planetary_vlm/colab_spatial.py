"""Predeclared spatial controls, not natural-image benchmark evaluation."""

from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import shutil

from benchmark_services.common.files import atomic_json
from planetary_vlm.contracts import ModelRequest
from planetary_vlm.datasets.manifest import load_requests
from planetary_vlm.inference.config import RunConfig
from planetary_vlm.inference.runner import file_hash, run
from planetary_vlm.io import read_jsonl, write_jsonl

PROTOCOL = "spacellava-spatial-control-v1"
SOURCE_ID = "moon_seg_002"
SOURCE_SHA256 = "4f8f93f91a302daf31476f7a2fb51ba3441c59ae1a6a4753fd81b2815f840238"
INPUT_FOLDER = "data_orbital/colab_spatial_v1"
PROMPTS = (
    ("free", "Where is the largest crater in this image? Describe its position briefly.", 64),
    ("forward", "Which horizontal third of the image contains the center of the largest crater? "
                "Answer with exactly one word: LEFT, CENTER, or RIGHT.", 32),
    ("reverse", "Which horizontal third of the image contains the center of the largest crater? "
                "Answer with exactly one word: RIGHT, CENTER, or LEFT.", 32),
)
FRESH_REQUEST_IDS = tuple("spatial_" + cid + "__" + kind + "__v1"
                         for cid in ("c00", "c04") for kind in ("forward", "reverse"))


def prepare_controls(root):
    """CPU-only, deterministic padding/placement/mirror; never edit originals."""
    from PIL import Image, ImageOps, __version__ as pillow_version
    from planetary_vlm.colab import verify_bundle, within
    root = Path(root).resolve()
    bundle = verify_bundle(root)
    inputs = root / "data_orbital/colab_inputs"
    original = next((r for r in load_requests(inputs / "requests.jsonl", resolve_paths=False)
                     if r.request_id == SOURCE_ID + "__crater_presence_v1__clean"), None)
    if original is None:
        raise ValueError("Missing pinned spatial-control source")
    origin = next(r for r in read_jsonl(inputs / "provenance.jsonl") if r["request_id"] == original.request_id)
    if (origin.get("sample_id") != SOURCE_ID or origin.get("source_split") != "val"
            or origin.get("condition") != "clean" or origin.get("evaluation_eligible") is not True):
        raise ValueError("Spatial source must be the pinned eligible clean val scene")
    source = within(root, original.image_paths[0])
    if file_hash(source) != SOURCE_SHA256 or origin.get("image_sha256") != SOURCE_SHA256:
        raise ValueError("Spatial source SHA-256 changed")
    folder = within(root, INPUT_FOLDER)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    if image.size != (256, 256):
        raise ValueError("Pinned source must be 256x256")
    histogram = image.convert("L").histogram()
    cumulative, background = 0, 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= 256 * 256 // 2:
            background = value
            break
    patch = image.crop((32, 32, 224, 224))
    assets, controls = {}, []
    for x, label in ((16, "LEFT"), (160, "CENTER"), (304, "RIGHT")):
        canvas = Image.new("RGB", (512, 512), (background,) * 3)
        canvas.paste(patch, (x, 160))
        for mirrored in (False, True):
            control_id = "c" + str(len(controls)).zfill(2)
            rendered = ImageOps.mirror(canvas) if mirrored else canvas
            buffer = BytesIO()
            rendered.save(buffer, format="PNG")
            name = control_id + ".png"
            assets[name] = buffer.getvalue()
            expected = {"LEFT": "RIGHT", "CENTER": "CENTER", "RIGHT": "LEFT"}[label] if mirrored else label
            controls.append({"control_id": control_id, "image": INPUT_FOLDER + "/" + name,
                "archived_image": name, "image_sha256": hashlib.sha256(assets[name]).hexdigest(),
                "placement_xy_before_mirror": [x, 160], "mirrored": mirrored, "expected_third": expected})
    record = {"protocol": PROTOCOL, "source": origin, "source_image": original.image_paths[0],
        "source_sha256": SOURCE_SHA256, "bundle_sha256": bundle["bundle_sha256"],
        "source_crop_xyxy": [32, 32, 224, 224], "source_crater_anchor_xy": [128, 128],
        "canvas_wh": [512, 512], "background_rgb": [background] * 3, "pillow": pillow_version,
        "renderer_sha256": file_hash(Path(__file__)), "controls": controls,
        "reference_scope": "Predeclared, visually checkable development placement controls, not accepted Q1-Q4 GT. "
            "Composite padded/mirrored images are diagnostic inputs, not natural planetary benchmark images.",
        "scientific_result": False, "benchmark_admitted": False}
    metadata = folder / "controls.json"
    if folder.exists():
        if metadata.is_symlink() or not metadata.is_file() or json.loads(metadata.read_text(encoding="utf-8")) != record:
            raise ValueError("Existing spatial controls differ or are incomplete; use a fresh workspace")
        if any((folder / name).is_symlink() or not (folder / name).is_file()
               or (folder / name).read_bytes() != data for name, data in assets.items()):
            raise ValueError("Existing spatial image changed; use a fresh workspace")
    else:
        folder.mkdir(parents=True)
        for name, data in assets.items():
            with (folder / name).open("xb") as stream:
                stream.write(data)
        atomic_json(metadata, record)
    return record


def prepare_requests(root, controls):
    """Only opaque image IDs and prompts; evaluator targets never enter requests."""
    from planetary_vlm.colab import within
    requests, settings = [], {}
    for index, control in enumerate(controls["controls"]):
        for kind, prompt, budget in PROMPTS:
            key = "spatial_" + control["control_id"] + "__" + kind + "__v1"
            requests.append(ModelRequest(key, (str(within(root, control["image"])),), prompt, (), budget))
            settings[key] = {"conversation_template": "llava_v1", "use_cache": True,
                             "compare_first_forward": index < 2 and kind == "free"}
    return requests, settings


def summarize(rows, traces, controls, *, request_ids=None):
    """Exact-label evaluation of synthetic controls only; no parsing of prose."""
    from planetary_vlm.evaluation.parser import parse_answer
    by_id = {r["request_id"]: r for r in rows}
    results, checks = [], []
    for control in controls["controls"]:
        answers = {}
        for kind, _, _ in PROMPTS:
            key = "spatial_" + control["control_id"] + "__" + kind + "__v1"
            if request_ids is not None and key not in request_ids:
                continue
            row = by_id.get(key, {})
            answers[kind] = row.get("raw_response", "")
            if kind != "free":
                parsed = parse_answer(answers[kind], ("LEFT", "CENTER", "RIGHT")) if row.get("status") == "ok" else None
                checks.append({"request_id": key, "parsed": parsed, "expected": control["expected_third"],
                    "format_valid": parsed is not None, "correct": parsed == control["expected_third"]})
        if answers:
            results.append({"control_id": control["control_id"], "expected_third": control["expected_third"],
                            "mirrored": control["mirrored"], "answers": answers})
    return {"protocol": PROTOCOL, "completed": len(rows), "runtime_errors": sum(r["status"] != "ok" for r in rows),
        "token_budget_hits": sum(t.get("hit_token_budget") is True for t in traces.values()),
        "controls": results, "closed_control_checks": checks,
        "development_controls_correct": sum(c["correct"] for c in checks), "development_controls_total": len(checks),
        "free_responses_automatically_scored": False, "scientific_score_available": False, "benchmark_admitted": False,
        "interpretation": "Synthetic placement controls only, not scientific accuracy. Compare movement/mirroring and "
            "forward/reverse prompt pairs. Success alone does not prove order bias, correct grounding, or a model fix. "
            "No cache/template/precision ablation and no Q1-Q4 admission."}


def run_pilot(root, persistent, run_id, session_id, *, fresh_process=False):
    from planetary_vlm.colab import environment, digest, model_options, checkpoint
    from api.logging import get_logger
    root, persistent = Path(root).resolve(), Path(persistent).resolve()
    if "data_rover" in {p.casefold() for p in persistent.parts}:
        raise ValueError("Protected archive cannot contain pilot outputs")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", run_id) or not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise ValueError("Invalid spatial run/session ID")
    controls = prepare_controls(root)
    env = environment(root)
    requests, settings = prepare_requests(root, controls)
    if fresh_process:
        by_id = {r.request_id: r for r in requests}
        requests = [by_id[key] for key in FRESH_REQUEST_IDS]
        settings = {key: settings[key] for key in FRESH_REQUEST_IDS}
        if any(s["compare_first_forward"] for s in settings.values()):
            raise ValueError("Cold-start controls must have no preliminary forward probes")
    actual_id = run_id + ("_fresh_" if fresh_process else "_spatial_") + session_id[:8]
    destination, durable = root / "outputs/SpaceLLaVA" / actual_id, persistent / "runs/SpaceLLaVA" / actual_id
    if destination.exists() or durable.exists():
        raise ValueError("Spatial pilot exists; use a fresh session/run ID, never overwrite or resume")
    manifest = root / INPUT_FOLDER / (actual_id + "_requests.jsonl")
    write_jsonl(manifest, map(asdict, requests))
    options = {**model_options(root), "generation_diagnostics": True, "diagnostic_output_dir": str(destination),
               "diagnostic_request_settings": settings}
    execution = {"protocol": PROTOCOL, "session_id": session_id, "environment": env,
        "bundle_sha256": controls["bundle_sha256"], "settings": settings, "controls_sha256": digest(controls),
        "scientific_result": False, "benchmark_admitted": False}
    if fresh_process:
        execution.update(isolation_mode="fresh-process-control-v1", preliminary_inference=False,
                         request_ids=list(FRESH_REQUEST_IDS), independent_model_loads=len(requests))
    config = RunConfig(actual_id, manifest, root / "outputs/SpaceLLaVA", "remyxai/SpaceLLaVA", "spacellava",
                       options, 42, digest(execution), root / "colab_bundle.json")
    # Checkpoint the six small diagnostic PNGs once, including on load failure.
    atomic_json(durable / "spatial_controls.json", controls)
    atomic_json(durable / "spatial_execution.json", execution)
    for control in controls["controls"]:
        target = durable / control["archived_image"]
        shutil.copyfile(root / control["image"], target)
        if file_hash(target) != control["image_sha256"]:
            raise OSError("Spatial image checkpoint failed SHA-256 verification")
    logger = get_logger("colab.spatial_pilot", run_id=actual_id)
    def publish(folder):
        checkpoint(folder, durable)
        if fresh_process:
            # All nested paths are generated job/worker logs in this result,
            # never source data. Preserve them for process-level provenance.
            for nested in sorted(folder.rglob("*")):
                if nested.is_symlink():
                    raise ValueError("Symlink in fresh-process result")
                if nested.is_dir():
                    checkpoint(nested, durable / nested.relative_to(folder))
    def initialized(adapter, folder):
        from benchmark_services.spacellava.diagnostics import diagnose
        paths = [str(root / c["image"]) for c in controls["controls"][:2]]
        report = diagnose(adapter, paths)
        atomic_json(folder / "vision_diagnostics.json", report)
        if report["success"] is not True:
            raise RuntimeError("Visual diagnostics failed before spatial pilot")
    def progress(folder, completed, total):
        logger.info("spatial_request_persisted", completed=completed, total=total)
        atomic_json(folder / "spatial_execution.json", execution)
        publish(folder)
    logger.info("spatial_pilot_started", images=2 if fresh_process else 6, requests=len(requests),
                fresh_process=fresh_process, benchmark_admitted=False)
    try:
        if fresh_process:
            from planetary_vlm.inference.fresh_process import run_fresh
            result = run_fresh(config, on_response=progress)
        else:
            result = run(config, on_initialized=initialized, on_response=progress)
    finally:
        if destination.is_dir():
            atomic_json(destination / "spatial_execution.json", execution)
            atomic_json(destination / "spatial_controls.json", controls)
            publish(destination)
    rows = read_jsonl(result / "predictions.jsonl")
    traces = json.loads((result / "generation_diagnostics.json").read_text(encoding="utf-8"))
    report = summarize(rows, traces, controls, request_ids=set(FRESH_REQUEST_IDS) if fresh_process else None)
    if fresh_process:
        report.update(isolation_mode="fresh-process-control-v1", preliminary_inference=False,
                      independent_model_loads=len(requests),
                      interpretation="Every selected request is the first predict call in a newly loaded process. "
                      "Same four v8 prompts/images and 4-bit loading; no preliminary model forwards. "
                      "Compare archived v8 answers separately; changes alone do not prove a cache bug. "
                      "No scientific accuracy or benchmark admission.")
    atomic_json(result / "spatial_report.json", report)
    publish(result)
    # No smoke marker or scientific reference admission, regardless of answers.
    return {"success": len(rows) == len(requests) and all(r["status"] == "ok" for r in rows), "requests": len(requests),
            "output_dir": str(result), "persistent_dir": str(durable),
            "scientific_score_available": False, "benchmark_admitted": False}
