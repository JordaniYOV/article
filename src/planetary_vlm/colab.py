"""Portable, image-only orbital bundles and interactive Colab batch runs.

No website, SQLite, training, archive access or implicit weight acquisition.
Export is CPU-only. The remaining commands run inside the notebook's isolated
SpaceLLaVA environment; GPU compatibility is admitted only by a real smoke test.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import zipfile

from planetary_vlm.contracts import ModelRequest
from planetary_vlm.io import read_jsonl
from planetary_vlm.inference.runner import file_hash
from benchmark_services.common.files import atomic_json

SCHEMA = "planetary-colab-v1"
LLAVA_COMMIT = "c121f0432da27facab705978f83c4ada465e46fd"
MODEL_REVISION = "5a69f71bda9dea3e4d8113c1047c07fd3da483f8"
VISION_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
MODEL_ID = "remyxai/SpaceLLaVA"
VISION_ID = "openai/clip-vit-large-patch14-336"
CONDITIONS = ("clean", "shear", "radiation", "lens_flare", "hard_shadows", "combined")
PARAMETERS = dict(max_shift_px=4.0, control_spacing_px=32, scan_axis="both",
                  num_hits=80, max_streak_length=10, flare_strength=.55, shadow_strength=.65)
PROMPT = ("Describe the visible surface features in this planetary orbital image. "
          "Give approximate image locations and state uncertainty when details are unclear. "
          "Do not infer unseen features, physical hazards or traversability from appearance alone.")
PROVENANCE_FIELDS = ("sample_id", "planet", "track", "source_dataset", "source_revision", "image_source",
    "source_split", "scene_group_id", "source_image_id", "license_claim", "display_transform",
    "evaluation_eligible", "final_test_eligible", "height_units")


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_relative(value):
    if not isinstance(value, str):
        raise ValueError("Unsafe bundle path")
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or path.is_absolute() or ".." in path.parts
        or "\\" in value or ":" in value or any(ord(c) < 32 for c in value)
        or any(part.casefold() in {"data_rover", ".env", ".git", "weights", "external"} for part in path.parts)):
        raise ValueError("Unsafe bundle path")
    return path


def within(root, value):
    root = Path(root).resolve()
    candidate = (root / safe_relative(value)).resolve()
    if not candidate.is_relative_to(root) or "data_rover" in {p.casefold() for p in candidate.parts}:
        raise ValueError("Bundle path escapes its workspace")
    return candidate


def export_bundle(root, output, *, conditions=CONDITIONS, seed=42, prompt=PROMPT, max_tokens=128, questions=None,
                  crater_questions=None, collection="data_orbital/orbital_300_v3",
                  requirements_profiles=("requirements/spacellava.txt",), source_ids=None,
                  inference_cohort=None):
    """Prepare image inputs; an optional local question build stays GT-free in ZIP."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if "data_rover" in {p.casefold() for p in root.parts} or not output.is_relative_to(root / "outputs" / "colab"):
        raise ValueError("Export must use this workspace and outputs/colab")
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(CONDITIONS):
        raise ValueError("Unknown, duplicate or empty conditions")
    ModelRequest("validate", (), prompt, (), max_new_tokens=max_tokens)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative")
    if (not requirements_profiles or len(set(requirements_profiles)) != len(requirements_profiles)
        or any(safe_relative(p).parent != PurePosixPath("requirements") or not p.endswith(".txt")
               for p in requirements_profiles)):
        raise ValueError("Requirements profiles must be distinct requirements/*.txt paths")
    if source_ids is not None:
        if (not isinstance(source_ids, (list, tuple)) or not source_ids
            or any(not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", s) for s in source_ids)
            or len(set(source_ids)) != len(source_ids)):
            raise ValueError("source_ids must be distinct nonempty sample IDs")
    if inference_cohort is not None:
        fields = {"schema", "selection_version", "pilot_sample_ids", "smoke_sample_ids"}
        if not isinstance(inference_cohort, dict) or set(inference_cohort) != fields or source_ids is None:
            raise ValueError("Inference cohort must use the image-only allowlist and explicit source_ids")
        if any(not isinstance(inference_cohort[k], str) or not inference_cohort[k] for k in ("schema", "selection_version")):
            raise ValueError("Inference cohort needs schema and selection_version")
        for key in ("pilot_sample_ids", "smoke_sample_ids"):
            values = inference_cohort[key]
            if (not isinstance(values, list) or not values or any(not isinstance(s, str) for s in values)
                or len(set(values)) != len(values)):
                raise ValueError("Inference cohort IDs must be distinct strings")
        pilot, smoke = map(set, (inference_cohort["pilot_sample_ids"], inference_cohort["smoke_sample_ids"]))
        if pilot & smoke or pilot | smoke != set(source_ids):
            raise ValueError("Cohort must exactly partition the explicit source_ids")
    question_rows, question_metadata = [], None
    if questions is not None and crater_questions is not None:
        raise ValueError("Select one question build, not mixed height/crater inputs")
    if questions is not None or crater_questions is not None:
        if tuple(conditions) != ("clean",):
            raise ValueError("Question pilot supports clean only; geometric corruptions need transformed reference geometry")
        if crater_questions is not None:
            from planetary_vlm.datasets.crater_annotations import load_crater_requests
            question_rows, question_metadata = load_crater_requests(root, crater_questions)
        else:
            from planetary_vlm.datasets.orbital_questions import load_question_requests
            question_rows, question_metadata = load_question_requests(root, questions)
    by_source = {}
    for request in question_rows:
        by_source.setdefault(request.request_id.split("__")[0], []).append(request)
    collection = within(root, str(collection))
    if collection == root/"data_orbital" or not collection.is_relative_to(root/"data_orbital"):
        raise ValueError("Use a versioned data_orbital collection")
    audit = json.loads((collection/"audit.json").read_text(encoding="utf-8"))
    selected, seen, excluded, manifests = [], set(), Counter(), {}
    for planet in ("moon", "mars"):
        for track in ("segmentation", "height", "train"):
            path = collection / planet / (track+"_manifest.jsonl")
            manifests[str(path.relative_to(root)).replace("\\", "/")] = file_hash(path)
            for row in read_jsonl(path):
                sid = row.get("sample_id")
                if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sid) or sid in seen:
                    raise ValueError("Source sample IDs must be safe and globally unique")
                seen.add(sid)
                if (track == "train" or row.get("source_split") not in {"val", "test"}
                    or row.get("evaluation_eligible") is not True or row.get("model_inference_use") is False):
                    excluded["publisher_train_or_ineligible"] += 1
                    continue
                image = within(root, row["image"])
                if not image.is_relative_to(collection) or not image.is_file() or file_hash(image) != row["image_sha256"]:
                    raise ValueError(f"Missing or changed source image: {sid}")
                selected.append((row, image))
    if len(seen) != audit["total_images"] or not selected:
        raise ValueError("Collection counts differ from its audit, or no admitted inputs exist")
    if question_metadata:
        if any(manifests.get(name) != checksum for name,checksum in question_metadata["source_manifests_sha256"].items()):
            raise ValueError("Question source manifests differ from collection")
        selected_ids = {r["sample_id"] for r, _ in selected}
        if not set(by_source) <= selected_ids:
            raise ValueError("Questions contain train or ineligible sources")
        excluded["no_admitted_question"] = sum(row["sample_id"] not in by_source for row, _ in selected)
        selected = [(row, image) for row, image in selected if row["sample_id"] in by_source]
        for row, _ in selected:
            if any(r.image_paths != (row["image"],) for r in by_source[row["sample_id"]]):
                raise ValueError("Question source image differs from selected collection image")
    if source_ids is not None:
        available_ids = {row["sample_id"] for row, _ in selected}
        if not set(source_ids) <= available_ids:
            raise ValueError("source_ids contain train, ineligible, unknown or questionless samples")
        excluded["outside_explicit_cohort"] = len(selected)-len(source_ids)
        selected = [(row, image) for row, image in selected if row["sample_id"] in set(source_ids)]
        question_rows = [r for r in question_rows if r.request_id.split("__")[0] in set(source_ids)]
    selected.sort(key=lambda pair: pair[0]["sample_id"])
    from planetary_vlm.inference.full_systems import transform
    from PIL import Image
    import numpy as np
    renderer = {name: metadata.version(name) for name in ("numpy", "Pillow", "opencv-python-headless")}
    renderer["code_sha256"] = file_hash(Path(__file__).with_name("inference") / "full_systems.py")
    protocol = {"collection": collection.name, "source_manifests_sha256": manifests,
        "conditions": list(conditions), "seed": seed, "prompt": prompt, "max_new_tokens": max_tokens,
        "parameters": PARAMETERS, "renderer": renderer, "source_ids": [r["sample_id"] for r, _ in selected],
        "question_type": "freeform_description_not_a_scored_Q1_Q8_task"}
    if inference_cohort is not None:
        protocol["inference_cohort"] = inference_cohort
    if question_metadata:
        protocol.update(question_type=question_metadata.get("question_type","reference_ready_Q5_Q6_clean_pilot"),
            questions_sha256=question_metadata["bundle_sha256"], question_rules=question_metadata["rules"],
            prompt_templates={r.request_id.split("__")[1]+"/"+digest(r.prompt)[:12]: r.prompt for r in question_rows})
        protocol.pop("prompt")
        protocol["max_new_tokens"] = 16
    output.parent.mkdir(parents=True, exist_ok=True)
    files, requests, provenance = {}, [], []
    # Exclusive creation: a previous export is never silently replaced.
    with output.open("xb") as target:
        try:
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                def put(name, data):
                    safe_relative(name)
                    if name in files:
                        raise ValueError("Duplicate bundle entry")
                    archive.writestr(name, data)
                    files[name] = sha256(data).hexdigest()
                for base in (root / "src" / "planetary_vlm", root / "src" / "benchmark_services"):
                    for path in sorted(base.rglob("*")):
                        relative = str(path.relative_to(root)).replace("\\", "/")
                        if "__pycache__" in path.parts or "example" in path.parts:
                            continue
                        safe_relative(relative)
                        if path.suffix in {".py", ".toml", ".ps1"}:
                            if path.is_symlink() or not path.resolve().is_relative_to(base):
                                raise ValueError("Symlink/out-of-source bundle entry")
                            put(relative, path.read_bytes())
                for name in ("src/api/__init__.py", "src/api/logging.py", "pyproject.toml", *requirements_profiles):
                    put(name, within(root, name).read_bytes())
                for index, (row, image) in enumerate(selected, 1):
                    sid = row["sample_id"]
                    image_seed = int.from_bytes(sha256(f"{seed}:{sid}".encode()).digest()[:4], "big")
                    with Image.open(image) as source:
                        pixels = np.asarray(source.convert("RGB"))
                    for condition in conditions:
                        name = f"data_orbital/colab_inputs/images/{sid}/{condition}.png"
                        if condition == "clean":
                            data = image.read_bytes()
                        else:
                            changed = transform(pixels, condition, image_seed, PARAMETERS)
                            stream = BytesIO()
                            Image.fromarray(changed).save(stream, format="PNG")
                            data = stream.getvalue()
                        put(name, data)
                        from dataclasses import replace
                        prepared = ([replace(r, image_paths=(name,)) for r in by_source[sid]] if question_metadata else
                            [ModelRequest(sid + "__" + condition, (name,), prompt, (), max_new_tokens=max_tokens)])
                        for request in prepared:
                            requests.append(asdict(request))
                            provenance.append({**{k: row[k] for k in PROVENANCE_FIELDS if k in row},
                                "request_id": request.request_id, "condition": condition, "seed": image_seed,
                                "image_sha256": files[name], "source_image_sha256": row["image_sha256"],
                                **({"task_id": request.request_id.split("__")[1]} if question_metadata else {})})
                    if index % 25 == 0:
                        print(json.dumps({"prepared_sources": index, "total_sources": len(selected)}), flush=True)
                for name, rows in (("requests.jsonl", requests), ("provenance.jsonl", provenance)):
                    put("data_orbital/colab_inputs/" + name,
                        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
                manifest = {"schema": SCHEMA, "protocol": protocol, "protocol_sha256": digest(protocol),
                    "collection_images": len(seen), "selected_images": len(selected), "requests": len(requests),
                    "excluded": dict(excluded), "counts": dict(Counter(f"{r['planet']}/{r['track']}/{r['source_split']}" for r, _ in selected)),
                    "files_sha256": files, "scientific_result": False,
                    "limitation": "Freeform outputs have no automatic accuracy score. Val is development, not final test. "
                                  "No mask/height/reference answer is included; height images are RGB only. "
                                  "Current distortions are shear/radiation/flare/shadows/combined, not blur/noise E2."}
                if question_metadata:
                    manifest["limitation"] = ("Clean RGB-only closed-form question inference. Targets remain in the local "
                        "question folder, not this inference bundle. No IBM pairwise admission or scientific result. "
                        "Val is development; source/model overlap and physical-height alignment audits remain open.")
                    if crater_questions is not None:
                        manifest["limitation"] = ("Clean RGB-only Q1-Q4 CANDIDATE inference, not accepted scoring GT. "
                            "Candidate answers, masks, height arrays and targets are excluded. Complete candidates do not "
                            "imply final reference admission or IBM pairing. Preserve source val/test and scene groups.")
                manifest["bundle_sha256"] = digest(manifest)
                archive.writestr("colab_bundle.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode())
        except BaseException:
            target.close()
            output.unlink(missing_ok=True)  # Only the new, exclusively created failed export.
            raise
    return {"output": str(output), "selected_images": len(selected), "requests": len(requests),
            "excluded": dict(excluded), "bundle_sha256": manifest["bundle_sha256"]}


def verify_bundle(root):
    root = Path(root).resolve()
    if "data_rover" in {p.casefold() for p in root.parts}:
        raise ValueError("Protected archive cannot be a Colab workspace")
    record = json.loads((root / "colab_bundle.json").read_text(encoding="utf-8"))
    claimed = record.pop("bundle_sha256")
    if record["schema"] != SCHEMA or digest(record) != claimed:
        raise ValueError("Bundle metadata changed")
    record["bundle_sha256"] = claimed
    for name, expected in record["files_sha256"].items():
        path = within(root, name)
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Bundle file missing or changed: {name}")
    return record


def acquire_assets(root, *, download=False, clone_runtime=False):
    root = Path(root).resolve()
    if "data_rover" in {p.casefold() for p in root.parts}:
        raise ValueError("Protected archive cannot be a Colab workspace")
    repo = root / "external" / "LLaVA"
    if not repo.exists():
        if not (download or clone_runtime):
            raise ValueError("Runtime/weights absent: set DOWNLOAD_WEIGHTS=True only when ready for ~28 GB download")
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/haotian-liu/LLaVA.git", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", LLAVA_COMMIT], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != LLAVA_COMMIT:
        raise ValueError("Existing LLaVA checkout is at a different revision; do not silently reset it")
    if download:
        if shutil.disk_usage(root).free < 35 * 1024 ** 3:
            raise ValueError("Need at least 35 GiB free local disk before downloading full checkpoints")
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(root / "weights" / "SpaceLLaVA"),
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"])
        snapshot_download(VISION_ID, revision=VISION_REVISION, local_dir=str(root / "weights" / "clip-vit-large-patch14-336"),
            allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"])
    from benchmark_services.spacellava.adapter import SpaceLLaVAAdapter
    SpaceLLaVAAdapter(model_options(root)).preflight()


def model_options(root):
    return {"runtime_repo": str(root / "external" / "LLaVA"), "code_revision": LLAVA_COMMIT,
        "model_dir": str(root / "weights" / "SpaceLLaVA"), "model_revision": MODEL_REVISION,
        "vision_dir": str(root / "weights" / "clip-vit-large-patch14-336"), "vision_revision": VISION_REVISION,
        "local_files_only": True, "quantization": "4bit", "device_map": "auto"}


def environment(root):
    """Small CUDA operation, never a model load. Notebook runs this explicitly."""
    from planetary_vlm.inference.runner import dependency_versions
    if "data_rover" in {p.casefold() for p in Path(root).resolve().parts}:
        raise ValueError("Protected archive cannot be a Colab workspace")
    import torch
    if sys.version_info[:2] != (3, 12) or not torch.cuda.is_available():
        raise ValueError("Need isolated Python 3.12 and a working CUDA GPU")
    value = torch.ones((2, 2), device="cuda:0")
    (value @ value).cpu()
    props = torch.cuda.get_device_properties(0)
    result = {"dependencies": dependency_versions(), "cuda": torch.version.cuda,
        "gpu": props.name, "gpu_memory_bytes": props.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "pip_freeze": sorted(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines())}
    result["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    atomic_json(Path(root) / "colab_environment.json", result)
    return result


def checkpoint(local, persistent):
    """Publish whole files atomically; never append JSONL directly on Drive/FUSE."""
    local, persistent = Path(local), Path(persistent)
    if local.resolve() == persistent.resolve() or persistent.resolve().is_relative_to(local.resolve()):
        raise ValueError("Checkpoint destination must be separate from local results")
    persistent.mkdir(parents=True, exist_ok=True)
    if not local.is_dir():
        return
    for path in local.iterdir():
        if path.is_file() and path.suffix in {".json", ".jsonl", ".log", ".txt"}:
            target = persistent / path.name
            temporary = persistent / (path.name + ".uploading")
            shutil.copyfile(path, temporary)
            if file_hash(temporary) != file_hash(path):
                raise OSError("Checkpoint copy failed integrity verification")
            # Windows file scanners/readers may briefly lock an existing target.
            # Retry only PermissionError, preserve atomic replacement, and fail
            # after a bounded delay rather than losing a completed checkpoint.
            for attempt in range(6):
                try:
                    os.replace(temporary, target)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(.05 * (attempt + 1))


def restore_run(local, persistent):
    local, persistent = Path(local), Path(persistent)
    if local.exists() or not persistent.is_dir():
        return
    local.mkdir(parents=True)
    for path in persistent.iterdir():
        if path.is_file() and path.suffix in {".json", ".jsonl", ".log", ".txt"}:
            target = local / path.name
            shutil.copyfile(path, target)
            if file_hash(target) != file_hash(path):
                raise OSError("Restored result failed integrity verification")


def run_job(root, persistent, run_id, session_id, *, smoke=False, resume=False, checkpoint_every=10,
            max_requests=None, smoke_controls=None):
    """Only SpaceLLaVA, one subprocess/GPU and image-only requests; no fake scores."""
    from dataclasses import replace
    from planetary_vlm.datasets.manifest import load_requests
    from planetary_vlm.inference.config import RunConfig
    from planetary_vlm.inference.runner import run
    from api.logging import get_logger
    root, persistent = Path(root).resolve(), Path(persistent).resolve()
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", run_id)
        or not re.fullmatch(r"[a-f0-9]{32}", session_id) or checkpoint_every < 1
        or max_requests is not None and max_requests < 1):
        raise ValueError("Invalid run/session/checkpoint options")
    bundle = verify_bundle(root)
    env = environment(root)
    env_hash = digest(env)
    provenance = read_jsonl(root / "data_orbital" / "colab_inputs" / "provenance.jsonl")
    original = load_requests(root / "data_orbital" / "colab_inputs" / "requests.jsonl", resolve_paths=False)
    controls = None
    if bundle["protocol"]["question_type"] == "candidate_Q1_Q4_clean_unscored":
        from planetary_vlm.colab_smoke import load_controls
        controls = load_controls(root, smoke_controls, original, provenance)
        if smoke and (resume or max_requests is not None):
            raise ValueError("Two-image smoke cannot resume or truncate its ten fixed requests; use a fresh session ID")
    proof_path = root / "colab_smoke_passed.json"
    if not smoke:
        proof = json.loads(proof_path.read_text(encoding="utf-8")) if proof_path.exists() else {}
        if (proof.get("session_id") != session_id or proof.get("environment_sha256") != env_hash
            or proof.get("bundle_sha256") != bundle["bundle_sha256"] or proof.get("success") is not True):
            raise ValueError("A successful smoke test in this session/environment/bundle is required")
        if controls:
            from benchmark_services.spacellava.diagnostics import PROTOCOL
            if (proof.get("smoke_protocol") != controls["schema"]
                    or proof.get("controls_sha256") != controls["sha256"]
                    or proof.get("vision_diagnostic_protocol") != PROTOCOL):
                raise ValueError("A fresh two-image diagnostic smoke test is required; old one-image proofs are not admitted")
            smoke_result = Path(proof.get("result", "")).resolve()
            if not smoke_result.is_relative_to(root / "outputs" / "SpaceLLaVA"):
                raise ValueError("Smoke evidence is outside this workspace")
            for name, key in (("vision_diagnostics.json", "vision_diagnostics_sha256"),
                              ("smoke_assessment.json", "smoke_assessment_sha256")):
                evidence = smoke_result / name
                if not evidence.is_file() or file_hash(evidence) != proof.get(key):
                    raise ValueError("Smoke diagnostic evidence missing or changed; run a fresh smoke test")
    if smoke:
        candidates = {p["request_id"] for p in provenance if p["source_split"] == "val" and p["condition"] == "clean"}
        selected = controls["requests"] if controls else [r for r in original if r.request_id in candidates][:1]
        actual_id = run_id + "_smoke_" + session_id[:8]
    else:
        selected = original[:max_requests] if max_requests else original
        actual_id = run_id
    if not selected:
        raise ValueError("No admitted requests (smoke requires one clean val image)")
    selected = [replace(r, image_paths=tuple(str(within(root, p)) for p in r.image_paths)) for r in selected]
    requests_path = root / "data_orbital" / "colab_inputs" / (actual_id + "_requests.jsonl")
    request_text = "".join(json.dumps(asdict(r), ensure_ascii=False, allow_nan=False) + "\n" for r in selected)
    if requests_path.exists():
        if requests_path.read_text(encoding="utf-8") != request_text:
            raise ValueError("Prepared request manifest changed; use a new run ID")
    else:
        requests_path.parent.mkdir(parents=True, exist_ok=True)
        requests_path.write_text(request_text, encoding="utf-8", newline="\n")
    output_root = root / "outputs" / "SpaceLLaVA"
    destination, durable = output_root / actual_id, persistent / "runs" / "SpaceLLaVA" / actual_id
    if resume:
        restore_run(destination, durable)
    if destination.exists():
        saved = json.loads((destination / "colab_execution.json").read_text(encoding="utf-8"))
        if (not resume or saved["environment_sha256"] != env_hash
            or saved["bundle_sha256"] != bundle["bundle_sha256"]):
            raise ValueError("Resume refused: environment/bundle changed, or resume not requested")
    execution = {"environment": env, "environment_sha256": env_hash,
        "bundle_sha256": bundle["bundle_sha256"], "backend": "spacellava", "mock": False,
        "source_counts": bundle["counts"], "checkpoint_every": checkpoint_every,
        "note": ("Q1-Q4 candidate-only inference. Final scoring references remain pending, never inferred from predictions."
                 if bundle["protocol"]["question_type"] == "candidate_Q1_Q4_clean_unscored" else
                 "Closed-form Q5/Q6 pilot. Evaluate locally against separate targets; IBM pairing is not admitted."
                 if bundle["protocol"]["question_type"] == "reference_ready_Q5_Q6_clean_pilot" else
                 "Freeform perception outputs, not automatic Q1-Q8 accuracy or hallucination scores")}
    if controls:
        execution["smoke_controls_sha256"] = controls["sha256"]
    signature = digest({"bundle": bundle["bundle_sha256"], "request_ids": [r.request_id for r in selected],
        "model_options": model_options(root), "run_id": actual_id, "output_root": str(output_root),
        **({"smoke_controls_sha256": controls["sha256"]} if controls else {})})
    config = RunConfig(actual_id, requests_path, output_root, MODEL_ID, "spacellava", model_options(root),
                       bundle["protocol"]["seed"], signature, root / "colab_bundle.json")
    logger = get_logger("colab.runner", run_id=actual_id, backend="spacellava", mock=False)
    def diagnose_loaded(adapter, folder):
        from benchmark_services.spacellava.diagnostics import diagnose
        logger.info("visual_diagnostics_started", images=2, labels_passed_to_model=False)
        report = diagnose(adapter, controls["images"])
        atomic_json(folder / "vision_diagnostics.json", report)
        logger.info("visual_diagnostics_finished", success=report["success"])
        if report["success"] is not True:
            raise RuntimeError("Visual diagnostics failed; inspect vision_diagnostics.json before generation")
    def progress(folder, completed, total):
        logger.info("request_persisted", completed=completed, total=total)
        if completed % checkpoint_every == 0:
            atomic_json(folder / "colab_execution.json", execution)
            (folder / "pip-freeze.txt").write_text("\n".join(env.get("pip_freeze", [])) + "\n", encoding="utf-8")
            checkpoint(folder, durable)
    logger.info("smoke_started" if smoke else "benchmark_started", total=len(selected))
    if smoke:
        # A failed new attempt must not leave a previous successful marker live.
        atomic_json(proof_path, {"success": False, "session_id": session_id,
            "environment_sha256": env_hash, "bundle_sha256": bundle["bundle_sha256"]})
    try:
        result = run(config, resume=resume, on_response=progress,
                     on_initialized=diagnose_loaded if smoke and controls else None)
    finally:
        if destination.is_dir():
            atomic_json(destination / "colab_execution.json", execution)
            (destination / "pip-freeze.txt").write_text("\n".join(env.get("pip_freeze", [])) + "\n", encoding="utf-8")
            shutil.copyfile(root / "data_orbital" / "colab_inputs" / "provenance.jsonl", destination / "provenance.jsonl")
            atomic_json(destination / "bundle_metadata.json", {key: value for key, value in bundle.items() if key != "files_sha256"})
            if controls:
                atomic_json(destination / "smoke_controls.json", controls["record"])
        checkpoint(destination, durable)
    rows = read_jsonl(result / "predictions.jsonl")
    counts = dict(Counter(row["status"] for row in rows))
    success = len(rows) == len(selected) and all(r["status"] == "ok" and r["raw_response"].strip() and not r["mock"] for r in rows)
    if smoke:
        loading = json.loads((result / "loading.json").read_text(encoding="utf-8"))
        # A CPU/disk-offloaded run is not admitted as the intended single-GPU pilot.
        device_map = loading["provenance"].get("device_map", "")
        success = success and "cpu" not in device_map.lower() and "disk" not in device_map.lower()
        if any(r.allowed_answers for r in selected):
            from planetary_vlm.evaluation.parser import parse_answer
            allowed = {r.request_id: r.allowed_answers for r in selected}
            success = success and all(parse_answer(r["raw_response"], allowed[r["request_id"]]) is not None for r in rows)
        extra_proof = {}
        if controls:
            from planetary_vlm.colab_smoke import assess
            diagnostics = json.loads((result / "vision_diagnostics.json").read_text(encoding="utf-8"))
            assessment = assess(rows, selected, controls["expected"], diagnostics)
            success = success and assessment["success"]
            atomic_json(result / "smoke_assessment.json", assessment)
            extra_proof = {"smoke_protocol": controls["schema"], "controls_sha256": controls["sha256"],
                "vision_diagnostic_protocol": diagnostics["protocol"],
                "vision_diagnostics_sha256": file_hash(result / "vision_diagnostics.json"),
                "smoke_assessment_sha256": file_hash(result / "smoke_assessment.json")}
            logger.info("smoke_controls_assessed", success=success,
                invalid_formats=sum(not c["format_valid"] for c in assessment["checks"]),
                incorrect_presence_controls=sum(c["control_correct"] is False for c in assessment["checks"]))
            checkpoint(result, durable)
        atomic_json(proof_path, {"success": success, "session_id": session_id, "environment_sha256": env_hash,
            "bundle_sha256": bundle["bundle_sha256"], "result": str(result), "counts": counts, **extra_proof})
    logger.info("smoke_finished" if smoke else "benchmark_finished", completed=len(rows), total=len(selected),
                status="ok" if success else "incomplete")
    return {"success": success, "counts": counts, "output_dir": str(result), "persistent_dir": str(durable),
            "scientific_score_available": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="CPU-only versioned orbital image bundle, no downloads")
    export.add_argument("--workspace", default=str(Path(__file__).resolve().parents[2]))
    export.add_argument("--output", required=True)
    export.add_argument("--clean-only", action="store_true")
    export.add_argument("--prompt", default=PROMPT)
    export.add_argument("--max-new-tokens", type=int, default=128)
    export.add_argument("--questions", help="Local orbital-questions build folder; requires --clean-only; no targets in ZIP")
    export.add_argument("--crater-questions", help="Complete candidate Q1-Q4 build; unscored RGB-only inference; requires --clean-only")
    export.add_argument("--collection", default="data_orbital/orbital_300_v3")
    for name in ("verify", "assets", "environment", "smoke", "run", "language-pilot", "spatial-prepare", "spatial-pilot", "spatial-fresh", "evidence-pilot", "reference-pilot", "precision-preflight", "precision-pilot"):
        item = commands.add_parser(name)
        item.add_argument("--workspace", required=True)
        if name == "assets":
            item.add_argument("--download", action="store_true")
            item.add_argument("--clone-runtime", action="store_true")
        if name in {"smoke", "run"}:
            item.add_argument("--persist", required=True)
            item.add_argument("--run-id", required=True)
            item.add_argument("--session-id", required=True)
            item.add_argument("--resume", action="store_true")
            item.add_argument("--checkpoint-every", type=int, default=10)
            item.add_argument("--max-requests", type=int)
            item.add_argument("--smoke-controls", help="Separate evaluator-only YES/NO val control JSON, required for Q1-Q4")
        if name in {"language-pilot", "spatial-pilot", "spatial-fresh", "evidence-pilot", "reference-pilot", "precision-pilot"}:
            item.add_argument("--persist", required=True)
            item.add_argument("--run-id", required=True)
            item.add_argument("--session-id", required=True)
        if name in {"reference-pilot", "precision-pilot"}:
            item.add_argument("--baseline", required=True, help="Exact audited v10 result ZIP; no new weights")
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_bundle(args.workspace, args.output, conditions=("clean",) if args.clean_only else CONDITIONS,
                               prompt=args.prompt, max_tokens=args.max_new_tokens, questions=args.questions,
                               crater_questions=args.crater_questions, collection=args.collection)
    elif args.command == "verify":
        record = verify_bundle(args.workspace)
        result = {k: record[k] for k in ("selected_images", "requests", "excluded", "counts", "bundle_sha256")}
    elif args.command == "assets":
        acquire_assets(args.workspace, download=args.download, clone_runtime=args.clone_runtime)
        result = {"assets_available": True, "gpu_verified": False}
    elif args.command == "environment":
        result = environment(args.workspace)
    elif args.command == "language-pilot":
        from planetary_vlm.colab_language import run_pilot
        result = run_pilot(args.workspace,args.persist,args.run_id,args.session_id)
    elif args.command == "spatial-prepare":
        from planetary_vlm.colab_spatial import prepare_controls
        controls = prepare_controls(args.workspace)
        result = {"protocol": controls["protocol"], "images": len(controls["controls"]), "requests": 18,
                  "scientific_score_available": False, "benchmark_admitted": False}
    elif args.command == "spatial-pilot":
        from planetary_vlm.colab_spatial import run_pilot
        result = run_pilot(args.workspace,args.persist,args.run_id,args.session_id)
    elif args.command == "spatial-fresh":
        from planetary_vlm.colab_spatial import run_pilot
        result = run_pilot(args.workspace,args.persist,args.run_id,args.session_id,fresh_process=True)
    elif args.command == "evidence-pilot":
        from planetary_vlm.colab_evidence import run_pilot
        result = run_pilot(args.workspace,args.persist,args.run_id,args.session_id)
    elif args.command == "reference-pilot":
        from planetary_vlm.colab_reference import run_pilot
        result = run_pilot(args.workspace,args.persist,args.run_id,args.session_id,args.baseline)
    elif args.command == "precision-preflight":
        from planetary_vlm.colab_precision import preflight
        result = preflight(args.workspace)
    elif args.command == "precision-pilot":
        from planetary_vlm.colab_precision import run_pair
        result = run_pair(args.workspace,args.persist,args.run_id,args.session_id,args.baseline)
    else:
        result = run_job(args.workspace, args.persist, args.run_id, args.session_id, smoke=args.command == "smoke",
                        resume=args.resume, checkpoint_every=args.checkpoint_every, max_requests=args.max_requests,
                        smoke_controls=args.smoke_controls)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("success") is False: return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
