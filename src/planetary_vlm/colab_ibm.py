"""Independent IBM crater Colab pilot. No training, labels, web API or LLM.

Asset acquisition requires --download. Smoke is execution/visual QA, not accuracy.
"""
import argparse
from dataclasses import asdict
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

from benchmark_services.common.files import atomic_json
from benchmark_services.common.local_runtime import checked_repo
from benchmark_services.ibm.crater import (IBMCraterDetector, BACKBONE_ID, BACKBONE_REVISION,
    BACKBONE_SHA256, BACKBONE_CONFIG_SHA256, CODE_REVISION, CONFIG_SHA256, HEAD_FILE, HEAD_ID, HEAD_REVISION,
    HEAD_SHA256, PROTOCOL, SCORE_THRESHOLD)
from planetary_vlm.colab import digest, verify_bundle, within
from planetary_vlm.contracts import ModelResponse
from planetary_vlm.datasets.crater_reference import TASKS, ANSWERS
from planetary_vlm.inference.runner import file_hash
from planetary_vlm.io import read_jsonl

SCHEMA = "ibm-craters-colab-v1"
SMOKE_IDS = ("moon_seg_002", "moon_seg_012")  # Existing dev probes, not new test tuning.


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
        raise ValueError("Use a simple run/session/sample ID")
    return value


def options(root):
    root = Path(root).resolve()
    return {"runtime_repo": str(root/"external/IBM-Lunar"), "code_revision": CODE_REVISION,
        "local_files_only": True, "input_protocol": PROTOCOL, "device": "cuda:0",
        "head_config_path": str(root/"weights/IBM-Craters/NAC_config.yaml"),
        "head_checkpoint_path": str(root/"weights/IBM-Craters"/HEAD_FILE),
        "backbone_config_path": str(root/"weights/IBM-Lunar/backbone/config.yaml"),
        "backbone_checkpoint_path": str(root/"weights/IBM-Lunar/backbone/checkpoint.pt")}


def acquire_assets(root, *, download=False):
    root = Path(root).resolve()
    verify_bundle(root)
    opts = options(root)
    repo = Path(opts["runtime_repo"])
    if not repo.exists():
        if not download:
            raise ValueError("IBM runtime absent: explicitly enable DOWNLOAD_WEIGHTS first")
        repo.mkdir(parents=True)
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        for args in (("init",), ("fetch", "--depth", "1",
                "https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model.git", CODE_REVISION),
                ("checkout", "--detach", "FETCH_HEAD")):
            subprocess.run(["git", "-C", str(repo), *args], check=True, env=env)
    checked_repo(opts, "terratorch_integration")
    assets = ((HEAD_ID, HEAD_REVISION, "NAC_config.yaml", root/"weights/IBM-Craters", CONFIG_SHA256),
              (HEAD_ID, HEAD_REVISION, HEAD_FILE, root/"weights/IBM-Craters", HEAD_SHA256),
              (BACKBONE_ID, BACKBONE_REVISION, "backbone/config.yaml", root/"weights/IBM-Lunar", BACKBONE_CONFIG_SHA256),
              (BACKBONE_ID, BACKBONE_REVISION, "backbone/checkpoint.pt", root/"weights/IBM-Lunar", BACKBONE_SHA256))
    if download and any(not (folder/name).is_file() for _,_,name,folder,_ in assets) and shutil.disk_usage(root).free < 8*1024**3:
        raise RuntimeError("Need at least 8 GiB free local disk for IBM assets/cache")
    for model_id, revision, name, folder, expected in assets:
        path = folder/name
        if not path.is_file():
            if not download:
                raise ValueError(f"Missing IBM asset: {path}. Set DOWNLOAD_WEIGHTS=True explicitly")
            from huggingface_hub import hf_hub_download
            hf_hub_download(model_id, name, revision=revision, local_dir=str(folder))
        if expected and file_hash(path) != expected:
            raise ValueError(f"IBM asset differs from pinned release: {name}; no automatic overwrite")
    return {"assets_available": True, "gpu_verified": False, "options": opts}


def environment(root):
    verify_bundle(root)
    import torch
    from torchvision.ops import nms
    if sys.version_info[:2] != (3, 12) or not torch.cuda.is_available():
        raise RuntimeError("Use separate Python 3.12 + CUDA IBM environment")
    # Exercise both ordinary CUDA and torchvision detection kernels, not just
    # is_available(). This can expose an unsupported driver/T4 wheel early.
    tensor = torch.ones((1, 1, 8, 8), device="cuda")
    torch.nn.functional.conv2d(tensor, tensor[:, :, :3, :3])
    nms(torch.tensor([[0., 0., 4., 4.]], device="cuda"), torch.ones(1, device="cuda"), .3)
    torch.cuda.synchronize()
    del tensor
    gc.collect()
    torch.cuda.empty_cache()
    names = ("torch", "torchvision", "terratorch", "torchmetrics", "lightning", "numpy", "Pillow",
             "timm", "einops", "peft", "transformers", "huggingface-hub", "pycocotools", "PyYAML")
    dependencies = {name: metadata.version(name) for name in names}
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    props = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    result = {"python": sys.version, "dependencies": dependencies, "cuda": torch.version.cuda,
        "gpu": props.name, "gpu_total_bytes": props.total_memory, "driver": driver,
        "capability": list(torch.cuda.get_device_capability(0)), "cuda_arch_list": torch.cuda.get_arch_list(),
        "pip_freeze_sha256": digest(sorted(frozen.splitlines())), "pip_freeze": sorted(frozen.splitlines()),
        "cuda_kernel_check": "passed"}
    atomic_json(Path(root)/"ibm_environment.json", result)
    return result


def select_inputs(root, *, smoke=False):
    """Group four requests per image using image-only provenance, never targets."""
    root = Path(root).resolve()
    requests = read_jsonl(root/"data_orbital/colab_inputs/requests.jsonl")
    provenance = read_jsonl(root/"data_orbital/colab_inputs/provenance.jsonl")
    if len(requests) != len(provenance) or len({p["request_id"] for p in provenance}) != len(provenance):
        raise ValueError("Duplicate or missing request provenance")
    index = {p["request_id"]: p for p in provenance}
    selected = {}
    for request in requests:
        row = index[request["request_id"]]
        sid = safe_name(row["sample_id"])
        task = row.get("task_id")
        if (task not in TASKS or row["condition"] != "clean" or row["track"] != "segmentation"
            or row["source_split"] not in {"val", "test"} or row["evaluation_eligible"] is not True
            or request.get("modality_paths") or len(request["image_paths"]) != 1
            or tuple(request["allowed_answers"]) != ANSWERS[TASKS.index(task)]):
            raise ValueError("IBM pilot expects clean image-only Q1–Q4 val/test inputs")
        path = within(root, request["image_paths"][0])
        if not path.is_relative_to(root/"data_orbital/colab_inputs/images") or file_hash(path) != row["image_sha256"]:
            raise ValueError("Input/provenance image checksum mismatch")
        entry = selected.setdefault(sid, {"sample_id": sid, "image_path": str(path),
            "image_sha256": row["image_sha256"], "planet": row["planet"],
            "source_dataset": row["source_dataset"], "source_split": row["source_split"],
            "provenance": row, "requests": {}})
        if task in entry["requests"] or entry["image_path"] != str(path):
            raise ValueError("Repeated task or inconsistent image for a sample")
        entry["requests"][task] = request
    if not selected or any(set(s["requests"]) != set(TASKS) for s in selected.values()):
        raise ValueError("Each image must have all four tasks")
    if smoke:
        if any(sid not in selected or selected[sid]["planet"] != "moon"
               or selected[sid]["source_split"] != "val" for sid in SMOKE_IDS):
            raise ValueError("Two frozen lunar val smoke probes are missing")
        return [selected[sid] for sid in SMOKE_IDS]
    return [selected[sid] for sid in sorted(selected)]


def render_overlay(image_path, record, destination):
    from PIL import Image, ImageDraw
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for k in (1, 2):
        draw.line((k*w//3, 0, k*w//3, h-1), fill="gray")
        draw.line((0, k*h//3, w-1, k*h//3), fill="gray")
    for box in record["detections"]:
        if box["score"] >= SCORE_THRESHOLD:
            draw.rectangle(box["bbox_xyxy"], outline="lime" if box["eligible"] else "orange", width=2)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")


def copy_checkpoint(source, destination):
    """Checkpoint only this result tree, including PNGs/boxes, after each image."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or "data_rover" in {p.casefold() for p in (*source.parts, *destination.parts)}:
        raise ValueError("Invalid checkpoint destination")
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".png"}:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError("Result symlink is not allowed")
        target = destination/path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and file_hash(target) == file_hash(path):
            continue
        temporary = target.with_suffix(target.suffix+".tmp")
        shutil.copyfile(path, temporary)
        os.replace(temporary, target)


def validate_saved(root, images):
    root = Path(root)
    index = {s["sample_id"]: s for s in images}
    records = []
    paths = sorted((root/"detections").glob("*.json"))
    journal_path = root/"progress.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.is_file() else {"detections_sha256": {}}
    if set(journal["detections_sha256"]) != {p.stem for p in paths}:
        raise ValueError("Incomplete IBM checkpoint transaction; resume refused")
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        sid = record["sample_id"]
        if sid not in index or path.stem != sid or record["image_sha256"] != index[sid]["image_sha256"]:
            raise ValueError("Foreign or changed saved IBM sample")
        if set(record["artifacts_sha256"]) != {f"inputs/{sid}.png", f"overlays/{sid}.png"}:
            raise ValueError("Incomplete IBM artifact hashes")
        for name, expected in record["artifacts_sha256"].items():
            artifact = within(root, name)
            if not artifact.is_file() or file_hash(artifact) != expected:
                raise ValueError("Saved IBM artifact missing or changed")
        if set(record["answers"]) != set(TASKS):
            raise ValueError("Incomplete saved answers")
        # JSON is itself hash-bound by the progress journal, not self-hashed.
        if journal["detections_sha256"].get(sid) != file_hash(path):
            raise ValueError("Saved IBM detections changed; resume refused")
        records.append(record)
    return records


def run_job(root, persist, run_id, session_id, *, smoke=False, resume=False, sample_ids=None):
    root, persist = Path(root).resolve(), Path(persist).resolve()
    if "data_rover" in {p.casefold() for p in persist.parts}:
        raise ValueError("Protected archive cannot be an output")
    safe_name(run_id)
    safe_name(session_id)
    bundle = verify_bundle(root)
    env = environment(root)
    acquire_assets(root)  # Offline; --download is a separate explicit operation.
    identity = {"schema": SCHEMA, "bundle_sha256": bundle["bundle_sha256"], "environment": env,
        "code_revision": CODE_REVISION, "head_revision": HEAD_REVISION,
        "backbone_revision": BACKBONE_REVISION, "protocol": PROTOCOL, "score_threshold": SCORE_THRESHOLD,
        "scientific_score_available": False}
    identity_hash = digest(identity)
    proof_path = root/"ibm_smoke.json"
    if not smoke:
        proof = json.loads(proof_path.read_text()) if proof_path.is_file() else {}
        if (proof.get("success") is not True or proof.get("identity_sha256") != identity_hash
                or proof.get("session_id") != session_id):
            raise ValueError("Run successful IBM smoke in this session/environment first")
    images = select_inputs(root, smoke=smoke)
    if sample_ids is not None:
        if (smoke or not isinstance(sample_ids, (list, tuple)) or not sample_ids
            or any(not isinstance(s, str) for s in sample_ids) or len(set(sample_ids)) != len(sample_ids)):
            raise ValueError("Explicit run sample_ids must be distinct and cannot override smoke")
        if not set(sample_ids) <= {image["sample_id"] for image in images}:
            raise ValueError("Explicit run sample_ids contain unknown images")
        images = [image for image in images if image["sample_id"] in set(sample_ids)]
    name = run_id+("_smoke_"+session_id[:8] if smoke else "")
    local = root/"outputs/IBM-Craters"/name
    durable = persist/"runs/IBM-Craters"/name
    run_record = {**identity, "sample_ids": [s["sample_id"] for s in images], "run_id": run_id, "smoke": smoke}
    if smoke and proof_path.exists():
        proof_path.unlink()  # Invalidate old permission before an attempted new smoke.
    if (local.exists() or durable.exists()) and not resume:
        raise ValueError("IBM run exists. Set RESUME=True for batch, or use a new SESSION_ID for smoke")
    if resume and smoke:
        raise ValueError("Smoke must be fresh; set a new SESSION_ID")
    if durable.exists():
        if local.exists():
            # Durable state is authoritative. Never overwrite divergent local files.
            for file in local.rglob("*.json"):
                other = durable/file.relative_to(local)
                if other.is_file() and file.read_bytes() != other.read_bytes():
                    raise ValueError("Local/Drive IBM state differs; use a fresh local workspace")
        else:
            shutil.copytree(durable, local)
    if local.exists():
        if json.loads((local/"run.json").read_text()) != run_record:
            raise ValueError("IBM run metadata/dependencies changed; resume refused")
    else:
        local.mkdir(parents=True)
        atomic_json(local/"run.json", run_record)
    records = validate_saved(local, images)
    done = {r["sample_id"] for r in records}
    detector = IBMCraterDetector(options(root))
    try:
        if len(done) < len(images):
            print(json.dumps({"event": "ibm_loading", "precision": "float32", "quantization": None}), flush=True)
            loading = detector.initialize()
            atomic_json(local/"loading.json", loading)
        for image in images:
            sid = image["sample_id"]
            if sid in done:
                continue
            started = time.perf_counter()
            prediction = detector.detect(image["image_path"])
            record = {**prediction, "sample_id": sid, "image_sha256": image["image_sha256"],
                "planet": image["planet"], "source_dataset": image["source_dataset"],
                "source_split": image["source_split"], "source_provenance": image["provenance"],
                "elapsed_seconds": time.perf_counter()-started,
                "domain_status": "lunar_png_transfer" if image["planet"] == "moon" else "mars_exploratory_transfer",
                "scored": False}
            input_path, overlay = local/"inputs"/(sid+".png"), local/"overlays"/(sid+".png")
            input_path.parent.mkdir(exist_ok=True)
            shutil.copyfile(image["image_path"], input_path)
            render_overlay(input_path, record, overlay)
            record["artifacts_sha256"] = {p.relative_to(local).as_posix(): file_hash(p) for p in (input_path, overlay)}
            atomic_json(local/"detections"/(sid+".json"), record)
            records.append(record)
            done.add(sid)
            atomic_json(local/"progress.json", {"completed_images": len(records), "total_images": len(images),
                "detections_sha256": {r["sample_id"]: file_hash(local/"detections"/(r["sample_id"]+".json")) for r in records}})
            predictions = []
            image_index = {s["sample_id"]: s for s in images}
            for r in records:
                for task in TASKS:
                    request = image_index[r["sample_id"]]["requests"][task]
                    predictions.append(asdict(ModelResponse(request["request_id"], r["answers"][task], "ok",
                        r["elapsed_seconds"]/4)))
            atomic_json(local/"predictions.jsonl", predictions, lines=True)
            copy_checkpoint(local, durable)
            print(json.dumps({"event": "ibm_image_completed", "sample_id": sid, "images": len(records),
                "total": len(images), "eligible_craters": record["eligible_count"], "answers": record["answers"]}), flush=True)
        atomic_json(local/"summary.json", {"success": len(done) == len(images), "images": len(done),
            "answers": 4*len(done), "scientific_score_available": False,
            "smoke_means": "execution passed; visual/head/input/holdout admission remains separate"})
    except BaseException as exc:
        atomic_json(local/"error.json", {"type": type(exc).__name__, "message": str(exc), "success": False})
        raise
    finally:
        detector.close()
        copy_checkpoint(local, durable)
        with zipfile.ZipFile(persist/(name+"_results.zip"), "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(durable.rglob("*")):
                if path.is_file() and path.suffix in {".json", ".jsonl", ".png"}:
                    archive.write(path, name+"/"+path.relative_to(durable).as_posix())
    if smoke:
        atomic_json(proof_path, {"success": True, "identity_sha256": identity_hash,
            "session_id": session_id, "result_dir": str(local)})
    return {"success": True, "images": len(images), "answers": 4*len(images), "result_dir": str(local),
            "persistent_dir": str(durable), "scientific_score_available": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "assets", "environment", "smoke", "run"))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--persist")
    parser.add_argument("--run-id", default="ibm_craters_v1_clean")
    parser.add_argument("--session-id")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "verify":
        bundle = verify_bundle(args.workspace)
        result = {"images": len(select_inputs(args.workspace)), "bundle_sha256": bundle["bundle_sha256"]}
    elif args.command == "assets":
        result = acquire_assets(args.workspace, download=args.download)
    elif args.command == "environment":
        result = environment(args.workspace)
    else:
        if not args.persist or not args.session_id:
            parser.error("smoke/run requires --persist and --session-id")
        result = run_job(args.workspace, args.persist, args.run_id, args.session_id,
                         smoke=args.command == "smoke", resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
