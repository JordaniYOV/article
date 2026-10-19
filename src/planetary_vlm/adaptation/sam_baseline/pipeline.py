"""SAM automatic proposals -> frozen geometry filter -> shared binary evaluator."""
from pathlib import Path
import re

import numpy as np
from PIL import Image

from ..common.data import Index
from ..common.io import atomic, code_identity, environment, guard, logger, orbital, read, sha, within
from ..common.metrics import binary_counts, summarize
from .filter import union_candidates


def assets(root, cfg):
    s = cfg["sam"]
    if not re.fullmatch(r"[0-9a-f]{64}", s["sha256"]):
        raise ValueError("Set sam.sha256 to the full verified hash of a trusted official SAM ViT-B checkpoint")
    path = within(root, s["checkpoint"])
    if not path.is_relative_to(Path(root).resolve() / "weights") or not path.is_file() or sha(path) != s["sha256"]:
        raise ValueError("SAM local checkpoint path/hash differs")
    import segment_anything
    package = Path(segment_anything.__file__).resolve().parent
    code = {p.relative_to(package).as_posix(): sha(p) for p in sorted(package.rglob("*.py"))}
    return path, {"checkpoint_sha256": s["sha256"], "runtime_files": code,
                  "origin": "Operator-supplied trusted official SAM ViT-B checkpoint; no implicit download"}


def load(root, cfg, device):
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no automatic fallback")
    path, verified = assets(root, cfg)
    model = sam_model_registry["vit_b"](checkpoint=str(path)).to(device).eval().requires_grad_(False)
    s = cfg["sam"]
    generator = SamAutomaticMaskGenerator(model, points_per_side=s["points_per_side"],
        points_per_batch=s["points_per_batch"], pred_iou_thresh=s["pred_iou_thresh"],
        stability_score_thresh=s["stability_score_thresh"], crop_n_layers=0, min_mask_region_area=0)
    return generator, verified


def predict(generator, image_path, cfg):
    with Image.open(image_path) as image:
        if image.mode != "L" or image.size != (512, 512):
            raise ValueError("SAM comparison requires original 512x512 intensity image")
        # Format adaptation only: repeating grayscale does not create colour information.
        pixels = np.asarray(image.convert("RGB")).copy()
    candidates = generator.generate(pixels)
    mask, accepted = union_candidates(candidates, pixels.shape[:2], cfg["sam"])
    return mask, {"proposals": len(candidates), "accepted_proposals": accepted}


def identity(index, cfg, verified, device):
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    return {"config": cfg, "dataset": index.identity, "assets": verified, "code": code_identity(),
            "environment": environment(), "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}


def smoke(root, cfg, output, device="cuda"):
    import torch
    index = Index(root, cfg)
    generator, verified = load(root, cfg, device)
    controls = index.controls()
    with logger(within(output, "sam_baseline")) as log, torch.no_grad():
        results = []
        for row in controls:
            mask, info = predict(generator, orbital(root, row["image"], row["image_sha256"]), cfg)
            results.append({"sample_id": row["sample_id"], **info, "predicted_pixels": int(mask.sum())})
            log.info("sam_smoke sample=%s proposals=%s accepted=%s", row["sample_id"], info["proposals"], info["accepted_proposals"])
        marker = {"identity": identity(index, cfg, verified, device), "controls": results, "quality_claim": False}
        atomic(within(output, "sam_baseline/smoke.json"), marker)
    return {"controls": results, "quality_claim": False, "training": False}


def evaluate(root, cfg, output, split="val", allow_test=False, device="cuda"):
    import torch
    if split == "test" and not allow_test:
        raise ValueError("Test scoring requires explicit --allow-test")
    index = Index(root, cfg)
    generator, verified = load(root, cfg, device)
    current = identity(index, cfg, verified, device)
    marker = within(output, "sam_baseline/smoke.json")
    if not marker.is_file() or read(marker)["identity"] != current:
        raise ValueError("SAM smoke required for this exact configuration/environment")
    directory = within(output, f"sam_baseline/evaluation/{split}")
    guard(directory / "identity.json", current)
    records = []
    with logger(directory) as log, torch.no_grad():
        for number, row in enumerate(index.split(split)):
            target = index.mask(row["sample_id"])
            path = within(directory, "masks/" + row["sample_id"] + ".png")
            record_path = within(directory, "records/" + row["sample_id"] + ".json")
            if record_path.is_file():
                record = read(record_path)
                if record["prediction_sha256"] != sha(path):
                    raise ValueError("Saved SAM prediction changed")
                with Image.open(path) as image:
                    mask = np.asarray(image)
                if mask.shape != target.shape or not set(np.unique(mask)) <= {0, 1}:
                    raise ValueError("Invalid saved SAM mask")
                if any(record.get(k) != v for k, v in binary_counts(mask, target).items()):
                    raise ValueError("Saved SAM evaluation differs from reference")
            else:
                mask, info = predict(generator, orbital(root, row["image"], row["image_sha256"]), cfg)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask.astype(np.uint8)).save(path)
                record = {"sample_id": row["sample_id"], "scene_group_id": row["scene_group_id"],
                          **binary_counts(mask, target), **info,
                          "prediction_mask": str(path), "prediction_sha256": sha(path)}
                atomic(record_path, record)
            records.append(record)
            if (number + 1) % 25 == 0:
                atomic(directory / "progress.json", {"completed": number + 1, "total": len(index.split(split))})
                log.info("sam_completed=%s/%s", number + 1, len(index.split(split)))
    report = {"schema": "mars-segmentation-report-v1", "method": "sam_baseline", "seed": None, "split": split,
              "dataset": index.identity, "protocol": cfg, "identity": current, "records": records,
              "metrics": summarize(records), "training": False,
              "interpretation": "Class-agnostic masks plus fixed geometry filter, not a pretrained crater classifier"}
    atomic(directory / "report.json", report)
    return {"report": str(directory / "report.json"), "metrics": report["metrics"]}


def predict_image(root, cfg, output, image_path, destination, device="cuda"):
    import torch
    generator, verified = load(root, cfg, device)
    with torch.no_grad():
        mask, info = predict(generator, orbital(root, image_path), cfg)
    destination = within(output, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Prediction output already exists")
    Image.fromarray(mask.astype(np.uint8)).save(destination)
    return {"mask": str(destination), "mask_sha256": sha(destination), "assets": verified, **info}


def main(argv=None):
    from ..common.cli import main as run
    return run("sam_baseline", argv)
