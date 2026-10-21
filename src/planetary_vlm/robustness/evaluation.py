"""Inference-only adapters and resumable, ignore-aware catalog scoring."""
from pathlib import Path

import numpy as np
from PIL import Image

from planetary_vlm.adaptation.common.io import atomic, environment, fingerprint, guard, logger, orbital, read, sha, within
from planetary_vlm.adaptation.common.metrics import binary_counts, summarize
from .bundle import read_target, save_png


class NeuralPredictor:
    def __init__(self, root, clean, job, device):
        import torch
        from planetary_vlm.adaptation.common.engine import evaluation_system
        self.root, self.cfg, self.device = root, job["config"], device
        # This existing loader verifies exact training environment, backbone,
        # math policy, code and checkpoint identity. No override is provided.
        self.index, self.stats, identity, _, saved, self.system = evaluation_system(
            root, self.cfg, clean, job["seed"], device)
        if fingerprint(identity) != job["source_identity_sha256"]:
            raise ValueError("Training identity changed after planning")
        self.threshold = float(saved["threshold"])
        if self.threshold not in self.cfg["training"]["thresholds"]:
            raise ValueError("Checkpoint threshold outside original validation grid")
        self.system.eval().requires_grad_(False)
        self.torch = torch

    def __call__(self, row):
        from planetary_vlm.adaptation.common.engine import image_tensor, autocast
        # row is image-only. Original train stats and input resize are unchanged.
        x = image_tensor(self.index, row, self.stats)[None].to(self.device)
        with self.torch.no_grad(), autocast(self.device):
            probability = self.system(x, 512).float().sigmoid()[0, 0].cpu().numpy()
        if probability.shape != (512, 512) or not np.isfinite(probability).all():
            raise ValueError("Nonfinite/wrong-grid robustness prediction")
        return (probability >= self.threshold).astype(np.uint8), {}


class SAMPredictor:
    def __init__(self, root, clean, job, device):
        from planetary_vlm.adaptation.common.data import Index
        from planetary_vlm.adaptation.sam_baseline import pipeline
        self.root, self.cfg, self.pipeline = root, job["config"], pipeline
        self.generator, verified = pipeline.load(root, self.cfg, device)
        current = pipeline.identity(Index(root, self.cfg), self.cfg, verified, device)
        if fingerprint(current) != job["source_identity_sha256"]:
            raise ValueError("Original SAM environment/checkpoint/filter identity changed")
        self.threshold = None

    def __call__(self, row):
        import torch
        with torch.no_grad():
            return self.pipeline.predict(self.generator, orbital(self.root, row["image"], row["image_sha256"]), self.cfg)


def checked_mask(mask):
    mask = np.asarray(mask)
    if mask.shape != (512, 512) or not set(np.unique(mask)) <= {0, 1}:
        raise ValueError("Prediction must be a binary mask on the native 512x512 grid")
    return mask.astype(np.uint8)


def counts(mask, target):
    valid = target != 255
    if not valid.any():
        raise ValueError("Target has no valid evaluation pixels")
    return {**binary_counts(mask[valid], target[valid]), "valid_pixels": int(valid.sum()),
            "ignored_pixels": int((~valid).sum())}


def smoke(predictor, inputs, directory):
    """Two TRAIN controls per condition, no gradient update or label-based gate."""
    controls = [r for r in inputs if r["role"] == "control"]
    signatures = []
    for row in controls:
        first, _ = predictor(row)
        second, _ = predictor(row)
        first, second = checked_mask(first), checked_mask(second)
        changed = float(np.mean(first != second))
        if changed > .001:
            raise ValueError("Inference smoke is not repeatable within declared 0.1% mask tolerance")
        signatures.append({"sample_id": row["sample_id"], "condition": row["condition"],
                           "predicted_pixels": int(first.sum()), "repeat_changed_fraction": changed})
    marker = {"controls": signatures, "training": False, "quality_claim": False,
              "gate": "finite_native_binary_mask_and_repeatability_not_accuracy", "max_repeat_changed_fraction": .001}
    atomic(directory / "smoke.json", marker)
    return marker


def score_condition(root, predictor, inputs, targets, condition, directory, identity, log):
    selected = [r for r in inputs if r["role"] == "evaluation" and r["condition"] == condition]
    guard(directory / "identity.json", identity)
    records = []
    for number, row in enumerate(selected):
        sid = row["sample_id"]
        target_row = targets[(condition, sid)]
        # Only the input row is passed to predictor. No mask/outcome telemetry.
        path = within(directory, f"masks/{sid}.png")
        record_path = within(directory, f"records/{sid}.json")
        descriptor = {"sample_id": sid, "scene_group_id": row["scene_group_id"],
                      "image_sha256": row["image_sha256"], "target_sha256": target_row["mask_sha256"]}
        if record_path.is_file():
            record = read(record_path)
            if any(record.get(k) != v for k, v in descriptor.items()) or record["prediction_sha256"] != sha(path):
                raise ValueError("Saved inference record/mask changed")
            with Image.open(path) as image:
                mask = checked_mask(np.asarray(image))
        else:
            mask, info = predictor(row)
            mask = checked_mask(mask)
            save_png(path, mask)
            record = {**descriptor, "prediction_sha256": sha(path), "proposal_info": info}
        target = read_target(root, target_row)
        expected = counts(mask, target)
        if record_path.is_file() and any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("Saved metrics differ from prediction/reference")
        if not record_path.is_file():
            record.update(expected)
            atomic(record_path, record)
        records.append(record)
        if (number + 1) % 25 == 0 or number + 1 == len(selected):
            atomic(directory / "progress.json", {"completed": number + 1, "total": len(selected)})
            log.info("condition=%s completed=%s/%s", condition, number + 1, len(selected))
    report = {"schema": "mars-robustness-report-v1", "identity": identity, "records": records,
              "metrics": summarize(records),
              "scene_metrics": {group: summarize([r for r in records if r["scene_group_id"] == group])
                                for group in sorted({r["scene_group_id"] for r in records})},
              "valid_pixels": sum(r["valid_pixels"] for r in records),
              "ignored_pixels": sum(r["ignored_pixels"] for r in records),
              "training": False, "test_used_for_selection": False}
    atomic(directory / "report.json", report)
    return report


def evaluate_job(root, clean, output, job, bundle_identity, bundle_sha256, inputs, targets, device):
    from .protocol import job_id
    directory = within(output, job_id(job))
    identity = {"job": job, "bundle_complete_sha256": bundle_sha256,
                "split": bundle_identity["split"], "device": device, "environment": environment(),
                "implementation": bundle_identity["implementation"]}
    guard(directory / "identity.json", identity)
    with logger(directory) as log:
        log.info("model_loading method=%s seed=%s", job["method"], job["seed"])
        predictor = (SAMPredictor if job["method"] == "sam_baseline" else NeuralPredictor)(root, clean, job, device)
        log.info("inference_smoke_started TRAIN_controls=2 per_condition=true")
        smoke(predictor, inputs, directory)
        for condition in bundle_identity["protocol"]["conditions"]:
            cid = condition["id"]
            report = score_condition(root, predictor, inputs, targets, cid, within(directory, cid),
                                     {**identity, "condition": cid, "threshold": predictor.threshold}, log)
            log.info("condition_completed=%s scene_iou=%.6f", cid, report["metrics"]["scene_macro_iou"])
        atomic(directory / "complete.json", {"identity_sha256": fingerprint(identity),
               "reports": {c["id"]: sha(directory / c["id"] / "report.json") for c in bundle_identity["protocol"]["conditions"]}})
    return {"method": job["method"], "seed": job["seed"], "completed": True}
