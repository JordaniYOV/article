"""Fit/score one frozen-feature WAC crater probe from a checked feature manifest.

Feature rows: sample_id, rgb_image_sha256, split, scene_group_id, feature,
feature_sha256, extractor_id, model_revision, preprocessing_id. `feature` is
an HxWxC .npy spatial feature map under linear_head_approach/outputs/; no
masks belong in it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from linear_head_approach.linear_probe import (
    SpatialRidgeProbe, choose_threshold, fit, grouped_mean_interval, image_scores,
)

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def asset(relative: str, expected_sha256: str, allowed_root: Path) -> Path:
    path = (ROOT / relative).resolve()
    if allowed_root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Missing or unsafe asset: {relative}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    return path


def joined(probe_dir: Path, features_path: Path) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict]:
    inputs = {r["sample_id"]: r for r in read_jsonl(probe_dir / "encoder_inputs.jsonl")}
    targets = {r["sample_id"]: r for r in read_jsonl(probe_dir / "targets.jsonl")}
    features_list = read_jsonl(features_path)
    features = {r["sample_id"]: r for r in features_list}
    if len(inputs) != 120 or len(targets) != 120 or len(features_list) != 120 or len(features) != 120:
        raise ValueError("Expected 120 unique records in each manifest")
    if set(inputs) != set(targets) or set(inputs) != set(features):
        raise ValueError("Sample IDs differ across manifests")
    provenance = set()
    for sample_id, row in features.items():
        original = inputs[sample_id]
        target = targets[sample_id]
        if target["split"] != original["split"] or target["scene_group_id"] != original["scene_group_id"]:
            raise ValueError(f"Target provenance mismatch: {sample_id}")
        if any(key in row for key in ("mask", "target", "answer", "height_map")):
            raise ValueError("Targets or non-image modalities leaked into feature manifest")
        if (row["split"] != original["split"] or row["scene_group_id"] != original["scene_group_id"]
                or row["rgb_image_sha256"] != original["rgb_image_sha256"]):
            raise ValueError(f"Feature provenance mismatch: {sample_id}")
        provenance.add((row["extractor_id"], row["model_revision"], row["preprocessing_id"]))
    if len(provenance) != 1:
        raise ValueError("Mixed extractor or model revisions")
    extractor_id, model_revision, preprocessing_id = provenance.pop()
    return inputs, targets, features, {
        "extractor_id": extractor_id, "model_revision": model_revision,
        "preprocessing_id": preprocessing_id,
        "feature_manifest_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
    }


def samples(split: str, inputs: dict, targets: dict, features: dict):
    maps, masks, ids = [], [], []
    for sample_id in sorted(inputs):
        if inputs[sample_id]["split"] != split:
            continue
        row = features[sample_id]
        feature_path = asset(row["feature"], row["feature_sha256"], ROOT / "linear_head_approach" / "outputs")
        feature_map = np.load(feature_path, allow_pickle=False)
        mask_row = targets[sample_id]
        mask_path = asset(mask_row["mask"], mask_row["mask_sha256"], ROOT / "data_orbital")
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
        maps.append(feature_map)
        masks.append(mask)
        ids.append(sample_id)
    return maps, masks, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "score-test"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, default=ROOT / "linear_head_approach" / "outputs" / "encoder_probe_wac_v1")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs, targets, features, provenance = joined(args.probe_dir, args.features)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.result_dir / "linear_probe.npz"
    metadata_path = args.result_dir / "linear_probe.json"
    if args.action == "fit":
        train_x, train_y, _ = samples("train", inputs, targets, features)
        val_x, val_y, _ = samples("val", inputs, targets, features)
        probe = fit(train_x, train_y)
        threshold, val_iou = choose_threshold(probe, val_x, val_y)
        np.savez(model_path, coefficients=probe.coefficients, intercept=probe.intercept,
                 mean=probe.mean, scale=probe.scale, grid=probe.grid,
                 regularization=probe.regularization)
        metadata = {**provenance, "threshold": threshold, "validation_mean_iou": val_iou,
                    "train_images": len(train_x), "validation_images": len(val_x),
                    "trainable_head_parameters": int(len(probe.coefficients) + 1),
                    "feature_width": int(len(probe.coefficients)),
                    "test_opened": False,
                    "interpretation_status": "pending_feature_and_provenance_review"}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2))
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata[key] != provenance[key] for key in provenance):
            raise ValueError("Feature provenance changed after fitting")
        with np.load(model_path, allow_pickle=False) as saved:
            probe = SpatialRidgeProbe(saved["coefficients"], float(saved["intercept"]),
                                      saved["mean"], saved["scale"], tuple(saved["grid"]),
                                      float(saved["regularization"]))
        test_x, test_y, ids = samples("test", inputs, targets, features)
        scores = [image_scores(probe.predict_grid(x), y, metadata["threshold"])
                  for x, y in zip(test_x, test_y)]
        ious = [score["iou"] for score in scores]
        groups = [inputs[sample_id]["scene_group_id"] for sample_id in ids]
        report = {**provenance, "test_images": len(scores), "test_mean_iou": float(np.mean(ious)),
                  "test_mean_dice": float(np.mean([score["dice"] for score in scores])),
                  "all_foreground_mean_iou": float(np.mean([score["all_foreground_iou"] for score in scores])),
                  "test_mean_iou_group_bootstrap_95pct": grouped_mean_interval(ious, groups),
                  "per_image_scores": dict(zip(ids, scores)), "threshold_frozen_from_validation": metadata["threshold"],
                  "article_level_claim": False}
        (args.result_dir / "test_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
