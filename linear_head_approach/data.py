"""Curated WAC inputs and publisher polygon targets, strictly separated."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .common import ROOT, atomic_json, rows, safe_asset, sha

DATA = "data_orbital/head_training_v2"
INPUT_KEYS = {"sample_id", "split", "scene_group_id", "source_dataset", "source_revision",
              "source_image_id", "rgb_image", "rgb_image_sha256", "native_vis", "native_vis_sha256",
              "display_transform", "license_claim"}


def validate_inputs(inputs):
    """Metadata-only checks usable by the encoder without opening any target."""
    ids, hashes, groups = set(), set(), {}
    for row in inputs:
        if set(row) != INPUT_KEYS:
            raise ValueError("Input contract changed or target information exposed")
        sid, split, group = row["sample_id"], row["split"], row["scene_group_id"]
        if not sid or Path(sid).name != sid or ".." in sid or "/" in sid or "\\" in sid or ":" in sid:
            raise ValueError("Unsafe sample ID")
        if sid in ids or split not in {"train", "val", "test"} or not group:
            raise ValueError("Duplicate ID or invalid split/group")
        if row["source_dataset"] != "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection" or row["source_revision"] != "20f800becb64a0c7ad314652a683f97f0281fe67":
            raise ValueError("Wrong task source/revision")
        if group in groups and groups[group] != split:
            raise ValueError(f"Scene group crosses splits: {group}")
        if row["rgb_image_sha256"] in hashes:
            raise ValueError("Duplicate RGB image")
        ids.add(sid)
        hashes.add(row["rgb_image_sha256"])
        groups[group] = split
    if dict(Counter(r["split"] for r in inputs)) != {"train": 50, "val": 39, "test": 31}:
        raise ValueError("Expected frozen 50/39/31 cohort")
    return groups


def audit(root=ROOT, data=DATA):
    root = Path(root)
    folder = root / data
    inputs, targets = rows(folder / "inputs.jsonl"), rows(folder / "targets.jsonl")
    validate_inputs(inputs)
    target_map = {r["sample_id"]: r for r in targets}
    if not inputs or len(target_map) != len(targets) or len(targets) != len(inputs):
        raise ValueError("Missing or duplicate targets")
    ids, hashes, native_hashes, group_splits = set(), set(), set(), {}
    coverage = defaultdict(list)
    for row in inputs:
        if set(row) != INPUT_KEYS:
            raise ValueError("Input contract changed or target information exposed")
        sid, split, group = row["sample_id"], row["split"], row["scene_group_id"]
        if not sid or not group or split not in {"train", "val", "test"} or sid in ids or sid not in target_map:
            raise ValueError("Invalid ID/split or duplicate input")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Scene group crosses splits: {group}")
        if row["rgb_image_sha256"] in hashes or row["native_vis_sha256"] in native_hashes:
            raise ValueError("Duplicate image/native raster across samples")
        group_splits[group] = split
        ids.add(sid)
        hashes.add(row["rgb_image_sha256"])
        native_hashes.add(row["native_vis_sha256"])
        target = target_map[sid]
        if target["split"] != split or target["scene_group_id"] != group or target["target"] != "catalog_crater_polygon":
            raise ValueError("Target/input alignment or task mismatch")
        image = safe_asset(root, row["rgb_image"], row["rgb_image_sha256"])
        safe_asset(root, row["native_vis"], row["native_vis_sha256"])
        mask_path = safe_asset(root, target["mask"], target["mask_sha256"])
        with Image.open(image) as source, Image.open(mask_path) as mask:
            if source.size != mask.size or source.size != (512, 512):
                raise ValueError("Square image/mask alignment required (no hidden center crop)")
            values = np.asarray(mask)
        if values.ndim != 2 or not set(np.unique(values)).issubset({0, 1}):
            raise ValueError("Expected publisher binary 0/1 crater mask, not semantic classes")
        coverage[split].append(float(values.mean()))
    counts = dict(Counter(r["split"] for r in inputs))
    if not all(counts.get(s) for s in ("train", "val", "test")):
        raise ValueError("All three source splits required")
    result = {"split_counts": counts, "scene_groups": dict(Counter(group_splits.values())),
              "mean_mask_fraction": {s: float(np.mean(v)) for s, v in coverage.items()},
              "inputs_sha256": sha(folder / "inputs.jsonl"), "targets_sha256": sha(folder / "targets.jsonl"),
              "warning": "Source-heldout only; pretraining independence NOT demonstrated. Catalog polygons are not exhaustive visual crater truth.",
              "scientific_result": False}
    atomic_json(folder / "audit.json", result)
    return result


def metrics(prediction, target):
    p, y = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if p.shape != y.shape:
        raise ValueError("Metric shapes differ")
    tp = int((p & y).sum())
    fp = int((p & ~y).sum())
    fn = int((~p & y).sum())
    return {"iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "tp": tp, "fp": fp, "fn": fn, "empty_target": not bool(y.any())}


def scene_macro(records, key="iou"):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["scene_group_id"]].append(row[key])
    if not grouped:
        raise ValueError("No records to aggregate")
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def bootstrap(records, key="iou", seed=2026, repeats=2000):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["scene_group_id"]].append(row[key])
    means = np.array([np.mean(v) for _, v in sorted(grouped.items())])
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(0, len(means), size=(repeats, len(means)))].mean(1)
    return {"mean": float(means.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(), "groups": len(means)}
