"""Separate inputs/targets and streaming, train-only normalization; no tensor framework."""
from collections import Counter
from pathlib import Path
import re

import numpy as np
from PIL import Image
from planetary_vlm.datasets.mars_binary import INPUT_KEYS, REPO, REVISION

from .io import atomic, fingerprint, guard, orbital, read, rows, sha, within


class Index:
    def __init__(self, root, cfg):
        self.root, self.cfg = Path(root).resolve(), cfg
        self.folder = self.root / cfg["dataset"]
        relative = str(Path(cfg["dataset"]) / "metadata.json")
        metadata = read(orbital(self.root, relative))
        if metadata.get("schema") != "mars-crater-binary-v1" or metadata.get("status") != "complete":
            raise ValueError("A completed Mars binary collection is required")
        if (metadata.get("source_dataset"), metadata.get("source_revision")) != (REPO, REVISION):
            raise ValueError("Wrong pinned Mars source/revision")
        inputs = orbital(self.root, str(Path(cfg["dataset"]) / "inputs.jsonl"))
        targets = orbital(self.root, str(Path(cfg["dataset"]) / "targets.jsonl"))
        for path in (inputs, targets):
            if sha(path) != metadata["files_sha256"][path.name]:
                raise ValueError("Dataset manifest differs from completed collection")
        self.inputs, target_rows = rows(inputs), rows(targets)
        self.targets = {r["sample_id"]: r for r in target_rows}
        ids, groups = set(), {}
        for row in self.inputs:
            sid, split = row["sample_id"], row["split"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", sid) or sid in ids or sid not in self.targets:
                raise ValueError("Unsafe, duplicate or unmatched sample ID")
            ids.add(sid)
            if split not in cfg["counts"] or row["scene_group_id"] in groups and groups[row["scene_group_id"]] != split:
                raise ValueError("Invalid split or scene group crossing splits")
            groups[row["scene_group_id"]] = split
            if not set(row) <= set(INPUT_KEYS):
                raise ValueError("Ground truth or unknown fields must not be in model input rows")
            if (row["width"], row["height"], row["channels"], row["planet"]) != (512, 512, 1, "mars"):
                raise ValueError("Expected native single-intensity Mars 512x512")
        if len(ids) != len(target_rows) or len(self.targets) != len(target_rows):
            raise ValueError("Missing/duplicate/orphan target rows")
        if dict(Counter(r["split"] for r in self.inputs)) != cfg["counts"]:
            raise ValueError("Source split counts changed")
        self.identity = {"inputs_sha256": sha(inputs), "targets_sha256": sha(targets),
                         "metadata_sha256": sha(self.folder / "metadata.json")}

    def split(self, name):
        if name not in self.cfg["counts"]:
            raise ValueError("Unknown split")
        return [r for r in self.inputs if r["split"] == name]

    def image(self, row, resized=True):
        with Image.open(orbital(self.root, row["image"], row["image_sha256"])) as image:
            if image.mode != "L" or image.size != (512, 512):
                raise ValueError("Native image format differs")
            if resized:
                side = self.cfg["encoder"]["input_size"]
                image = image.resize((side, side), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.float32).copy() / 255.

    def mask(self, sid):
        row = self.targets[sid]
        with Image.open(orbital(self.root, row["mask"], row["mask_sha256"])) as image:
            mask = np.asarray(image).copy()
        if mask.shape != (512, 512) or not set(np.unique(mask)) <= {0, 1}:
            raise ValueError("Invalid native binary mask")
        return mask

    def controls(self):
        chosen = {}
        for row in self.split("train"):
            positive = bool(self.mask(row["sample_id"]).any())
            chosen.setdefault(positive, row)
            if len(chosen) == 2:
                return [chosen[False], chosen[True]]
        raise ValueError("Smoke requires a train negative and a train positive")


def prepare(index, output):
    path = within(output, "preparation/stats.json")
    identity = {"data": index.identity, "input_size": index.cfg["encoder"]["input_size"],
                "normalization": "train_resized_DN_div255_global_mean_std_v1",
                "positive_weight_cap": index.cfg["training"]["positive_weight_cap"]}
    guard(within(output, "preparation/identity.json"), identity)
    if path.exists():
        result = read(path)
        if result["identity"] != identity:
            raise ValueError("Train statistics identity differs")
        return result
    total = squares = count = foreground = background = 0
    training = index.split("train")
    for number, row in enumerate(training):
        image = index.image(row).astype(np.float64)
        total += float(image.sum())
        squares += float(np.square(image).sum())
        count += image.size
        mask = index.mask(row["sample_id"])
        foreground += int(mask.sum())
        background += mask.size - int(mask.sum())
        if (number + 1) % 100 == 0:
            print(f"prepare_train_completed={number + 1}/{len(training)}", flush=True)
    mean = total / count
    std = max(squares / count - mean * mean, 0.) ** .5
    if std <= 1e-8 or not foreground or not background:
        raise ValueError("Degenerate train normalization or class coverage")
    result = {"identity": identity, "mean": mean, "std": std,
              "positive_weight": min(identity["positive_weight_cap"], background / foreground),
              "train_foreground_pixels": foreground, "train_background_pixels": background,
              "controls": [row["sample_id"] for row in index.controls()], "test_used": False}
    atomic(path, result)
    return result


def cache_identity(index, stats, assets, code, environment):
    return {"inputs_sha256": index.identity["inputs_sha256"], "encoder": index.cfg["encoder"],
            "normalization": {k: stats[k] for k in ("mean", "std")}, "assets": assets,
            "code": code, "environment": environment, "cache_precision": "float16"}


def cache_directory(output, identity):
    return within(output, "cache/" + fingerprint(identity)[:20])
