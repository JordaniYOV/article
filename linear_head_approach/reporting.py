"""Paired scene-level comparisons and bounded visual checks."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import atomic_json, read_json, ROOT
from .data import bootstrap


def compare(directory, split="test"):
    directory = Path(directory)
    files = sorted(directory.glob(f"*/*/seed_*/evaluation_{split}.json"))
    evaluations = [read_json(p) for p in files]
    if not evaluations:
        raise ValueError("No evaluation files found")
    protocols = {(r["protocol_sha256"], r["targets_sha256"]) for r in evaluations}
    if len(protocols) != 1:
        raise ValueError("Cannot compare different protocols/targets")
    by_cell = defaultdict(list)
    for result in evaluations:
        by_cell[result["encoder"], result["family"]].append(result)
    summaries, collapsed = {}, {}
    for (encoder, family), results in by_cell.items():
        key = f"{encoder}/{family}"
        if len({r["seed"] for r in results}) != len(results):
            raise ValueError("Duplicate seed in comparison")
        sample_sets = [{(r["sample_id"], r["scene_group_id"]) for r in result["rows"]} for result in results]
        if any(s != sample_sets[0] for s in sample_sets[1:]):
            raise ValueError("Seed evaluations use different images/groups")
        values = defaultdict(list)
        for result in results:
            for row in result["rows"]:
                values[row["sample_id"], row["scene_group_id"]].append(row["iou"])
        combined = [{"sample_id": sid, "scene_group_id": group, "iou": float(np.mean(v))} for (sid, group), v in sorted(values.items())]
        collapsed[encoder, family] = combined
        summaries[key] = {"seeds": sorted(r["seed"] for r in results), "seed_scene_iou": [r["scene_iou"]["mean"] for r in results],
                          "mean_over_seeds_then_scene_bootstrap": bootstrap(combined),
                          "parameters": sorted({r["head_parameters"] for r in results})}
    differences = {}
    for family in sorted({k[1] for k in by_cell}):
        if ("ibm", family) not in collapsed or ("spacellava", family) not in collapsed:
            continue
        left, right = by_cell["ibm", family], by_cell["spacellava", family]
        if {r["seed"] for r in left} != {r["seed"] for r in right}:
            raise ValueError("Paired encoder comparison requires matching seeds")
        a = {(r["sample_id"], r["scene_group_id"]): r["iou"] for r in collapsed["ibm", family]}
        b = {(r["sample_id"], r["scene_group_id"]): r["iou"] for r in collapsed["spacellava", family]}
        if a.keys() != b.keys():
            raise ValueError("Paired encoder comparison requires identical scenes")
        differences[family] = bootstrap([{"scene_group_id": key[1], "iou": a[key] - b[key]} for key in a])
    result = {"split": split, "cells": summaries, "paired_ibm_minus_spacellava_iou": differences,
              "caution": "Native 5-band vs rendered RGB; unequal channels/parameter counts. CI resamples scene groups after averaging seeds, not pixels or QA rows. Not a pure specialization effect."}
    atomic_json(directory / f"comparison_{split}.json", result)
    return result


def preview(cache, output, root=ROOT, device="cuda"):
    """Validation examples chosen deterministically, never cherry-picked test frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from .common import rows, safe_asset
    from .training import checked_features, CachedDataset
    from .heads import CraterHead
    output, root = Path(output), Path(root)
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    identity = saved["identity"]
    from .common import sha
    if identity != read_json(output / "identity.json") or identity["feature_manifest_sha256"] != sha(Path(cache) / "features.jsonl"):
        raise ValueError("Preview cache/checkpoint identity differs")
    targets = {r["sample_id"]: r for r in rows(root / "data_orbital/head_training_v2/targets.jsonl") if r["split"] == "val"}
    inputs = {r["sample_id"]: r for r in rows(root / "data_orbital/head_training_v2/inputs.jsonl")}
    records = [r for r in checked_features(cache, root) if r["split"] == "val"][:3]
    dataset = CachedDataset(cache, records, targets, saved["mean"], saved["std"], root)
    head = CraterHead(saved["channels"], identity["family"], identity["config"]["hidden_channels"]).to(device).eval()
    head.load_state_dict(saved["head"])
    fig, axes = plt.subplots(len(records), 3, figsize=(10, 3.5 * len(records)), squeeze=False)
    for i, (x, target, _) in enumerate(dataset):
        sid = records[i]["sample_id"]
        with torch.inference_mode():
            probability = head(x[None].to(device), 512).sigmoid()[0, 0].cpu().numpy()
        with Image.open(safe_asset(root, inputs[sid]["rgb_image"])) as image:
            axes[i, 0].imshow(image.convert("RGB"))
        axes[i, 1].imshow(target[0], vmin=0, vmax=1, cmap="gray")
        axes[i, 2].imshow(probability >= saved["threshold"], vmin=0, vmax=1, cmap="gray")
        for ax, title in zip(axes[i], (sid, "Publisher catalog target", "Predicted mask (val threshold)")):
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    path = output / "validation_preview.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    history = read_json(output / "history.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key, title in zip(axes, ("train_loss", "validation_scene_iou"), ("Train BCE + Dice", "Validation scene IoU")):
        ax.plot([r["epoch"] for r in history], [r[key] for r in history])
        ax.set(xlabel="Epoch", title=title)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=120)
    plt.close(fig)
    return {"preview": str(path), "curves": str(output / "learning_curves.png")}
