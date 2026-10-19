"""Binary catalog-coverage metrics; seeds and scene uncertainty are distinct."""
from collections import defaultdict
import numpy as np


def binary_counts(prediction, target):
    p, y = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if p.shape != y.shape:
        raise ValueError("Prediction/target shapes differ")
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
            "empty_target": not bool(y.any())}


def summarize(records):
    if not records:
        raise ValueError("No predictions to summarize")
    groups = defaultdict(list)
    for row in records:
        groups[row["scene_group_id"]].append(row)
    tp, fp, fn = (sum(r[k] for r in records) for k in ("tp", "fp", "fn"))
    return {"images": len(records), "scene_groups": len(groups),
            "scene_macro_iou": float(np.mean([np.mean([r["iou"] for r in group]) for group in groups.values()])),
            "image_macro_iou": float(np.mean([r["iou"] for r in records])),
            "image_macro_dice": float(np.mean([r["dice"] for r in records])),
            "foreground_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "empty_targets": sum(r["empty_target"] for r in records),
            "empty_target_false_positive_images": sum(r["empty_target"] and r["fp"] > 0 for r in records),
            "tp": tp, "fp": fp, "fn": fn}


def select_threshold(records):
    if not records:
        raise ValueError("No validation thresholds")
    # Smaller threshold wins exact ties. Never inspect test metrics here.
    return max(sorted(records), key=lambda t: summarize(records[t])["scene_macro_iou"])
