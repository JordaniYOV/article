"""Summarize predeclared seeds; never select a favorable test run."""
import numpy as np
from .io import atomic, read, within


def summarize_seeds(cfg, output, split):
    reports = [read(within(output, f"{cfg['method']}/seed_{seed}/evaluation/{split}/report.json")) for seed in cfg["seeds"]]
    first = reports[0]
    expected_ids = [(r["sample_id"], r["scene_group_id"]) for r in first["records"]]
    for seed, report in zip(cfg["seeds"], reports):
        if (report["seed"] != seed or report["split"] != split or report["method"] != cfg["method"]
                or report["dataset"] != first["dataset"] or report["protocol"] != cfg
                or [(r["sample_id"], r["scene_group_id"]) for r in report["records"]] != expected_ids):
            raise ValueError("Seed reports do not share a frozen protocol and evaluation cohort")
    metrics = {}
    for key in ("scene_macro_iou", "image_macro_iou", "image_macro_dice", "foreground_iou", "precision", "recall"):
        values = [r["metrics"][key] for r in reports]
        metrics[key] = {"values": values, "mean": float(np.mean(values)) if None not in values else None,
                        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 and None not in values else None}
    result = {"method": cfg["method"], "split": split, "seeds": cfg["seeds"], "metrics": metrics,
              "dataset": first["dataset"], "images": len(expected_ids),
              "scene_groups": len({group for _, group in expected_ids}),
              "note": "Mean/std across training seeds, not an ensemble prediction or confidence interval; do not select a best seed."}
    path = within(output, f"{cfg['method']}/summary_{split}.json")
    atomic(path, result)
    return {"summary": str(path), **result}
