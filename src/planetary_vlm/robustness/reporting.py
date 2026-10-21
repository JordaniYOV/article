"""Mean/std of all declared seeds and per-seed clean-relative changes."""
import statistics

from planetary_vlm.adaptation.common.io import atomic, fingerprint, read, sha
from .protocol import job_id

METRICS = ("scene_macro_iou", "image_macro_iou", "image_macro_dice", "foreground_iou", "precision", "recall")


def distribution(values):
    if any(value is None for value in values):
        return {"values": values, "mean": None, "sample_std": None}
    return {"values": values, "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def summarize(output, jobs, bundle_identity, bundle_sha256):
    reports, result = {}, []
    conditions = [c["id"] for c in bundle_identity["protocol"]["conditions"]]
    for job in jobs:
        folder = output / job_id(job)
        marker = read(folder / "complete.json")
        identity = read(folder / "identity.json")
        if (marker["identity_sha256"] != fingerprint(identity) or identity["job"] != job
                or identity["bundle_complete_sha256"] != bundle_sha256):
            raise ValueError("Completed job belongs to another frozen protocol")
        for cid in conditions:
            path = folder / cid / "report.json"
            if marker["reports"].get(cid) != sha(path):
                raise ValueError("Incomplete/changed condition report")
            report = read(path)
            report_identity = report["identity"]
            if {k: report_identity[k] for k in identity} != identity or report_identity["condition"] != cid:
                raise ValueError("Condition report identity differs")
            if [r["sample_id"] for r in report["records"]] != bundle_identity["selected_ids"]:
                raise ValueError("Conditions do not use exactly the same cohort/order")
            reports[(job_id(job), cid)] = report
    for method in dict.fromkeys(j["method"] for j in jobs):
        method_jobs = [j for j in jobs if j["method"] == method]
        for cid in conditions:
            metrics, drops = {}, {}
            for metric in METRICS:
                values = [reports[(job_id(j), cid)]["metrics"][metric] for j in method_jobs]
                clean = [reports[(job_id(j), "clean")]["metrics"][metric] for j in method_jobs]
                metrics[metric] = distribution(values)
                drops[metric] = distribution([None if v is None or c is None else c - v for c, v in zip(clean, values)])
            result.append({"method": method, "condition": cid, "seeds": [j["seed"] for j in method_jobs],
                           "metrics": metrics, "clean_minus_condition": drops})
    summary = {"schema": "mars-robustness-summary-v1", "split": bundle_identity["split"],
               "images_per_condition": len(bundle_identity["selected_ids"]), "rows": result,
               "bundle_complete_sha256": bundle_sha256,
               "notes": ["Mean/std across training seeds, not an ensemble or confidence interval; no best-seed selection.",
                         "SAM is one frozen full-system baseline; no training-seed std.",
                         "Positive clean_minus_condition means degradation; thresholds and weights unchanged.",
                         "Shear uses aligned nearest-neighbor targets, ignoring exterior pixels (255). Other effects keep all GT pixels.",
                         "Synthetic overlays on thermal-IR intensity, not calibrated physical sensor corruptions.",
                         "Catalog coverage is not exhaustive visible-crater truth; FP is not automatically a hallucination."]}
    atomic(output / "summary.json", summary)
    lines = ["# Mars robustness", "", f"Split: {summary['split']}; images per condition: {summary['images_per_condition']}.", "",
             "IoU and clean-relative drop are in percentage points. ± is sample std across training seeds, not a confidence interval.", "",
             "| Approach | Condition | Scene IoU (%) | Clean − condition (pp) |", "|---|---|---:|---:|"]
    for row in result:
        def formatted(item):
            if item["mean"] is None:
                return "undefined"
            suffix = f" ± {100 * item['sample_std']:.2f}" if item["sample_std"] is not None else ""
            return f"{100 * item['mean']:.2f}" + suffix
        lines.append(f"| {row['method']} | {row['condition']} | {formatted(row['metrics']['scene_macro_iou'])} | {formatted(row['clean_minus_condition']['scene_macro_iou'])} |")
    lines.extend(["", *summary["notes"], ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return {"summary": str(output / "summary.json"), "table": str(output / "summary.md"), "complete": True}
