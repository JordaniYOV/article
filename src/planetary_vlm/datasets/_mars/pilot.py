"""Bounded image-level checks of six new partners from the metadata search."""
import json
from pathlib import Path

from .labels import inspect_label
from .stereo import run
from .partners import ROOT, load


def select(data_root=None):
    data_root = Path(data_root) if data_root is not None else ROOT / "data"
    RAW = data_root / "raw/mastcam_partner_search_v1"
    OUT = data_root / "source_audit_v1/mastcam_partner_search"
    models = {r["product_id"]: r for r in load(OUT / "source_camera_models.json")}
    eligible = []
    for anchor in load(OUT / "anchor_candidates.json"):
        if anchor["eye"] != "MR":
            continue
        for p in anchor["best_candidates"]:
            if p["sol"] in ("0703", "0788") or p["time_delta_seconds"] > 90 or not p["same_sequence"]:
                continue
            near = next(r for r in p["coverage_by_assumed_distance"] if r["distance_m"] == 3)
            if near["right"] < .9:
                continue
            if any(models[p[k]]["axes"]["Band"] != 3 or models[p[k]]["axes"]["Sample"] > 1600 or models[p[k]]["axes"]["Line"] > 1200 for k in ("left", "right")):
                continue
            eligible.append(p)
    eligible.sort(key=lambda p: (p["time_delta_seconds"], p["optical_axis_separation_deg"]))
    chosen = {}
    for p in eligible:
        chosen.setdefault(p["sol"], p)
        if len(chosen) == 6:
            break
    selected = list(chosen.values())
    requests = [{"product_id": models[p[k]]["product_id"], "label_name": models[p[k]]["product_id"].lower()+".xml", "image_name": models[p[k]]["source_image"], "image_url": models[p[k]]["image_url"]} for p in selected for k in ("left", "right")]
    (RAW / "verification_requests.json").write_text(json.dumps(requests, indent=2), encoding="utf-8")
    (RAW / "verification_pairs.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps({"selected_pairs": len(selected), "sols": list(chosen)}, indent=2))
    return RAW / "verification_pairs.json"


def verify(data_root=None, output=None):
    data_root = Path(data_root) if data_root is not None else ROOT / "data"
    RAW = data_root / "raw/mastcam_partner_search_v1"
    SOURCE = data_root / "raw/mastcam_pair_verification_v1"
    DEST = Path(output) if output is not None else data_root / "source_audit_v1/mastcam_pair_verification_v2"
    if DEST.exists():
        raise FileExistsError("Pilot output exists; select a new version")
    DEST.mkdir(parents=True, exist_ok=False)
    images = DEST / "images"
    images.mkdir(exist_ok=True)
    models = [inspect_label(p, images) for p in sorted(SOURCE.glob("*.xml"))]
    pairs = load(RAW / "verification_pairs.json")
    audit = DEST / "audit.json"
    audit.write_text(json.dumps({"labels": models, "pairs": pairs}, indent=2), encoding="utf-8")
    results = []
    for p in pairs:
        output = DEST / "reconstructions" / (p['left'] + '__' + p['right'])
        try:
            run(audit, output, p["sol"], refine_pointing=True)
            report = load(output / "report.json")
            results.append({"sol": p["sol"], "left": p["left"], "right": p["right"], "status": report["status"], "rectification_qa_pass": report["rectification_qa_pass"], "right_source_valid_fraction": report["right_source_depth"]["valid_fraction"], "report_path": str(output / "report.json")})
        except Exception as exc:
            results.append({"sol": p["sol"], "left": p["left"], "right": p["right"], "status": "image_check_failed_not_admitted", "error": str(exc)})
    summary = {"status": "bounded_pilot_not_full_validation", "attempted_pairs": len(results), "pairs_passing_epipolar_qa": sum(r.get("rectification_qa_pass", False) for r in results), "pairs_passing_qa_and_native_coverage_ge_50_percent": sum(r.get("rectification_qa_pass", False) and r.get("right_source_valid_fraction", 0) >= .5 for r in results), "benchmark_admitted_depth_maps": 0, "results": results}
    (DEST / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return DEST / "verification.json"
