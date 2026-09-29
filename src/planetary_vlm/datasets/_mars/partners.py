"""Metadata-only partner search: geometric candidates, never confirmed depth GT."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from planetary_vlm.datasets.base import PROJECT_ROOT as ROOT
from planetary_vlm.datasets.cahvor import pixel_rays, project_direction
from .labels import inspect_label

DISTANCES = (0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalized(product):
    """Only normalize omitted non-video GOP zero, never scene identity."""
    return re.sub(r"E0*(\d+)_DXXX$", lambda m: f"E{int(m[1])}_DXXX", product.upper())


def identity_hypothesis(benchmark_id, archive_id):
    """Filename hypotheses require image registration, not automatic aliases."""
    benchmark_id, archive_id = benchmark_id.upper(), archive_id.upper()
    if re.sub(r"E\d+_DXXX$", "", benchmark_id) == re.sub(r"E\d+_DXXX$", "", archive_id):
        return "same_capture_other_compression_version"
    legacy = re.fullmatch(r"(\d{4}M[LR])(\d{4})(\d{3})000E\d+_DXXX", benchmark_id)
    if legacy and archive_id.startswith(legacy[1] + legacy[2].zfill(6) + legacy[3]):
        return "legacy_short_id_sequence_command_hypothesis"
    return None


def targets(data_root=None):
    data_root = Path(data_root) if data_root is not None else ROOT / "data_rover"
    return [json.loads(x) for x in (data_root / "interim/mars_bench_msl_v1/test/provenance.jsonl").read_text().splitlines()]


def requests(data_root=None):
    data_root = Path(data_root) if data_root is not None else ROOT / "data_rover"
    RAW = data_root / "raw/mastcam_partner_search_v1"
    bench = {normalized(r["mission_image_id"]) for r in targets(data_root)}
    by_sol = defaultdict(list)
    for sid in bench:
        by_sol[sid[:4]].append(sid)
    chosen = {}
    counts = Counter()
    for directory in load(RAW / "catalog.json"):
        if "error" in directory:
            continue
        anchors = by_sol[directory["sol"]]
        for name in directory["label_names"]:
            sid = normalized(Path(name).stem)
            if sid in bench:
                priority = 0
            elif any(identity_hypothesis(a, sid) for a in anchors):
                priority = 0
            elif any(sid[4:6] != a[4:6] and sid[6:12] == a[6:12] for a in anchors):
                priority = 1
            elif any(sid[4:6] != a[4:6] for a in anchors):
                priority = 2
            else:
                continue
            # Repeated products across archive volumes must not duplicate requests.
            chosen.setdefault(name.lower(), {"name": name.lower(), "url": directory["url"] + name, "priority": priority})
    ordered = sorted(chosen.values(), key=lambda r: (r["priority"], r["name"]))
    if len(ordered) > 12000:
        raise ValueError(f"{len(ordered)} labels exceed bounded batch: do not silently truncate")
    for row in ordered:
        counts[row["priority"]] += 1
    (RAW / "label_requests.json").write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    print(json.dumps({"requested_labels": len(ordered), "priority_counts": dict(counts)}))
    return RAW / "label_requests.json"


def rays(model):
    xx, yy = np.meshgrid(np.linspace(4, model["axes"]["Sample"]-5, 15), np.linspace(4, model["axes"]["Line"]-5, 15))
    return pixel_rays(model, np.stack((xx, yy), axis=-1).reshape(-1, 2))


def coverage(source, destination, source_rays):
    points = np.asarray(source["C"])[None, None, :] + np.asarray(DISTANCES)[..., None, None]*source_rays[None, :, :]
    xy, forward = project_direction(destination, points-np.asarray(destination["C"]))
    inside = forward & (xy[..., 0] >= 4) & (xy[..., 0] < destination["axes"]["Sample"]-4) & (xy[..., 1] >= 4) & (xy[..., 1] < destination["axes"]["Line"]-4)
    return inside.mean(axis=1)


def candidate(left, right, ray_cache):
    if any(left[k] != right[k] for k in ("frame", "frame_indices", "frame_solution")):
        return None, "coordinate_frame_mismatch"
    baseline = float(np.linalg.norm(np.asarray(left["C"])-right["C"]))
    if not 0.15 <= baseline <= 0.4:
        return None, "baseline_outside_mastcam_search_gate"
    lc = coverage(left, right, ray_cache[left["product_id"]])
    rc = coverage(right, left, ray_cache[right["product_id"]])
    acceptable = (rc >= 0.2) & (lc >= 0.02)
    if not acceptable.any():
        return None, "insufficient_predicted_overlap"
    index = int(np.argmax(np.where(acceptable, rc, -1)))
    delta = abs((datetime.fromisoformat(left["start_date_time"].replace("Z", "+00:00"))-datetime.fromisoformat(right["start_date_time"].replace("Z", "+00:00"))).total_seconds())
    axis_cos = np.dot(left["A"], right["A"]) / (np.linalg.norm(left["A"])*np.linalg.norm(right["A"]))
    return {"left": left["product_id"], "right": right["product_id"], "sol": left["product_id"][:4], "baseline_m": baseline, "time_delta_seconds": delta, "same_sequence": left["product_id"][6:12] == right["product_id"][6:12], "optical_axis_separation_deg": float(np.rad2deg(np.arccos(np.clip(axis_cos, -1, 1)))), "distance_m_at_best_predicted_right_coverage": DISTANCES[index], "predicted_left_coverage": float(lc[index]), "predicted_right_coverage": float(rc[index]), "coverage_by_assumed_distance": [{"distance_m": d, "left": float(l), "right": float(r)} for d, l, r in zip(DISTANCES, lc, rc)], "status": "metadata_geometry_candidate_not_image_verified", "occlusion_and_actual_scene_depth": "unknown", "image_verified": False, "depth_reconstructed": False}, None


def audit(data_root=None):
    data_root = Path(data_root) if data_root is not None else ROOT / "data_rover"
    RAW = data_root / "raw/mastcam_partner_search_v1"
    OUT = data_root / "source_audit_v1/mastcam_partner_search"
    OUT.mkdir(parents=True, exist_ok=True)
    models, parse_errors = {}, []
    recorded = {}
    for path in (RAW / "label_acquisition.json", RAW / "label_acquisition_aliases.json"):
        if path.exists():
            recorded.update({r["name"]: r for r in load(path)})
    requested = load(RAW / "label_requests.json")
    completed = all(r["name"] in recorded for r in requested)
    acquisition = list(recorded.values()) if completed else [dict(r, status="downloaded") for r in requested if (RAW / "labels" / r["name"]).exists()]
    for record in acquisition:
        if record["status"] != "downloaded":
            parse_errors.append(record)
            continue
        path = RAW / "labels" / record["name"]
        try:
            model = inspect_label(path, OUT, decode_image=False)
            model.update(label_url=record["url"], image_url=record["url"].rsplit("/", 1)[0]+"/"+model["source_image"])
            models[normalized(model["product_id"])] = model
        except Exception as exc:
            parse_errors.append({"name": record["name"], "error": str(exc)})
    by_sol = defaultdict(lambda: defaultdict(list))
    ray_cache = {}
    for model in models.values():
        try:
            ray_cache[model["product_id"]] = rays(model)
            by_sol[model["product_id"][:4]][model["product_id"][4:6]].append(model)
        except ValueError as exc:
            parse_errors.append({"name": model["product_id"], "error": str(exc)})
    pairs, anchor_results, rejected = {}, [], Counter()
    for anchor in targets(data_root):
        key = normalized(anchor["mission_image_id"])
        model = models.get(key)
        accepted = []
        if model and model["product_id"] in ray_cache:
            opposite = "MR" if key[4:6] == "ML" else "ML"
            for other in by_sol[key[:4]][opposite]:
                left, right = (model, other) if key[4:6] == "ML" else (other, model)
                pair_key = (left["product_id"], right["product_id"])
                if pair_key not in pairs:
                    record, reason = candidate(left, right, ray_cache)
                    pairs[pair_key] = record
                    if reason:
                        rejected[reason] += 1
                record = pairs[pair_key]
                if record:
                    accepted.append(record)
        coverage_field = "predicted_left_coverage" if key[4:6] == "ML" else "predicted_right_coverage"
        accepted.sort(key=lambda p: (-p[coverage_field], p["time_delta_seconds"], not p["same_sequence"]))
        distinct = {}
        for record in accepted:
            distinct.setdefault((record["left"][:22], record["right"][:22]), record)
        anchor_results.append({"sample_id": anchor["sample_id"], "benchmark_product_id": anchor["mission_image_id"], "archive_product_id": model["product_id"] if model else None, "eye": key[4:6], "status": "metadata_candidate_found" if accepted else ("source_label_missing" if not model else "no_partner_passed_geometry_gate"), "candidate_count": len(distinct), "candidate_product_variants": len(accepted), "best_candidates": list(distinct.values())[:3]})
    accepted_pairs = [p for p in pairs.values() if p]
    identity_candidates = []
    for anchor in anchor_results:
        if anchor["archive_product_id"] is None:
            hypotheses = [{"archive_product_id": m["product_id"], "hypothesis": identity_hypothesis(anchor["benchmark_product_id"], m["product_id"]), "label_url": m["label_url"]} for m in models.values() if identity_hypothesis(anchor["benchmark_product_id"], m["product_id"])]
            if hypotheses:
                identity_candidates.append({"sample_id": anchor["sample_id"], "benchmark_product_id": anchor["benchmark_product_id"], "status": "image_registration_required_before_identity_acceptance", "hypotheses": hypotheses})
    accepted_ids = {p[k] for p in accepted_pairs for k in ("left", "right")}
    source_records = [m for m in models.values() if m["product_id"] in accepted_ids]
    summary = {"status": "metadata_search_not_confirmed_stereo_dataset", "acquisition_complete": completed, "benchmark_frames": len(anchor_results), "benchmark_sols": len({r["mission_image_id"][:4] for r in targets(data_root)}), "catalog_directories": len(load(RAW / "catalog.json")), "catalog_errors": [r for r in load(RAW / "catalog.json") if "error" in r], "downloaded_parseable_camera_labels": len(models), "label_errors": len(parse_errors), "benchmark_source_labels_found": sum(a["archive_product_id"] is not None for a in anchor_results), "unique_geometric_candidate_pairs": len(accepted_pairs), "benchmark_frames_with_candidates": sum(a["candidate_count"] > 0 for a in anchor_results), "candidate_anchors_by_eye": dict(Counter(a["eye"] for a in anchor_results if a["candidate_count"])), "anchors_with_best_predicted_coverage_ge_50_percent": sum(a["candidate_count"] > 0 and a["best_candidates"][0]["predicted_left_coverage" if a["eye"] == "ML" else "predicted_right_coverage"] >= .5 for a in anchor_results), "rejected_pair_counts": dict(rejected), "new_image_verified_pairs": 0, "new_depth_maps": 0, "protocol": {"same_sol_only": True, "same_indexed_camera_coordinate_frame_required": True, "baseline_gate_m": [.15, .4], "sample_grid": [15, 15], "hypothetical_ranges_m": list(DISTANCES), "min_right_coverage": .2, "min_left_coverage": .02, "coverage_is_hypothetical_not_observed": True, "no_images_downloaded": True}, "catalog_sha256": hashlib.sha256((RAW / "catalog.json").read_bytes()).hexdigest()}
    summary["unresolved_anchors_with_filename_identity_hypotheses"] = len(identity_candidates)
    summary["candidate_product_variant_pairs"] = len(accepted_pairs)
    summary["unique_geometric_candidate_pairs"] = len({(p["left"][:22], p["right"][:22]) for p in accepted_pairs})
    summary["right_anchors_with_near_overlap_ge_80_percent_and_time_gap_le_120s"] = sum(a["eye"] == "MR" and any(p["time_delta_seconds"] <= 120 and any(c["distance_m"] <= 30 and c["right"] >= .8 for c in p["coverage_by_assumed_distance"]) for p in a["best_candidates"]) for a in anchor_results)
    pilot_path = data_root / "source_audit_v1/mastcam_pair_verification_v1/verification.json"
    if pilot_path.exists():
        pilot = load(pilot_path)
        summary["bounded_image_pilot"] = {k: pilot[k] for k in ("attempted_pairs", "pairs_passing_epipolar_qa", "pairs_passing_qa_and_native_coverage_ge_50_percent", "benchmark_admitted_depth_maps")}
        summary["new_image_verified_pairs"] = pilot["pairs_passing_epipolar_qa"]
        summary["new_depth_maps"] = sum("right_source_valid_fraction" in r for r in pilot["results"])
        summary["image_qa_definition"] = "Held-out epipolar correspondence QA, not independent metric-depth validation"
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, content in (("summary", summary), ("anchor_candidates", anchor_results), ("pairs", accepted_pairs), ("source_camera_models", source_records), ("errors", parse_errors), ("identity_candidates", identity_candidates)):
        (OUT / f"{name}.json").write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return OUT / "summary.json"
