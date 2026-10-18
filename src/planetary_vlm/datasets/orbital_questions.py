"""Versioned orbital question preparation; references never enter inference.

Only Q5/Q6 currently have usable reference arrays in the curated collection.
Other templates are retained with explicit per-image admission failures.
This is a reference/prompt pilot, not admission of the IBM output pipeline.
"""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re

from planetary_vlm.contracts import GroundTruth, ModelRequest
from planetary_vlm.datasets.manifest import request_from_dict
from planetary_vlm.io import read_jsonl, write_json, write_jsonl

VERSION = "orbital-questions-v1"
GRID = ("Split the image into 3 rows and 3 columns of equal extent. "
        "Number cells from left to right, top to bottom: 1 2 3 / 4 5 6 / 7 8 9. ")
TEMPLATES = {
    "crater_presence_v1": {
        "number": "Q1", "question": "Есть ли хотя бы один кратер учитываемого размера?",
        "allowed_answers": ["YES", "NO"],
        "blocked_reason": "size_rule_and_complete_crater_instances_not_available"},
    "crater_center_v1": {
        "number": "Q2", "question": "Есть ли центр кратера в области №5?",
        "allowed_answers": ["YES", "NO"],
        "blocked_reason": "complete_size_filtered_instance_centres_not_available"},
    "crater_largest_location_v1": {
        "number": "Q3", "question": "Где центр самого крупного полностью видимого кратера?",
        "allowed_answers": [str(i) for i in range(1, 10)] + ["NONE", "TIE"],
        "blocked_reason": "complete_instances_boundary_size_and_tie_rules_not_available"},
    "crater_half_count_v1": {
        "number": "Q4", "question": "В какой половине изображения больше центров кратеров?",
        "allowed_answers": ["LEFT", "RIGHT", "TIE"],
        "blocked_reason": "complete_size_filtered_crater_instances_not_available"},
    "height_pair_v1": {
        "number": "Q5", "question": "Какая область выше: A (№1) или B (№9)?",
        "allowed_answers": ["A", "B", "TIE"]},
    "height_lowest_v1": {
        "number": "Q6", "question": "В какой из девяти областей минимальная средняя высота?",
        "allowed_answers": [str(i) for i in range(1, 10)] + ["TIE"]},
    "route_choice_v1": {
        "number": "Q7", "question": "Какой из маршрутов A/B/C имеет минимальную заданную стоимость?",
        "allowed_answers": ["A", "B", "C", "TIE", "NONE"],
        "blocked_reason": "route_candidates_cost_model_constraints_and_output_admission_pending"},
    "route_grid_v1": {
        "number": "Q8", "question": "Построй минимальный по заданной стоимости путь из S в G.",
        "output": "ordered_grid_cell_sequence_or_NO_PATH",
        "blocked_reason": "route_graph_endpoints_cost_model_constraints_and_output_admission_pending"},
}
RULES = {
    "version": VERSION, "status": "reference_ready_pilot_not_final_scientific_admission",
    "grid": [3, 3], "grid_order": "row_major",
    "integer_boundary": "floor(k*dimension/3); half_open; full_pixel_coverage",
    "pair_cells": {"A": 1, "B": 9}, "minimum_valid_fraction_per_cell": 0.95,
    "tie_tolerance_fraction_of_valid_tile_height_range": 1e-6,
    "pair_sampling": "fixed_cells_independent_of_answer",
    "input_regime": "RGB_only_reference_height_hidden",
    "supported_conditions": ["clean"],
    "physical_slope_or_safety_claim": False,
}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def orbital_path(root, relative):
    """Check scope and symlink resolution before opening any supplied path."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe orbital path")
    parts = PurePosixPath(relative)
    if (parts.is_absolute() or ".." in parts.parts or any(ord(c) < 32 for c in relative)
            or any(p.casefold() == "data_rover" for p in parts.parts)):
        raise ValueError("Unsafe orbital path")
    path = (root / relative).resolve()
    if not path.is_relative_to((root / "data_orbital").resolve()):
        raise ValueError("Asset must stay inside data_orbital")
    if any(p.casefold() == "data_rover" for p in path.parts):
        raise ValueError("Protected archive cannot be accessed")
    return path


def prompt_for(task_id):
    if task_id not in {"height_pair_v1", "height_lowest_v1"}:
        raise ValueError("Task has no admitted reference converter")
    question = (
        "Which region has the higher mean surface elevation: A (cell 1) or B (cell 9)? "
        if task_id == "height_pair_v1" else
        "Which cell has the lowest mean surface elevation? "
    )
    answers = TEMPLATES[task_id]["allowed_answers"]
    return (GRID + question + "Infer relative elevation from the image only, not brightness alone. "
            "Means are defined over valid reference terrain pixels. Values within 0.000001 "
            "times the tile's valid elevation range are tied. "
            + ("Use TIE if the two means are tied. " if task_id == "height_pair_v1" else
               "Use TIE if multiple cells are tied for the lowest mean. ")
            + "Reply with exactly one label: " + ", ".join(answers) + ". No explanation.")


def height_answers(height, valid, image_size):
    """No guessed instances, no physical slopes, and no invalid-pixel means."""
    import numpy as np
    height, valid = np.asarray(height), np.asarray(valid)
    if (height.ndim != 2 or height.shape != valid.shape
            or height.shape != (image_size[1], image_size[0]) or min(height.shape) < 3):
        return {}, {"reason": "height_validity_or_image_shape_mismatch"}
    if height.dtype.kind not in "fiu" or valid.dtype.kind not in "biu" or not np.isin(valid, [0, 1]).all():
        return {}, {"reason": "invalid_height_or_validity_dtype_or_values"}
    valid = valid.astype(bool) & np.isfinite(height)
    means, coverage = [], []
    for r in range(3):
        for c in range(3):
            ys = slice(r * height.shape[0] // 3, (r + 1) * height.shape[0] // 3)
            xs = slice(c * height.shape[1] // 3, (c + 1) * height.shape[1] // 3)
            selected = valid[ys, xs]
            coverage.append(float(selected.mean()))
            means.append(float(height[ys, xs][selected].astype(np.float64).mean()) if selected.any() else None)
    values = height[valid].astype(np.float64)
    if not len(values):
        return {}, {"reason": "no_valid_height_pixels", "valid_fractions": coverage}
    span = float(values.max() - values.min())
    if not np.isfinite(span) or any(m is not None and not np.isfinite(m) for m in means):
        return {}, {"reason": "nonfinite_height_statistics"}
    tolerance = RULES["tie_tolerance_fraction_of_valid_tile_height_range"] * span
    minimum = RULES["minimum_valid_fraction_per_cell"]
    result = {}
    if coverage[0] >= minimum and coverage[8] >= minimum:
        delta = means[0] - means[8]
        result["height_pair_v1"] = "TIE" if abs(delta) <= tolerance else ("A" if delta > 0 else "B")
    if min(coverage) >= minimum:
        lowest = min(means)
        candidates = [i + 1 for i, mean in enumerate(means) if mean - lowest <= tolerance]
        result["height_lowest_v1"] = str(candidates[0]) if len(candidates) == 1 else "TIE"
    return result, {"reason": "reference_ready" if len(result) == 2 else "insufficient_valid_reference_coverage", "valid_fractions": coverage,
                    "region_means": means, "height_range": span, "tie_tolerance": tolerance}


def build_questions(root, output):
    """Offline, exclusive versioned build. No model, downloads or training."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if ("data_rover" in {p.casefold() for p in root.parts + output.parts}
            or not output.is_relative_to(root / "data_orbital")
            or output.is_relative_to(root / "data_orbital" / "orbital_300_v3")
            or output == root / "data_orbital"):
        raise ValueError("Use a new question folder under data_orbital, outside the source collection")
    if output.exists():
        raise FileExistsError("Question version already exists; use a new output directory")
    from PIL import Image
    import numpy as np
    rows, seen, manifests, train_groups = [], set(), {}, set()
    for planet in ("moon", "mars"):
        for track in ("segmentation", "height", "train"):
            relative = f"data_orbital/orbital_300_v3/{planet}/{track}_manifest.jsonl"
            path = orbital_path(root, relative)
            manifests[relative] = file_hash(path)
            for row in read_jsonl(path):
                sid = row.get("sample_id")
                if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sid) or sid in seen:
                    raise ValueError("Source IDs must be safe, unique and nonempty")
                seen.add(sid)
                row = {**row, "track": track, "planet": planet}
                if track == "train" or row.get("source_split") == "train":
                    train_groups.add((planet, row.get("source_dataset"), row.get("scene_group_id")))
                rows.append(row)
    audit = json.loads(orbital_path(root, "data_orbital/orbital_300_v3/audit.json").read_text(encoding="utf-8"))
    if len(seen) != audit["total_images"]:
        raise ValueError("Collection count differs from source audit")
    requests, targets, provenance, admission, references = [], [], [], [], []
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        if (row["track"] == "train" or row.get("source_split") not in {"val", "test"}
                or row.get("evaluation_eligible") is not True or row.get("model_inference_use") is False):
            continue
        sid = row["sample_id"]
        if not isinstance(row.get("scene_group_id"), str) or not row["scene_group_id"]:
            raise ValueError("Every evaluation source needs a scene group")
        answers, detail = {}, {"reason": "no_height_reference_for_this_track"}
        image = orbital_path(root, row["image"])
        if file_hash(image) != row["image_sha256"]:
            raise ValueError(f"Changed source image: {sid}")
        if row["track"] == "height":
            paths = {}
            for field in ("height_map", "height_valid_mask"):
                path = orbital_path(root, row[field])
                if file_hash(path) != row[field + "_sha256"]:
                    raise ValueError(f"Changed reference asset: {sid}/{field}")
                paths[field] = path
            with Image.open(image) as original:
                answers, detail = height_answers(np.load(paths["height_map"], allow_pickle=False),
                    np.load(paths["height_valid_mask"], allow_pickle=False), original.size)
            if (row["planet"], row.get("source_dataset"), row["scene_group_id"]) in train_groups:
                answers, detail = {}, {"reason": "scene_group_also_in_publisher_train"}
            references.append({"source_sample_id": sid, "source_split": row["source_split"],
                "height_map_sha256": row["height_map_sha256"],
                "height_valid_mask_sha256": row["height_valid_mask_sha256"], **detail})
        for task_id, template in TEMPLATES.items():
            admission.append({"source_sample_id": sid, "task_id": task_id, "source_split": row["source_split"],
                "reference_admitted": task_id in answers, "paired_ibm_admitted": False,
                "reason": "reference_ready_pilot_only" if task_id in answers else
                    template.get("blocked_reason", detail["reason"])})
        for task_id, answer in answers.items():
            rid = f"{sid}__{task_id}__clean"
            requests.append(asdict(ModelRequest(rid, (row["image"],), prompt_for(task_id),
                tuple(TEMPLATES[task_id]["allowed_answers"]), max_new_tokens=16)))
            group = f"{row['planet']}:{row['source_dataset']}:{row['scene_group_id']}"
            targets.append(asdict(GroundTruth(rid, answer, sid, group, VERSION, task_id, "clean")))
            provenance.append({"request_id": rid, "sample_id": sid, "task_id": task_id,
                **{k: row[k] for k in ("planet", "track", "source_dataset", "source_revision", "source_split",
                    "scene_group_id", "source_image_id", "license_claim", "image_sha256", "height_units") if k in row},
                "final_test_eligible": row.get("final_test_eligible") is True,
                "condition": "clean", "derivation_version": VERSION})
    output.mkdir(parents=True, exist_ok=False)
    payload = {"requests.jsonl": requests, "targets.jsonl": targets, "provenance.jsonl": provenance,
               "admission.jsonl": admission, "reference_audit.jsonl": references}
    for name, records in payload.items():
        write_jsonl(output / name, records)
    write_json(output / "templates.json", {"rules": RULES, "templates": TEMPLATES})
    summary = {"schema": VERSION, "rules": RULES, "source_manifests_sha256": manifests,
        "converter_sha256": file_hash(__file__), "renderer": {"numpy": np.__version__, "Pillow": Image.__version__},
        "collection_images": len(seen), "eligible_sources": len(admission) // len(TEMPLATES),
        "selected_images": len({r["sample_id"] for r in provenance}), "requests": len(requests),
        "by_task_split": dict(Counter(f"{r['task_id']}/{r['source_split']}" for r in provenance)),
        "reference_blocks": dict(Counter(r["reason"] for r in admission if not r["reference_admitted"])),
        "files_sha256": {name: file_hash(output / name) for name in (*payload, "templates.json")},
        "scientific_result": False, "paired_ibm_admitted": False,
        "limitations": ["Height alignment inherited from collection, not independently revalidated.",
            "Q1-Q4 require audited instances and completeness, not semantic connected components.",
            "Q7-Q8 have no frozen admitted graph/cost/output pipeline.",
            "Val is development. Model training/geographic overlap audit is still incomplete.",
            "Clean RGB-only Q5/Q6 pilot; no measured traversability or physical slopes."]}
    summary["bundle_sha256"] = digest(summary)
    write_json(output / "metadata.json", summary)
    return summary


def load_question_requests(root, folder):
    """Validate a local question build; return model-visible rows only."""
    root, folder = Path(root).resolve(), Path(folder).resolve()
    folder = orbital_path(root, folder.relative_to(root).as_posix())
    record = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    claimed = record.pop("bundle_sha256")
    if record.get("schema") != VERSION or digest(record) != claimed:
        raise ValueError("Question metadata changed")
    if record["rules"] != RULES:
        raise ValueError("Question rules differ from this converter")
    for name, expected in record["files_sha256"].items():
        path = orbital_path(root, (folder.relative_to(root) / name).as_posix())
        if path.parent != folder or file_hash(path) != expected:
            raise ValueError("Question build file changed")
    indexed = {}
    for row in read_jsonl(folder / "requests.jsonl"):
        request = request_from_dict(row)
        parts = request.request_id.split("__")
        if len(parts) != 3 or parts[2] != "clean" or parts[1] not in {"height_pair_v1", "height_lowest_v1"}:
            raise ValueError("Unsupported question request")
        if (request.prompt != prompt_for(parts[1]) or request.do_sample or request.max_new_tokens != 16
                or request.allowed_answers != tuple(TEMPLATES[parts[1]]["allowed_answers"])
                or len(request.image_paths) != 1 or request.modality_paths):
            raise ValueError("Question inference contract changed")
        if request.request_id in indexed:
            raise ValueError("Duplicate question request")
        orbital_path(root, request.image_paths[0])
        indexed[request.request_id] = request
    if len(indexed) != record["requests"] or not indexed:
        raise ValueError("Empty or inconsistent question request inventory")
    record["bundle_sha256"] = claimed
    return list(indexed.values()), record
