"""Evaluator-only two-image development controls; never scientific scoring GT."""

from dataclasses import replace
import json
import math
from pathlib import Path

from planetary_vlm.evaluation.parser import parse_answer
from planetary_vlm.inference.runner import file_hash

SCHEMA = "planetary-two-image-smoke-v1"
TASKS = ("crater_presence_v1", "crater_center_v1", "crater_largest_location_v1", "crater_half_count_v1")


def load_controls(root, path, requests, provenance):
    """Bind explicit val controls to actual images, not model outputs or empty masks."""
    from planetary_vlm.colab import within
    if path is None:
        raise ValueError("Q1-Q4 smoke/run requires --smoke-controls (separate evaluator-only JSON)")
    path = Path(path).resolve()
    if "data_rover" in {part.casefold() for part in path.parts}:
        raise ValueError("Protected archive cannot contain smoke controls")
    record = json.loads(path.read_text(encoding="utf-8"))
    controls = record.get("controls", [])
    if (record.get("schema") != SCHEMA or record.get("scientific_ground_truth") is not False
            or len(controls) != 2 or {c.get("expected_presence") for c in controls} != {"YES", "NO"}):
        raise ValueError("Need two explicit development controls, one YES and one NO; not scoring GT")
    origins = {row["request_id"]: row for row in provenance}
    selected, expected, images, seen, groups, hashes = [], {}, [], set(), set(), set()
    for control in controls:
        sid = control["sample_id"]
        if (sid in seen or control.get("source_split") != "val"
                or control.get("complete_for_size_rule") is not True or not control.get("review_basis")
                or not control.get("annotation_fingerprint")):
            raise ValueError("Smoke controls require distinct, explicitly reviewed val samples")
        seen.add(sid)
        chosen = [r for r in requests if r.request_id.split("__")[0] == sid
                  and r.request_id.endswith("__clean")]
        if len(chosen) != 4 or {r.request_id.split("__")[1] for r in chosen} != set(TASKS):
            raise ValueError("Each smoke image requires all four original clean Q1-Q4 requests")
        paths = {r.image_paths for r in chosen}
        if len(paths) != 1 or len(chosen[0].image_paths) != 1:
            raise ValueError("Smoke question inputs must identify one image per sample")
        for request in chosen:
            origin = origins[request.request_id]
            if (origin.get("source_split") != "val" or origin.get("condition") != "clean"
                    or origin.get("sample_id") != sid or origin.get("evaluation_eligible") is not True
                    or origin.get("image_sha256") != control["image_sha256"]):
                raise ValueError("Smoke controls differ from frozen val provenance")
        image_path = within(root, chosen[0].image_paths[0])
        if file_hash(image_path) != control["image_sha256"]:
            raise ValueError("Smoke control image bytes changed")
        origin = origins[chosen[0].request_id]
        if not origin.get("scene_group_id"):
            raise ValueError("Smoke controls require scene groups")
        groups.add(origin["scene_group_id"])
        hashes.add(control["image_sha256"])
        selected.extend(sorted(chosen, key=lambda r: TASKS.index(r.request_id.split("__")[1])))
        # A deliberately shorter Q1 control, outside the benchmark manifest.
        from PIL import Image
        with Image.open(image_path) as image:
            threshold = math.ceil(.04 * min(image.size))
        prompt = (f"Are any impact craters fully visible, at least {threshold} pixels wide and high, "
                  "with at least a 1 pixel margin from every image edge? Ignore pits and rocks. "
                  "Reply with exactly one label: YES or NO. No explanation.")
        short = replace(chosen[0], request_id=sid+"__smoke_presence_simple_v1__clean", prompt=prompt,
                        allowed_answers=("YES", "NO"), max_new_tokens=16)
        selected.append(short)
        expected[sid+"__crater_presence_v1__clean"] = control["expected_presence"]
        expected[short.request_id] = control["expected_presence"]
        images.append(str(image_path))
    if len(hashes) != 2 or len(groups) != 2:
        raise ValueError("Smoke controls must be different images from different scene groups")
    return {"schema": SCHEMA, "path": str(path), "sha256": file_hash(path),
            "record": record, "requests": selected, "expected": expected, "images": images}


def assess(rows, selected, expected, diagnostics):
    """Strict formats for ten requests; correctness only for explicit Q1 controls."""
    by_id = {r["request_id"]: r for r in rows}
    checks = []
    for request in selected:
        row = by_id.get(request.request_id, {})
        parsed = parse_answer(row.get("raw_response", ""), request.allowed_answers)
        target = expected.get(request.request_id)
        checks.append({"request_id": request.request_id, "raw_response": row.get("raw_response"),
            "parsed": parsed, "format_valid": parsed is not None,
            "expected_development_control": target, "control_correct": None if target is None else parsed == target,
            "runtime_ok": row.get("status") == "ok" and row.get("mock") is False})
    success = (len(by_id) == len(rows) == len(selected)
        and diagnostics.get("success") is True
        and all(c["runtime_ok"] and c["format_valid"] and c["control_correct"] is not False for c in checks))
    return {"schema": SCHEMA, "success": success, "scientific_score_available": False,
        "vision_checks_passed": diagnostics.get("success") is True, "checks": checks,
        "note": "Development smoke only. Q2-Q4 formats checked, not unaccepted scientific references."}
