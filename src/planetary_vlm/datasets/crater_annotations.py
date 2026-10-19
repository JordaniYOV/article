"""Offline, versioned CVAT/source-instance merge with separate candidate QA.

Publisher masks, source splits and image bytes are immutable. No automatic IoU
merge, semantic-component instances, implicit review acceptance or inference.
"""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tomllib
import xml.etree.ElementTree as ET
import zipfile

from planetary_vlm.contracts import GroundTruth, ModelRequest
from planetary_vlm.datasets.crater_reference import TASKS, VERSION, candidate_answers, geometry, prompt_for
from planetary_vlm.datasets.crater_sources import SourceAnnotations
from planetary_vlm.datasets.manifest import request_from_dict
from planetary_vlm.datasets.orbital_questions import digest, file_hash, orbital_path
from planetary_vlm.io import read_jsonl, write_json, write_jsonl

SCHEMA = "crater-annotations-v1"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
VISIBILITY_REVIEW_IDS = {
    "moon_seg_018__cvat_1d6deba4ef902a52",
    "mars_seg_033__cvat_c2dda6c247783f99",
    "mars_seg_046__cvat_ba529f78a09ee0da",
}


def attributes(node, expected):
    pairs = [(a.get("name"), a.text) for a in node.findall("attribute")]
    if len(pairs) != len(dict(pairs)) or set(dict(pairs)) != set(expected):
        raise ValueError("Missing, duplicate or unsupported CVAT attributes")
    return dict(pairs)


def explicit_bool(value):
    if value not in {"true", "false"}:
        raise ValueError("CVAT checkboxes require true or false")
    return value == "true"


def import_export(path, expected_rows, dimensions, rules):
    """Parse only bounded annotations.xml; never extract untrusted ZIP paths."""
    path = Path(path)
    if any(p.casefold() == "data_rover" for p in path.parts):
        raise ValueError("Protected archive cannot be accessed")
    path = path.resolve()
    if any(p.casefold() == "data_rover" for p in path.parts):
        raise ValueError("Protected archive cannot be accessed")
    raw_zip = path.read_bytes()
    if len(raw_zip) > 10 * 1024**2:
        raise ValueError("Use a small CVAT annotations-only export")
    with zipfile.ZipFile(__import__("io").BytesIO(raw_zip)) as bundle:
        if bundle.namelist().count("annotations.xml") != 1:
            raise ValueError("Expected exactly one annotations.xml")
        info = bundle.getinfo("annotations.xml")
        if info.file_size > 2 * 1024**2:
            raise ValueError("CVAT XML exceeds import limit")
        raw_xml = bundle.read(info)
    if b"<!DOCTYPE" in raw_xml.upper() or b"<!ENTITY" in raw_xml.upper():
        raise ValueError("DTD/entity declarations are not accepted")
    document = ET.fromstring(raw_xml)
    if document.tag != "annotations" or document.findtext("version") != "1.1":
        raise ValueError("Use native CVAT for images 1.1")
    nodes = document.findall("image")
    names = [n.get("name", "") for n in nodes]
    if any(Path(name).name != name or not name.endswith(".png") for name in names):
        raise ValueError("Use original <sample_id>.png frame names")
    ids = [Path(name).stem for name in names]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_rows):
        raise ValueError("CVAT frames do not exactly match the expected planet/source split")
    samples = {}
    for sid, node in zip(ids, nodes):
        width, height = dimensions[sid]
        if (int(node.get("width")), int(node.get("height"))) != (width, height):
            raise ValueError("Native frame dimensions changed")
        if any(child.tag not in {"box", "tag"} for child in node):
            raise ValueError("Only crater boxes and image_review tags are supported")
        tags = node.findall("tag")
        if len(tags) != 1 or tags[0].get("label") != "image_review":
            raise ValueError("Every image requires exactly one image_review tag")
        tag = attributes(tags[0], ("result", "complete_for_size_rule"))
        if tag["result"] not in rules["cvat"]["result_values"]:
            raise ValueError("Unknown assigned CVAT result; no silent typo normalization")
        complete = explicit_bool(tag["complete_for_size_rule"])
        objects = []
        for index, box in enumerate(node.findall("box"), 1):
            if box.get("label") != "crater":
                raise ValueError("Unexpected object class")
            attrs = attributes(box, ("uncertain", "truncated"))
            xy = [float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            if not all(math.isfinite(v) for v in xy) or not (0 <= xy[0] < xy[2] <= width and 0 <= xy[1] < xy[3] <= height):
                raise ValueError("Invalid native CVAT bbox coordinates")
            obj = {"origin": "human_CVAT_supplement", "bbox_xyxy": xy, "class": "crater",
                   "uncertain": explicit_bool(attrs["uncertain"]), "truncated": explicit_bool(attrs["truncated"])}
            identity = digest(obj)[:16]
            obj.update(instance_id=f"{sid}__cvat_{identity}", xml_index_not_object_id=index)
            geometry(obj, width, height)
            objects.append(obj)
        if len({o["instance_id"] for o in objects}) != len(objects):
            raise ValueError("Exact duplicate CVAT boxes; review before import")
        samples[sid] = {"cvat_result": tag["result"], "complete_for_size_rule": complete,
                        "supplemental_instances": objects}
    return samples, raw_zip, {"archive_name": path.name, "archive_sha256": hashlib.sha256(raw_zip).hexdigest(),
                             "xml_sha256": hashlib.sha256(raw_xml).hexdigest(), "images_in_export": len(nodes),
                             "uploaded_CVAT_image_bytes_verified": False}


def overlap_pairs(objects, width, height):
    """Review heuristic only, never automatic identity or duplicate removal."""
    candidates = [o for o in objects if geometry(o, width, height)["eligible"]]
    pairs = []
    for i, a in enumerate(candidates):
        for b in candidates[i+1:]:
            if a["origin"] == b["origin"] == "human_CVAT_supplement":
                continue
            x1, y1, x2, y2 = a["bbox_xyxy"]
            u1, v1, u2, v2 = b["bbox_xyxy"]
            intersect = max(0, min(x2,u2)-max(x1,u1))*max(0, min(y2,v2)-max(y1,v1))
            union = (x2-x1)*(y2-y1) + (u2-u1)*(v2-v1) - intersect
            iou = intersect/union
            if iou >= .5:
                pairs.append({"instance_ids": [a["instance_id"], b["instance_id"]], "bbox_iou": iou,
                              "decision": "pending_manual_identity_review_not_automatic_merge"})
    return pairs


def apply_review(objects, pending_catalog, review, fingerprint):
    """Explicit hash-bound human decisions only; never accept generated templates."""
    if not review or review.get("review_status") != "accepted":
        return objects, False, False, None
    for key in ("annotator", "reviewer", "reviewed_at", "review_basis"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            raise ValueError("Accepted review needs attribution, time and evidence")
    try:
        moment = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
        if moment.tzinfo is None: raise ValueError("Review time needs a timezone")
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid review timestamp") from error
    if review.get("annotation_fingerprint") != fingerprint:
        raise ValueError("Review does not match this immutable annotation version")
    for key in ("complete_for_size_rule", "source_instances_complete", "duplicates_reviewed", "uncertainty_accounted_for"):
        if review.get(key) is not True:
            raise ValueError("Accepted review lacks an explicit completeness/identity/uncertainty decision")
    if not isinstance(review.get("unknown_regions"), list):
        raise ValueError("Review must explicitly preserve/resolve unknown regions")
    indexed = {o["instance_id"]: dict(o) for o in objects}
    catalog = {o["instance_id"]: o for o in pending_catalog}
    exclusions = review.get("excluded_instances", [])
    excluded = set()
    for decision in exclusions:
        iid = decision.get("instance_id")
        if iid not in indexed and iid not in catalog or iid in excluded or not decision.get("reason"):
            raise ValueError("Invalid/duplicate instance exclusion without a reason")
        excluded.add(iid)
        indexed.pop(iid, None)
    overridden = set()
    for replacement in review.get("instance_overrides", []):
        iid = replacement.get("instance_id")
        if iid in excluded or iid in overridden or (iid not in indexed and iid not in catalog) or not replacement.get("review_reason"):
            raise ValueError("Invalid instance geometry override")
        original = indexed.get(iid) or catalog[iid]
        indexed[iid] = {**original, **replacement, "origin": original["origin"], "geometry_reviewed": True}
        overridden.add(iid)
    if set(catalog) - excluded - overridden:
        raise ValueError("Every unresolved publisher catalog object requires reviewed geometry or an explicit exclusion")
    for added in review.get("additional_instances", []):
        iid = added.get("instance_id")
        if not isinstance(iid, str) or not SAFE_ID.fullmatch(iid) or iid in indexed or not added.get("review_reason"):
            raise ValueError("Invalid additional reviewed crater identity")
        indexed[iid] = {**added, "origin": "human_second_review_addition", "class": "crater"}
    return list(indexed.values()), True, True, review["unknown_regions"]


def confirmed_visibility(context, imported, exports):
    """Resolve named clipping concerns only, not full review or missing imagery.

    The human confirmation must bind the exact exports, ID, bbox and raw flags.
    This cannot clear arbitrary unknown regions or admit any scoring targets.
    """
    confirmations = (context or {}).get("visibility_confirmations", [])
    if not confirmations:
        return {}
    if (not isinstance(confirmations, list) or set(context.get("export_sha256", {})) != set(exports)
            or any(context["export_sha256"][name] != record["archive_sha256"] for name, record in exports.items())):
        raise ValueError("Visibility confirmations need exact hashes for all four exports")
    resolved = {}
    for item in confirmations:
        sid, iid = item.get("sample_id"), item.get("instance_id")
        if (sid not in imported or iid not in VISIBILITY_REVIEW_IDS or iid in resolved
                or item.get("decision") != "original_bbox_and_truncated_flag_confirmed"
                or not isinstance(item.get("reviewed_by"), str) or not item["reviewed_by"].strip()
                or not isinstance(item.get("review_basis"), str) or not item["review_basis"].strip()):
            raise ValueError("Invalid or unattributed per-instance visibility confirmation")
        matched = [o for o in imported[sid]["supplemental_instances"] if o["instance_id"] == iid]
        if (len(matched) != 1 or matched[0]["bbox_xyxy"] != item.get("bbox_xyxy")
                or type(item.get("truncated")) is not bool or matched[0]["truncated"] != item["truncated"]
                or matched[0]["uncertain"] is not False):
            raise ValueError("Visibility confirmation does not match exact raw CVAT geometry/flags")
        resolved[iid] = dict(item)
    return resolved


def build_crater_questions(root, output, exports, source_cache, *, reviews=None, context=None, previews=True,
                           collection="data_orbital/orbital_300_v3"):
    """Build auditable candidates; only explicitly accepted records become targets."""
    from PIL import Image, ImageDraw
    root = Path(root).resolve()
    def folder(value):
        path = Path(value)
        if not path.is_absolute(): path = root / path
        return orbital_path(root, path.resolve().relative_to(root).as_posix())
    output = folder(output)
    cache = folder(source_cache)
    collection = folder(collection)
    forbidden = (root / "data_orbital/orbital_300_v3", collection, root / "data_orbital/cvat_images", cache)
    if output == root / "data_orbital" or any(output.is_relative_to(p) for p in forbidden):
        raise ValueError("Use a new build outside source/staging/cache directories")
    if output.exists():
        raise FileExistsError("Immutable annotation build already exists; choose a new version")
    if set(exports) != {("moon", "val"), ("moon", "test"), ("mars", "val"), ("mars", "test")}:
        raise ValueError("Provide four separate moon/mars source-val/test exports")
    rules_path = root / "configs/crater_annotation_v2.toml"
    rules = tomllib.loads(rules_path.read_text(encoding="utf-8"))
    if rules["schema_version"] != "crater-annotation-v2" or not rules["annotation_rules_frozen"]:
        raise ValueError("Frozen v2 annotation rules required")
    # Fixed operational constants in the pure converter must match the protocol.
    if (rules["objects"]["minimum_short_side_fraction"], rules["visibility"]["image_border_margin_px"],
        rules["answers"]["q3_near_largest_fraction"]) != (.04, 1, .05):
        raise ValueError("Geometry rules changed; use a new converter version")
    manifest_hashes, rows, dimensions, train_groups = {}, [], {}, set()
    for planet in ("moon", "mars"):
        for track in ("segmentation", "train"):
            path = collection / planet / (track+"_manifest.jsonl")
            manifest_hashes[path.relative_to(root).as_posix()] = file_hash(path)
            for row in read_jsonl(path):
                verified_replacement = None
                if track == "train":
                    train_groups.add((planet, row["source_dataset"], row["scene_group_id"]))
                    continue
                sid = row["sample_id"]
                if (not SAFE_ID.fullmatch(sid) or row["planet"] != planet or row["source_split"] not in {"val", "test"}
                        or row.get("evaluation_eligible") is not True or not row.get("scene_group_id")):
                    raise ValueError("Invalid semantic evaluation provenance")
                for key in ("image", "mask"):
                    asset = orbital_path(root, row[key])
                    if file_hash(asset) != row[key+"_sha256"]:
                        raise ValueError("Changed source asset: " + sid + "/" + key)
                with Image.open(orbital_path(root, row["image"])) as image, Image.open(orbital_path(root, row["mask"])) as mask:
                    if image.size != mask.size: raise ValueError("Unaligned image and source mask")
                    dimensions[sid] = image.size
                    stage = orbital_path(root, f"data_orbital/cvat_images/{planet}/{row['source_split']}/{sid}.png")
                    with Image.open(stage) as staged:
                        same = image.size == staged.size and image.convert("RGB").tobytes() == staged.convert("RGB").tobytes()
                        if not same and row.get("image_replacement_record"):
                            from planetary_vlm.datasets.image_replacement import verify_replacement
                            replacement = verify_replacement(root,row)
                            verified_replacement = digest(replacement)
                            with Image.open(orbital_path(root,replacement["original_image"]["path"])) as anchor:
                                same = (anchor.size == image.size == staged.size
                                        and anchor.convert("RGB").tobytes() == staged.convert("RGB").tobytes())
                        if not same:
                            raise ValueError("CVAT staging image differs from source")
                    row = {**row, "annotation_image_sha256": file_hash(stage)}
                    if verified_replacement:
                        row["image_replacement_sha256"] = verified_replacement
                rows.append(row)
    if len(rows) != len({r["sample_id"] for r in rows}):
        raise ValueError("Duplicate source sample ID")
    snapshots, export_records, imported = {}, {}, {}
    for (planet, split), path in exports.items():
        expected = {r["sample_id"]: r for r in rows if r["planet"] == planet and r["source_split"] == split}
        selected, raw, record = import_export(path, expected, dimensions, rules)
        name = f"{planet}_{split}.zip"
        snapshots[name], export_records[name] = raw, record
        imported.update({sid: {**item, "annotation_archive": name, "annotation_archive_sha256": record["archive_sha256"]}
                         for sid, item in selected.items()})
    reader = SourceAnnotations(root, cache)
    accepted_reviews = {}
    if reviews is not None:
        review_path = folder(reviews)
        review_hash = file_hash(review_path)
        for decision in read_jsonl(review_path):
            sid = decision["sample_id"]
            if sid in accepted_reviews or sid not in imported: raise ValueError("Unknown/duplicate review sample ID")
            accepted_reviews[sid] = decision
    else:
        review_hash = None
    context_data = None
    if context is not None:
        context_path = folder(context)
        context_data = json.loads(context_path.read_text(encoding="utf-8"))
        for name, checksum in context_data.get("export_sha256", {}).items():
            if name not in export_records or export_records[name]["archive_sha256"] != checksum:
                raise ValueError("Human review context does not match these exports")
    visibility_confirmed = confirmed_visibility(context_data, imported, export_records)
    samples, instances, catalog, questions, candidates, requests, targets, provenance, review_queue = [], [], [], [], [], [], [], [], []
    panels = []
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        sid, (width, height) = row["sample_id"], dimensions[row["sample_id"]]
        annotation = imported[sid]
        source, unresolved, source_audit = reader.recover(row, (width, height), orbital_path(root, row["mask"]))
        supplement = [{**obj, "visibility_review": visibility_confirmed[obj["instance_id"]]}
                      if obj["instance_id"] in visibility_confirmed else obj
                      for obj in annotation["supplemental_instances"]]
        base_objects = source + supplement
        fingerprint = digest({"sample": row, "source_audit": source_audit, "source_objects": source,
                              "source_candidates": unresolved, "CVAT": annotation, "rules": file_hash(rules_path)})
        decision = accepted_reviews.get(sid)
        objects, reviewed, complete_instances, reviewed_regions = apply_review(base_objects, unresolved, decision, fingerprint)
        unknown_regions = []
        visual_cases = []
        if sid == "mars_seg_018":
            unknown_regions = [{"bbox_xyxy": [225,170,245,200], "reason": "publisher crater label on inspected uniform-gray pixels; visual evidence unresolved"}]
            regional_review = (context_data or {}).get("replacement_visibility_review", {})
            if regional_review.get("sample_id") == sid:
                from planetary_vlm.datasets.image_replacement import verify_replacement
                replacement = verify_replacement(root,row)
                if (regional_review.get("image_sha256") != row["image_sha256"]
                        or regional_review.get("replacement_record_sha256") != replacement.get("record_sha256", digest(replacement))
                        or regional_review.get("decision") != "previously_missing_surface_visible_catalog_geometry_inspected"
                        or not regional_review.get("reviewed_by") or not regional_review.get("review_basis")):
                    raise ValueError("Replacement visibility review does not match exact new image/provenance")
                unknown_regions = []
        if any(obj["instance_id"] in VISIBILITY_REVIEW_IDS and obj["instance_id"] not in visibility_confirmed
               for obj in base_objects):
            visual_cases = ["previous_visual_review_possible_rim_clipping_requires_confirmation"]
        if reviewed:
            unknown_regions = reviewed_regions
            visual_cases = []  # Explicit accepted reviewer accounts for these cases.
        else:
            # Preserve the original CVAT uncertain/truncated fields. A derived
            # visibility concern must nevertheless prevent that SAME object
            # from proving Q1=YES or Q2=YES until its eligibility is reviewed.
            objects = [{**obj, "reference_uncertainty": "possible_rim_clipping_prior_visual_review"}
                       if obj["instance_id"] in VISIBILITY_REVIEW_IDS and obj["instance_id"] not in visibility_confirmed
                       else obj for obj in objects]
        duplicate_pairs = overlap_pairs(objects, width, height) if not reviewed else []
        complete = complete_instances if reviewed else source_audit["source_instances_complete"]
        candidate_rows, geometry_rows = candidate_answers(objects, width, height, source_complete=complete,
            unknown_regions=unknown_regions or visual_cases, duplicate_review_pending=bool(duplicate_pairs),
            image_complete=decision["complete_for_size_rule"] if reviewed else annotation["complete_for_size_rule"])
        effective_result = decision.get("result", annotation["cvat_result"]) if reviewed else annotation["cvat_result"]
        if effective_result not in rules["cvat"]["result_values"]:
            raise ValueError("Invalid reviewed result value")
        if effective_result == "no_eligible_craters" and any(g["eligible"] for g in geometry_rows):
            raise ValueError("Negative review tag contradicts size-filtered instances: " + sid)
        group_overlap = (row["planet"], row["source_dataset"], row["scene_group_id"]) in train_groups
        for candidate in candidate_rows:
            task = candidate["task_id"]
            rid = f"{sid}__{task}__clean"
            candidate.update(sample_id=sid, request_id=rid, planet=row["planet"], source_split=row["source_split"],
                             scene_group_id=row["scene_group_id"], annotation_fingerprint=fingerprint,
                             paired_ibm_admitted=False, annotation_version=SCHEMA)
            if group_overlap:
                candidate.update(candidate_answer=None, reference_status="UNKNOWN_REFERENCE", admission_reason="scene_group_also_in_publisher_train")
            req = asdict(ModelRequest(rid, (row["image"],), prompt_for(task, width, height), tuple(candidate["allowed_answers"]), max_new_tokens=16))
            candidates.append(req)
            if reviewed and candidate["candidate_answer"] is not None:
                answer = candidate["candidate_answer"]
                candidate.update(reference_status="REFERENCE_READY", reference_answer=answer, scored=False,
                                 admission_reason="explicit_hash_bound_review_and_answer_invariance")
                # Standard manifest loaders resolve image paths relative to the
                # request file. Candidate inspection rows remain root-relative.
                requests.append({**req, "image_paths": [Path(os.path.relpath(orbital_path(root, row["image"]), output)).as_posix()]})
                targets.append(asdict(GroundTruth(rid, answer, sid,
                    f"{row['planet']}:{row['source_dataset']}:{row['scene_group_id']}", VERSION, task, "clean")))
            elif candidate["candidate_answer"] is not None:
                candidate["admission_reason"] = "second_instance_identity_and_geometry_review_pending"
            questions.append(candidate)
            provenance.append({"request_id": rid, **{k: row[k] for k in ("sample_id", "planet", "source_dataset", "source_revision", "source_split",
                               "source_image_id", "scene_group_id", "license_claim", "image_sha256", "mask_sha256", "annotation_image_sha256", "final_test_eligible")},
                               "task_id": task, "annotation_fingerprint": fingerprint, "condition": "clean"})
        for obj in objects:
            instances.append({"sample_id": sid, **obj})
        catalog.extend({"sample_id": sid, **obj} for obj in unresolved)
        template = {"sample_id": sid, "annotation_fingerprint": fingerprint, "review_status": "pending", "annotator": "",
                    "reviewer": "", "reviewed_at": "", "review_basis": "", "result": annotation["cvat_result"],
                    "complete_for_size_rule": annotation["complete_for_size_rule"],
                    "source_instances_complete": complete, "duplicates_reviewed": False, "uncertainty_accounted_for": False,
                    "unknown_regions": unknown_regions, "excluded_instances": [], "instance_overrides": [], "additional_instances": [],
                    "duplicate_candidates": duplicate_pairs, "visual_cases": visual_cases,
                    "boundary_review_instance_ids": [g["instance_id"] for g in geometry_rows if g["eligible"] and g["boundary_review"]],
                    "source_geometry_status": source_audit["source_geometry_status"], "preview": f"review_previews/{sid}.png" if previews else None}
        review_queue.append(template)
        samples.append({**row, "native_size": [width,height], "annotation_version": SCHEMA,
                        "annotation_fingerprint": fingerprint, "source_annotation_audit": source_audit,
                        "cvat_review": {k:v for k,v in annotation.items() if k != "supplemental_instances"},
                        "source_instance_count": len(source), "supplemental_instance_count": len(supplement),
                        "unresolved_catalog_count": len(unresolved), "review_status": "accepted" if reviewed else "pending_second_review",
                        "annotator": decision["annotator"] if reviewed else "CVAT_export_user",
                        "reviewer": decision["reviewer"] if reviewed else None,
                        "reviewed_at": decision["reviewed_at"] if reviewed else None,
                        "unknown_regions": unknown_regions, "duplicate_candidates": duplicate_pairs})
        if previews:
            with Image.open(orbital_path(root, row["image"])) as original:
                original = original.convert("RGB")
                labelled = original.copy()
                draw = ImageDraw.Draw(labelled)
                for i, obj in enumerate(objects, 1):
                    color = "orange" if obj["uncertain"] else "cyan" if obj["origin"] == "human_CVAT_supplement" else "lime"
                    draw.rectangle(obj["bbox_xyxy"], outline=color, width=1)
                    draw.text((max(0,obj["bbox_xyxy"][0]), max(0,obj["bbox_xyxy"][1])), str(i), fill=color)
                panel = Image.new("RGB", (2*width,height+24), (20,24,32))
                panel.paste(original, (0,24)); panel.paste(labelled, (width,24))
                ImageDraw.Draw(panel).text((3,3), sid+" | source green / CVAT cyan / uncertain orange | NOT accepted GT", fill="white")
                panels.append((sid,panel))
    # Recheck every immutable input before publishing this exclusive build.
    for row in rows:
        for key in ("image", "mask"):
            if file_hash(orbital_path(root,row[key])) != row[key+"_sha256"]:
                raise ValueError("Source changed during annotation assembly")
    for name, expected in manifest_hashes.items():
        if file_hash(orbital_path(root,name)) != expected: raise ValueError("Source manifest changed during build")
    output.mkdir(parents=True, exist_ok=False)
    code_files = (__file__, __import__("planetary_vlm.datasets.crater_reference", fromlist=["__file__"]).__file__,
                  __import__("planetary_vlm.datasets.crater_sources", fromlist=["__file__"]).__file__,
                  __import__("planetary_vlm.datasets.mars_catalog_geometry", fromlist=["__file__"]).__file__,
                  __import__("planetary_vlm.datasets.image_replacement", fromlist=["__file__"]).__file__)
    (output / "converter_snapshot").mkdir()
    for code in code_files:
        with (output / "converter_snapshot" / Path(code).name).open("xb") as stream:
            stream.write(Path(code).read_bytes())
    (output / "exports").mkdir()
    for name, raw in snapshots.items():
        with (output / "exports" / name).open("xb") as stream: stream.write(raw)
    if previews:
        (output / "review_previews").mkdir()
        for sid, panel in panels: panel.save(output / "review_previews" / (sid+".png"))
    payloads = {"samples.jsonl": samples, "instances.jsonl": instances, "source_catalog_pending.jsonl": catalog,
                "questions.jsonl": questions, "candidate_requests.jsonl": candidates,
                "requests.jsonl": requests, "targets.jsonl": targets, "provenance.jsonl": provenance,
                "review_queue.jsonl": review_queue}
    for name, records in payloads.items(): write_jsonl(output/name, records)
    if context_data is not None: write_json(output/"review_context.json", context_data)
    summary = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(), "rules_version": rules["schema_version"],
        "rules_sha256": file_hash(rules_path), "source_manifests_sha256": manifest_hashes, "exports": export_records,
        "source_annotation_cache": cache.relative_to(root).as_posix(), "source_annotation_provenance": reader.provenance,
        "converter_sha256": {Path(p).name: file_hash(output / "converter_snapshot" / Path(p).name) for p in code_files},
        "input_reviews_sha256": review_hash, "images": len(samples), "instances_with_geometry": len(instances),
        "collection": collection.relative_to(root).as_posix(),
        "contextual_visibility_confirmations": len(visibility_confirmed),
        "supplemental_instances": sum(s["supplemental_instance_count"] for s in samples), "catalog_geometry_pending": len(catalog),
        "source_geometry_pending_images": sum(not s["source_annotation_audit"]["source_instances_complete"] for s in samples),
        "candidate_questions": len(questions), "invariant_candidate_answers": sum(q["candidate_answer"] is not None for q in questions),
        "unknown_references": sum(q["reference_status"] == "UNKNOWN_REFERENCE" for q in questions),
        "reference_ready_questions": len(targets), "accepted_image_reviews": sum(s["review_status"] == "accepted" for s in samples),
        "by_planet_task_split_status": dict(Counter(f"{q['planet']}/{q['task_id']}/{q['source_split']}/{q['reference_status']}" for q in questions)),
        "files_sha256": {name: file_hash(output/name) for name in
                         [*payloads, *(["review_context.json"] if context_data is not None else [])]},
        "scientific_result": False, "inference_run": False, "paired_ibm_admitted": False,
        "limitations": ["CANDIDATE_REFERENCE is not accepted GT; explicit second review is required before targets are emitted.",
                        "UNKNOWN_REFERENCE is unscored, not a model answer or negative label.",
                        "Mars raster reconstruction is reverse-engineered on source-val, not a publisher-documented physical projection; each frame must reproduce its source class mask exactly.",
                        "Exact raster replay verifies source alignment, not visual rim accuracy, catalog completeness, or accepted GT.",
                        "IoU flags request review, not duplicate removal; nested craters remain distinct unless explicitly adjudicated.",
                        "CVAT ZIPs contain XML only; uploaded image bytes cannot be checked from exports.",
                        "Publisher splits/groups are retained, not a proof of model-training independence."]}
    summary["bundle_sha256"] = digest(summary)
    write_json(output/"metadata.json", summary)
    return summary


def verify_crater_build(root, folder):
    """Technical integrity check only. No auto-review, network or model imports."""
    root = Path(root).resolve()
    folder = Path(folder)
    if not folder.is_absolute(): folder = root / folder
    folder = orbital_path(root, folder.resolve().relative_to(root).as_posix())
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    claimed = metadata.pop("bundle_sha256")
    if metadata.get("schema") != SCHEMA or digest(metadata) != claimed:
        raise ValueError("Annotation metadata changed")
    def local(name):
        path = orbital_path(root, (folder.relative_to(root) / name).as_posix())
        if not path.is_relative_to(folder): raise ValueError("Build path escaped version folder")
        return path
    for name, checksum in metadata["files_sha256"].items():
        if file_hash(local(name)) != checksum: raise ValueError("Annotation build file changed: " + name)
    export_storage = {}
    for name, record in metadata["exports"].items():
        archive = local("exports/"+name)
        if archive.exists():
            if file_hash(archive) != record["archive_sha256"]:
                raise ValueError("Snapshotted CVAT export changed")
            export_storage[name] = "original_zip_verified"
        else:
            # Cleanup may unpack the XML without changing any sealed metadata.
            # This verifies the original XML hash, NOT the absent ZIP bytes.
            xml = local("exports/" + Path(name).stem + "/annotations.xml")
            if file_hash(xml) != record.get("xml_sha256"):
                raise ValueError("Snapshotted CVAT XML changed")
            export_storage[name] = "original_xml_verified_zip_bytes_not_available"
    for name, checksum in metadata["converter_sha256"].items():
        if file_hash(local("converter_snapshot/"+name)) != checksum:
            raise ValueError("Snapshotted converter changed")
    for name, checksum in metadata["source_manifests_sha256"].items():
        if file_hash(orbital_path(root,name)) != checksum: raise ValueError("Source manifest changed")
    samples = {s["sample_id"]: s for s in read_jsonl(local("samples.jsonl"))}
    if len(samples) != metadata["images"]: raise ValueError("Inconsistent sample count")
    for row in samples.values():
        for key in ("image", "mask"):
            if file_hash(orbital_path(root,row[key])) != row[key+"_sha256"]:
                raise ValueError("Immutable original asset changed")
        if row.get("image_replacement_record"):
            from planetary_vlm.datasets.image_replacement import verify_replacement
            if digest(verify_replacement(root,row)) != row.get("image_replacement_sha256"):
                raise ValueError("Image replacement recipe changed after reference preparation")
    questions = {q["request_id"]: q for q in read_jsonl(local("questions.jsonl"))}
    if len(questions) != metadata["candidate_questions"]: raise ValueError("Inconsistent or duplicate question IDs")
    for record in read_jsonl(local("candidate_requests.jsonl")):
        request = request_from_dict(record)
        question = questions[request.request_id]
        sample = samples[question["sample_id"]]
        if (request.prompt != prompt_for(question["task_id"],*sample["native_size"]) or request.do_sample
                or request.allowed_answers != tuple(question["allowed_answers"]) or request.modality_paths
                or request.max_new_tokens != 16 or request.image_paths != (sample["image"],)):
            raise ValueError("Candidate model-input contract changed")
    ready = {rid for rid,q in questions.items() if q["reference_status"] == "REFERENCE_READY"}
    requests = read_jsonl(local("requests.jsonl"))
    targets = read_jsonl(local("targets.jsonl"))
    for records in (requests,targets):
        ids = [r["request_id"] for r in records]
        if len(ids) != len(set(ids)) or set(ids) != ready: raise ValueError("Ready request/target coverage mismatch")
    for record in requests:
        request = request_from_dict(record)
        question = questions[request.request_id]
        sample = samples[question["sample_id"]]
        if len(request.image_paths) != 1 or request.modality_paths or request.do_sample or request.max_new_tokens != 16:
            raise ValueError("Reference masks/extra modalities or decoding edits in request")
        actual = (folder / request.image_paths[0]).resolve()
        if actual != orbital_path(root,sample["image"]) or request.prompt != prompt_for(question["task_id"],*sample["native_size"]) or request.allowed_answers != tuple(question["allowed_answers"]):
            raise ValueError("Model-visible request does not match audited original/prompt")
    for target in targets:
        question = questions[target["request_id"]]
        if target["answer"] != question["reference_answer"] or target["answer"] not in question["allowed_answers"]:
            raise ValueError("Target does not match admitted reference")
    if len(targets) != metadata["reference_ready_questions"]:
        raise ValueError("Inconsistent admitted-target count")
    return {"schema": SCHEMA, "integrity_verified": True, "bundle_sha256": claimed, "images": len(samples),
            "export_storage": export_storage,
            "candidate_questions": len(questions), "reference_ready_questions": len(targets),
            "ready_for_scored_inference": bool(targets), "scientific_result": False}


def load_crater_requests(root, folder):
    """Return all invariant candidate inputs for UNSCORED inference, without GT.

    Candidate completeness is not scoring admission. Final accepted targets stay
    local and are deliberately not required or emitted by this input loader.
    """
    root = Path(root).resolve()
    folder = Path(folder)
    if not folder.is_absolute(): folder = root / folder
    folder = folder.resolve()
    verify_crater_build(root,folder)
    record = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    questions = read_jsonl(folder/"questions.jsonl")
    if (not questions or len(questions) != 4*record["images"]
            or any(q["candidate_answer"] is None for q in questions)):
        raise ValueError("Complete invariant Q1-Q4 candidates required; unresolved references remain")
    inputs = [request_from_dict(r) for r in read_jsonl(folder/"candidate_requests.jsonl")]
    if len(inputs) != len(questions): raise ValueError("Candidate input count mismatch")
    return inputs, {**record, "question_type":"candidate_Q1_Q4_clean_unscored",
                    "rules":{"version":record["rules_version"],"sha256":record["rules_sha256"]}}
