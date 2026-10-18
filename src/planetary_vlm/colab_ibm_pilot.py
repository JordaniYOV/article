"""Bounded, frozen val-only IBM pilot; shares the original detector and runner.

Inference receives RGB/intensity only. Local reference files are used exclusively
by the separate evaluate command after inference; candidates stay provisional.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from io import BytesIO
from pathlib import Path, PurePosixPath
import zipfile

from planetary_vlm import colab_ibm as ibm
from planetary_vlm.colab import digest, verify_bundle
from planetary_vlm.datasets.manifest import load_requests
from planetary_vlm.evaluation.report import evaluate
from planetary_vlm.io import read_jsonl, write_json
from benchmark_services.ibm.crater import readout

SCHEMA = "ibm-val-pilot-v1"


def cohort(root):
    manifest = verify_bundle(root)
    value = manifest["protocol"].get("inference_cohort", {})
    if (set(value) != {"schema", "selection_version", "pilot_sample_ids", "smoke_sample_ids"}
        or value.get("schema") != SCHEMA or value.get("selection_version") != "ibm-val14-source-group-v1"):
        raise ValueError("A sealed IBM val14 inference cohort is required")
    pilot, smoke = value["pilot_sample_ids"], value["smoke_sample_ids"]
    if (not isinstance(pilot, list) or not isinstance(smoke, list)
        or any(not isinstance(s, str) for s in pilot+smoke)
        or len(pilot) != 14 or len(set(pilot)) != 14 or set(smoke) != set(ibm.SMOKE_IDS)
        or len(smoke) != 2 or set(pilot) & set(smoke)):
        raise ValueError("Expected 14 distinct pilot samples and two separate frozen smoke probes")
    images = ibm.select_inputs(root)
    if {r["sample_id"] for r in images} != set(pilot+smoke):
        raise ValueError("Input bundle differs from its frozen cohort")
    groups = set()
    for image in images:
        row = image["provenance"]
        if row["source_split"] != "val":
            raise ValueError("Pilot must never include publisher test/train")
        key = (row["planet"], row["source_dataset"], row["scene_group_id"])
        if key in groups:
            raise ValueError("Related scene groups must not repeat, including smoke probes")
        groups.add(key)
    counts = {planet: sum(r["sample_id"] in pilot and r["planet"] == planet for r in images)
        for planet in ("moon", "mars")}
    if counts != {"moon": 7, "mars": 7}:
        raise ValueError("Expected a 7 Moon + 7 Mars development pilot")
    return manifest, value


def run_pilot(root, persist, run_id, session_id, *, smoke=False, resume=False):
    _, selection = cohort(root)
    return ibm.run_job(root, persist, run_id, session_id, smoke=smoke, resume=resume,
        sample_ids=None if smoke else selection["pilot_sample_ids"])


def score_results(results, reference_dir, output, *, provisional=False, scope="val14"):
    """Validate saved images/detections; use the common scorer, never rerun IBM."""
    scopes = {"val14": ("ibm-val14-local-references-v1", "pilot_sample_ids", 14,
                          "val14_candidate_annotation_pilot", 150, 64*2**20),
              "full100": ("ibm-full100-local-references-v2", "sample_ids", 100,
                            "full100_candidate_annotation_clean", 1000, 256*2**20)}
    if scope not in scopes:
        raise ValueError("Unknown frozen IBM evaluation scope")
    schema, ids_key, image_count, evaluation_scope, max_entries, max_bytes = scopes[scope]
    reference_dir, output = Path(reference_dir).resolve(), Path(output).resolve()
    if any("data_rover" in {p.casefold() for p in path.parts} for path in (reference_dir, output, Path(results).resolve())):
        raise ValueError("Protected archive is outside the pilot")
    if output.exists():
        raise ValueError("Use a new evaluation output file")
    seal = json.loads((reference_dir/"reference_manifest.json").read_text(encoding="utf-8"))
    claimed = seal.pop("manifest_sha256", None)
    if (seal.get("schema") != schema or digest(seal) != claimed
        or set(seal.get("files_sha256", {})) != {"requests.jsonl", "candidate_targets.jsonl", "source_candidates.jsonl"}
        or len(seal.get(ids_key, [])) != image_count or len(set(seal[ids_key])) != image_count
        or set(seal.get("image_sha256", {})) != set(seal[ids_key])):
        raise ValueError("Local reference manifest differs from its seal/schema")
    for name, expected in seal["files_sha256"].items():
        if name not in {"requests.jsonl", "candidate_targets.jsonl", "source_candidates.jsonl"}:
            raise ValueError("Unknown reference manifest file")
        if sha256((reference_dir/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Local reference file changed")
    requests = [asdict(r) for r in load_requests(reference_dir/"requests.jsonl")]
    targets = read_jsonl(reference_dir/"candidate_targets.jsonl")
    if (len(requests) != 4*image_count or len(targets) != 4*image_count
        or len({r["request_id"] for r in requests}) != 4*image_count
        or {r["request_id"] for r in requests} != {r["request_id"] for r in targets}):
        raise ValueError(f"Expected {4*image_count} matching frozen local requests/targets")
    original_targets = {row["request_id"]: row for row in targets}
    if any(row.get("reference_status") != "CANDIDATE_REFERENCE" or row.get("answer") is None for row in targets):
        raise ValueError("Expected the original known candidate targets; never synthesize negatives")
    with zipfile.ZipFile(results) as archive:
        entries = archive.infolist()
        if (len(entries) > max_entries or len({e.filename for e in entries}) != len(entries)
            or sum(e.file_size for e in entries) > max_bytes):
            raise ValueError("Unexpected result archive size/duplicates")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename or ":" in entry.filename
                or any(p.casefold() == "data_rover" for p in path.parts)):
                raise ValueError("Unsafe result ZIP path")
        runs = [e.filename for e in entries if e.filename.endswith("/run.json")]
        if len(runs) != 1 or archive.testzip() is not None:
            raise ValueError("Expected exactly one intact run")
        prefix = runs[0][:-len("run.json")]
        run = json.loads(archive.read(prefix+"run.json"))
        if (run.get("smoke") is not False or run.get("bundle_sha256") != seal["input_bundle_sha256"]
            or len(run.get("sample_ids", [])) != image_count
            or set(run.get("sample_ids", [])) != set(seal[ids_key])):
            raise ValueError("Not this frozen pilot run; smoke/full100/foreign bundles are refused")
        if (seal.get("environment_pip_freeze_sha256") is not None
            and run.get("environment", {}).get("pip_freeze_sha256") != seal["environment_pip_freeze_sha256"]):
            raise ValueError("Result environment differs from the frozen reference protocol")
        if any(run.get(key) != expected for key, expected in (
            ("protocol", ibm.PROTOCOL), ("code_revision", ibm.CODE_REVISION),
            ("head_revision", ibm.HEAD_REVISION), ("backbone_revision", ibm.BACKBONE_REVISION),
            ("score_threshold", ibm.SCORE_THRESHOLD))):
            raise ValueError("Pilot model or inference protocol changed")
        predictions = [json.loads(line) for line in archive.read(prefix+"predictions.jsonl").splitlines() if line.strip()]
        pred_index = {r["request_id"]: r for r in predictions}
        if len(pred_index) != len(predictions) or pred_index.keys()-original_targets.keys():
            raise ValueError("Duplicate/foreign prediction IDs")
        journal = json.loads(archive.read(prefix+"progress.json"))["detections_sha256"]
        if not set(journal) <= set(seal[ids_key]):
            raise ValueError("Foreign completed image")
        completed_ids = set()
        for sid, expected in journal.items():
            raw = archive.read(prefix+"detections/"+sid+".json")
            record = json.loads(raw)
            if sha256(raw).hexdigest() != expected:
                raise ValueError("Saved detection JSON changed")
            image = archive.read(prefix+"inputs/"+sid+".png")
            if sha256(image).hexdigest() != seal["image_sha256"][sid] or record["image_sha256"] != seal["image_sha256"][sid]:
                raise ValueError("Result image differs from its annotation-bound source")
            if record["sample_id"] != sid:
                raise ValueError("Detection/sample identity differs")
            # The frozen collection includes 8-bit TIFF bytes under .png names.
            # Decode the hash-bound original; do not transcode or change GT hashes.
            from PIL import Image
            with Image.open(BytesIO(image)) as decoded:
                if (decoded.format not in {"PNG", "TIFF"} or decoded.mode not in {"L", "RGB"}
                    or decoded.width*decoded.height > 4*2**20 or getattr(decoded, "n_frames", 1) != 1):
                    raise ValueError("Expected a single 8-bit PNG/TIFF source image")
                decoded.load()
                width, height = decoded.size
            if (record["preprocessing"]["native_width"], record["preprocessing"]["native_height"]) != (width,height):
                raise ValueError("Detection coordinates refer to different native dimensions")
            replay = readout([row["bbox_xyxy"] for row in record["detections"]],
                [row["score"] for row in record["detections"]], [row["label"] for row in record["detections"]],
                width, height)
            if replay["answers"] != record["answers"]:
                raise ValueError("Saved answers differ from the fixed detector readout")
            for task in ibm.TASKS:
                rid = sid+"__"+task+"__clean"
                if rid not in pred_index or pred_index[rid]["raw_response"] != record["answers"][task]:
                    raise ValueError("Responses differ from saved detections")
                completed_ids.add(rid)
        if pred_index.keys() != completed_ids:
            raise ValueError("Predictions and completed detections disagree")
    if scope == "full100":
        # Keep descriptive metadata out of the strict shared target contract.
        descriptive_fields = {"planet", "source_dataset", "source_split", "prior_ibm_development_scene",
            "prior_ibm_development_group", "cross_source_split_group"}
        targets = [{key:value for key,value in row.items() if key not in descriptive_fields} for row in targets]
    if provisional:
        targets = [{**row, "reference_status": "REFERENCE_READY"} for row in targets]
    report = evaluate(requests, targets, predictions)
    report.update(evaluation_scope=evaluation_scope, provisional_scoring=provisional,
        source_reference_status="CANDIDATE_REFERENCE", scientific_result=False,
        complete=len(journal) == image_count, completed_images=len(journal), requested_images=image_count,
        run_id=run["run_id"], source_result_zip_sha256=sha256(Path(results).read_bytes()).hexdigest())
    report["limitations"].insert(0, "Known candidate annotations are not accepted scientific GT. "
        + ("User explicitly selected provisional scoring." if provisional else "No candidate references were admitted for scoring."))
    for row in report["per_request"]:
        row["source_reference_status"] = "CANDIDATE_REFERENCE"
        if scope == "full100":
            target = original_targets[row["request_id"]]
            row.update({key: target[key] for key in ("planet", "source_dataset", "source_split",
                "prior_ibm_development_scene", "prior_ibm_development_group", "cross_source_split_group") if key in target})
    if scope == "full100":
        # Descriptive slices, never a pooled architecture ranking or new holdout.
        slices = {}
        for row in report["per_request"]:
            key = "/".join(str(row.get(k, "unknown")) for k in ("planet", "source_split", "task_id"))
            item = slices.setdefault(key, {"n_total":0,"n_scored":0,"n_correct":0})
            item["n_total"] += 1
            if row.get("correct") is not None:
                item["n_scored"] += 1
                item["n_correct"] += int(row["correct"])
        for item in slices.values():
            item["accuracy"] = item["n_correct"]/item["n_scored"] if item["n_scored"] else None
        report["descriptive_planet_split_task_slices"] = slices
        report["limitations"].insert(0, "Full pool includes previously viewed development scenes; original val/test labels are retained, not proof of a new untouched holdout.")
    write_json(output, report)
    return {"output": str(output), "complete": report["complete"], "overall": report["overall"],
        "provisional_scoring": provisional, "scientific_result": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "environment", "assets", "smoke", "pilot", "evaluate"))
    parser.add_argument("--workspace")
    parser.add_argument("--persist")
    parser.add_argument("--run-id", default="ibm_val14_v1_clean")
    parser.add_argument("--session-id")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results")
    parser.add_argument("--reference-dir")
    parser.add_argument("--output")
    parser.add_argument("--provisional", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if not all((args.results, args.reference_dir, args.output)):
            parser.error("evaluate needs --results, --reference-dir and --output")
        result = score_results(args.results, args.reference_dir, args.output, provisional=args.provisional)
    else:
        if not args.workspace:
            parser.error("Inference operations need --workspace")
        manifest, selection = cohort(args.workspace)
        if args.command == "verify":
            result = {"pilot_images": len(selection["pilot_sample_ids"]), "pilot_answers": 56,
                "separate_smoke_images": 2, "bundle_sha256": manifest["bundle_sha256"], "all_source_split": "val"}
        elif args.command == "assets":
            result = ibm.acquire_assets(args.workspace, download=args.download)
        elif args.command == "environment":
            result = ibm.environment(args.workspace)
        else:
            if not args.persist or not args.session_id:
                parser.error("smoke/pilot needs --persist and --session-id")
            result = run_pilot(args.workspace, args.persist, args.run_id, args.session_id,
                smoke=args.command == "smoke", resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
