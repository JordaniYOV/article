"""Validate saved records, join GT only here, and return a JSON-safe report."""

import math
from collections import Counter, defaultdict

from planetary_vlm.io import records_fingerprint

from .metrics import classification_metrics
from .parser import parse_answer, validate_answers

REQUEST_REQUIRED = {"request_id", "image_paths", "prompt", "allowed_answers", "max_new_tokens", "do_sample"}
TARGET_REQUIRED = {"request_id", "answer", "source_sample_id", "group_id", "derivation_version"}
PREDICTION_REQUIRED = {"request_id", "raw_response", "status", "elapsed_seconds"}


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _index(records: list[dict], kind: str, required: set, optional: set) -> dict:
    if not isinstance(records, list):
        raise ValueError(f"{kind} must be a list of records")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{kind} records must be objects")
        if not required <= record.keys():
            raise ValueError(f"{kind} missing required fields: {sorted(required - record.keys())}")
        if record.keys() - required - optional:
            raise ValueError(f"{kind} has unknown/forbidden fields: {sorted(record.keys() - required - optional)}")
        _text(record["request_id"], "request_id")
        if record["request_id"] in result:
            raise ValueError(f"duplicate {kind} request_id: {record['request_id']}")
        result[record["request_id"]] = record
    return result


def _validate_request(request: dict) -> None:
    _text(request["prompt"], "prompt")
    if not isinstance(request["image_paths"], (list, tuple)):
        raise ValueError("image_paths must be a list or tuple")
    for path in request["image_paths"]:
        _text(path, "image_path")
    validate_answers(request["allowed_answers"])
    if type(request["max_new_tokens"]) is not int or request["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(request["do_sample"]) is not bool:
        raise ValueError("do_sample must be a boolean")
    modalities = request.get("modality_paths", [])
    if not isinstance(modalities, (list, tuple)):
        raise ValueError("modality_paths must be a list of name/path pairs")
    names = set()
    for item in modalities:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("modality_paths must contain name/path pairs")
        _text(item[0], "modality name")
        _text(item[1], "modality path")
        if item[0] in names:
            raise ValueError("duplicate modality name")
        names.add(item[0])


def _validate_prediction(prediction: dict) -> None:
    if not isinstance(prediction["raw_response"], str):
        raise ValueError("raw_response must be a string")
    if prediction["status"] not in ("ok", "error", "unsupported"):
        raise ValueError("prediction status must be ok, error, or unsupported")
    elapsed = prediction["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_seconds must be finite and nonnegative")
    if prediction.get("error_message") is not None and not isinstance(prediction["error_message"], str):
        raise ValueError("error_message must be a string or null")
    for key in ("model_id", "run_id"):
        if key in prediction:
            _text(prediction[key], key)
    if "mock" in prediction and type(prediction["mock"]) is not bool:
        raise ValueError("mock must be a boolean")
    if "request_manifest_sha256" in prediction:
        fingerprint = prediction["request_manifest_sha256"]
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)):
            raise ValueError("request_manifest_sha256 must be 64 lowercase hexadecimal characters")


def _summarize(rows: list[dict], *, same_task: bool) -> dict:
    scored = [row for row in rows if row["scoring_status"] != "unsupported_scoring"]
    states = Counter(row["scoring_status"] for row in scored)
    correct = sum(row["correct"] for row in scored)
    result = {
        "n_total": len(rows), "n_scored": len(scored),
        "n_unsupported_scoring": len(rows) - len(scored),
        "n_correct": correct, "n_valid": states["valid"],
        "n_invalid": states["invalid"], "n_failure": states["failure"],
        "response_status_counts": dict(Counter(row["response_status"] for row in rows)),
        "accuracy": correct / len(scored) if scored else None,
        "accuracy_denominator": len(scored),
        "valid_only_accuracy": correct / states["valid"] if states["valid"] else None,
        "coverage": states["valid"] / len(scored) if scored else None,
        "scoring_coverage": len(scored) / len(rows) if rows else None,
        "per_class": None, "macro_f1": None, "macro_f1_classes": [],
        "confusion_matrix": None, "binary": None,
    }
    label_sets = {tuple(sorted(row["allowed_answers"])) for row in scored}
    if not scored:
        result["classification_status"] = "unsupported_scoring" if rows else "empty"
    elif len(label_sets) != 1:
        result["classification_status"] = "incompatible_label_sets"
    elif not same_task:
        result["classification_status"] = "multiple_tasks_use_slices"
    else:
        result["classification_status"] = "scored"
        result.update(classification_metrics(scored, list(next(iter(label_sets)))))
    return result


def evaluate(requests: list[dict], targets: list[dict], predictions: list[dict]) -> dict:
    """Evaluate one model/run, rejecting mismatched manifests instead of dropping rows.

    Free-form requests are recorded as unsupported_scoring. A supplied request
    fingerprint is checked, but it does not hash asset contents or ground truth.
    """
    req = _index(requests, "requests", REQUEST_REQUIRED, {"modality_paths"})
    gt = _index(targets, "targets", TARGET_REQUIRED, {"task_id", "condition_id"})
    pred = _index(predictions, "predictions", PREDICTION_REQUIRED,
                  {"error_message", "model_id", "run_id", "mock", "request_manifest_sha256"})
    if req.keys() != gt.keys():
        raise ValueError("targets must exactly match requests (no missing or stale IDs)")
    if pred.keys() - req.keys():
        raise ValueError("predictions contain unknown request IDs")
    for request in req.values():
        _validate_request(request)
    for prediction in pred.values():
        _validate_prediction(prediction)
    metadata = {}
    for key in ("model_id", "run_id", "mock", "request_manifest_sha256"):
        values = {record.get(key) for record in pred.values()}
        if len(values) > 1:
            raise ValueError(f"predictions must contain one consistent {key}")
        metadata[key] = next(iter(values)) if values else None
    fingerprint = metadata["request_manifest_sha256"]
    if fingerprint is not None and fingerprint != records_fingerprint(requests):
        raise ValueError("Prediction request_manifest_sha256 does not match current requests")
    rows = []
    for key, request in req.items():
        target = gt[key]
        for field in TARGET_REQUIRED | {"task_id", "condition_id"}:
            if field in target:
                _text(target[field], f"target {field}")
        allowed = request["allowed_answers"]
        answer = parse_answer(target["answer"], allowed) if allowed else target["answer"]
        if allowed and answer is None:
            raise ValueError(f"target answer is outside allowed_answers: {key}")
        prediction = pred.get(key)
        response_status = prediction["status"] if prediction else "missing"
        parsed = None
        if not allowed:
            scoring_status = "unsupported_scoring"
        elif response_status != "ok":
            scoring_status = "failure"
        else:
            parsed = parse_answer(prediction["raw_response"], allowed)
            scoring_status = "valid" if parsed is not None else "invalid"
        rows.append({
            "request_id": key, "task_id": target.get("task_id", "default"),
            "condition_id": target.get("condition_id", "clean"),
            "source_sample_id": target["source_sample_id"], "group_id": target["group_id"],
            "derivation_version": target["derivation_version"],
            "allowed_answers": list(allowed), "answer": answer, "parsed_answer": parsed,
            "scoring_status": scoring_status, "response_status": response_status,
            "correct": (parsed == answer if allowed else None),
            "raw_response": prediction["raw_response"] if prediction else None,
            "error_message": prediction.get("error_message") if prediction else "missing prediction",
            "elapsed_seconds": prediction["elapsed_seconds"] if prediction else None,
            "mock": prediction.get("mock") if prediction else metadata["mock"],
        })
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task_id"], row["condition_id"])].append(row)
    slices = [{"task_id": key[0], "condition_id": key[1],
               "metrics": _summarize(group, same_task=True)}
              for key, group in sorted(groups.items())]
    return {
        "schema_version": "evaluation-v1", **metadata,
        "request_manifest_verified": fingerprint is not None,
        "parser": "exact_casefold_collapsed_whitespace_v1",
        "overall": _summarize(rows, same_task=len({row["task_id"] for row in rows}) <= 1),
        "slices": slices, "per_request": rows,
        "limitations": [
            "Mock results test the pipeline only and are not scientific model results.",
            "No free-form judge; empty allowed_answers are unsupported_scoring.",
            "Overall accuracy is question-weighted over closed-form requests, not a cross-task scientific ranking.",
            "Optional request fingerprints bind manifest fields, not asset contents, GT history, or annotation completeness.",
            "Binary rates apply to YES/NO labels and valid answers; inspect coverage and classwise failures.",
        ],
    }
