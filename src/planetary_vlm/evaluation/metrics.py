"""Standard-library counts and explicit denominator policies."""

import math
import random
from collections import Counter, defaultdict

from .parser import normalize_label


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def classification_metrics(rows: list[dict], labels: list[str]) -> dict:
    """Score already-validated closed-form rows, retaining invalid and failures."""
    total = len(rows)
    counts = Counter(row["scoring_status"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    per_class = {}
    for label in labels:
        support = sum(row["answer"] == label for row in rows)
        tp = sum(row["answer"] == label and row["parsed_answer"] == label for row in rows)
        fp = sum(row["answer"] != label and row["parsed_answer"] == label for row in rows)
        fn = support - tp
        per_class[label] = {
            "support": support, "tp": tp, "fp": fp, "fn": fn,
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        }
    supported = [label for label in labels if per_class[label]["support"]]
    columns = [{"kind": "label", "value": label} for label in labels]
    columns += [{"kind": "status", "value": "invalid"}, {"kind": "status", "value": "failure"}]
    matrix = []
    for label in labels:
        selected = [row for row in rows if row["answer"] == label]
        matrix.append([sum(row["parsed_answer"] == predicted for row in selected)
                       for predicted in labels] +
                      [sum(row["scoring_status"] == status for row in selected)
                       for status in ("invalid", "failure")])
    result = {
        "n_scored": total, "n_correct": correct,
        "n_valid": counts["valid"], "n_invalid": counts["invalid"], "n_failure": counts["failure"],
        "accuracy": _ratio(correct, total),
        "valid_only_accuracy": _ratio(correct, counts["valid"]),
        "coverage": _ratio(counts["valid"], total),
        "per_class": per_class, "macro_f1_classes": supported,
        "macro_f1": (sum(per_class[label]["f1"] for label in supported) /
                     len(supported) if supported else None),
        "confusion_matrix": {"rows": labels, "columns": columns, "counts": matrix},
        "binary": None,
    }
    # A generic two-class task has no intrinsic positive class: never choose
    # alphabetically. YES/NO is the current explicit benchmark convention.
    normalized = {normalize_label(label): label for label in labels}
    if set(normalized) == {"yes", "no"}:
        positive, negative = normalized["yes"], normalized["no"]
        valid = [row for row in rows if row["scoring_status"] == "valid"]
        tp = sum(row["answer"] == positive and row["parsed_answer"] == positive for row in valid)
        fn = sum(row["answer"] == positive and row["parsed_answer"] == negative for row in valid)
        fp = sum(row["answer"] == negative and row["parsed_answer"] == positive for row in valid)
        tn = sum(row["answer"] == negative and row["parsed_answer"] == negative for row in valid)
        by_truth = {}
        for name, label in (("positive", positive), ("negative", negative)):
            selected = [row for row in rows if row["answer"] == label]
            states = Counter(row["scoring_status"] for row in selected)
            by_truth[name] = {"n_total": len(selected), "n_valid": states["valid"],
                              "n_invalid": states["invalid"], "n_failure": states["failure"],
                              "coverage": _ratio(states["valid"], len(selected))}
        result["binary"] = {
            "positive_label": positive, "negative_label": negative,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fpr": _ratio(fp, fp + tn), "fpr_denominator": fp + tn,
            "fnr": _ratio(fn, fn + tp), "fnr_denominator": fn + tp,
            "by_truth": by_truth,
            "positive_miss_rate_including_failures":
                _ratio(by_truth["positive"]["n_total"] - tp, by_truth["positive"]["n_total"]),
        }
    return result


def paired_accuracy_bootstrap(left_rows: list[dict], right_rows: list[dict], *,
                              seed: int = 0, repetitions: int = 1000) -> dict:
    """Right-minus-left accuracy on identical requests clustered by group_id.

    Input: per_request rows from evaluate. Condition contrasts with different
    request IDs and bootstrap intervals for other metrics are not implemented.
    """
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    def index(rows: list[dict]) -> dict:
        indexed = {}
        for row in rows:
            key = row.get("request_id")
            if not isinstance(key, str) or not key or key in indexed:
                raise ValueError("bootstrap requires unique nonempty request IDs")
            if (not isinstance(row.get("correct"), bool)
                    or not isinstance(row.get("group_id"), str) or not row["group_id"]):
                raise ValueError("bootstrap requires scored rows and nonempty group IDs")
            indexed[key] = row
        return indexed

    left, right = index(left_rows), index(right_rows)
    if not left or left.keys() != right.keys():
        raise ValueError("paired bootstrap requires the same nonempty request set")
    grouped = defaultdict(list)
    for key in sorted(left):
        a, b = left[key], right[key]
        fields = ("answer", "source_sample_id", "group_id", "task_id", "condition_id",
                  "derivation_version", "allowed_answers")
        if any(a.get(field) != b.get(field) for field in fields):
            raise ValueError("paired bootstrap rows have incompatible targets or provenance")
        grouped[a["group_id"]].append(int(b["correct"]) - int(a["correct"]))
    groups = sorted(grouped)
    point = sum(sum(values) for values in grouped.values()) / len(left)
    result = {"metric": "accuracy", "contrast": "right_minus_left", "difference": point,
              "n_pairs": len(left), "n_groups": len(groups), "seed": seed,
              "repetitions": repetitions, "confidence_level": 0.95,
              "method": "paired_cluster_percentile", "interval": None, "warning": None}
    if len(groups) < 2:
        result["warning"] = "insufficient_independent_groups"
        return result
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = [grouped[rng.choice(groups)] for _ in groups]
        estimates.append(sum(sum(values) for values in sampled) / sum(len(values) for values in sampled))
    estimates.sort()

    def percentile(p: float) -> float:
        position = (len(estimates) - 1) * p
        lower, upper = math.floor(position), math.ceil(position)
        return estimates[lower] + (estimates[upper] - estimates[lower]) * (position - lower)

    result["interval"] = [percentile(0.025), percentile(0.975)]
    return result
