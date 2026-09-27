"""Synthetic known-answer checks, not planetary benchmark findings."""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from planetary_vlm.evaluation import evaluate
from planetary_vlm.evaluation.metrics import paired_accuracy_bootstrap
from planetary_vlm.evaluation.parser import parse_answer
from planetary_vlm.io import records_fingerprint


def fixture(answers, outputs, allowed=None):
    allowed = ["YES", "NO"] if allowed is None else allowed
    requests, targets, predictions = [], [], []
    for index, answer in enumerate(answers):
        key = f"q{index}"
        requests.append({"request_id": key, "image_paths": ["synthetic.png"],
                         "prompt": "Choose one label.", "allowed_answers": list(allowed),
                         "max_new_tokens": 8, "do_sample": False})
        targets.append({"request_id": key, "answer": answer,
                        "source_sample_id": f"scene{index}", "group_id": f"group{index // 2}",
                        "derivation_version": "synthetic-v1"})
        if index < len(outputs) and outputs[index] is not None:
            output = outputs[index]
            predictions.append({"request_id": key,
                                "raw_response": output if isinstance(output, str) else "",
                                "status": "ok" if isinstance(output, str) else "error",
                                "elapsed_seconds": 0.0,
                                "error_message": None if isinstance(output, str) else "synthetic error"})
    return requests, targets, predictions


class ParserTests(unittest.TestCase):
    def test_normalization_without_prose_extraction(self):
        self.assertEqual(parse_answer(" \tYeS\n", ["YES", "NO"]), "YES")
        self.assertEqual(parse_answer("big   ROCK", ["big rock"]), "big rock")
        for text in ("YES.", "The answer is YES", '"YES"', "YES NO", ""):
            self.assertIsNone(parse_answer(text, ["YES", "NO"]))

    def test_ambiguous_labels_rejected(self):
        with self.assertRaises(ValueError):
            parse_answer("yes", ["YES", " yes "])


class EvaluationTests(unittest.TestCase):
    def test_known_binary_counts_and_failure_denominators(self):
        report = evaluate(*fixture(["YES"] * 4 + ["NO"] * 4,
                                   ["YES", "NO", "prose", None, "YES", "NO", "?", {}]))
        metric = report["overall"]
        self.assertEqual(metric["n_scored"], 8)
        self.assertEqual(metric["n_correct"], 2)
        self.assertEqual(metric["n_invalid"], 2)
        self.assertEqual(metric["n_failure"], 2)
        self.assertEqual(metric["accuracy"], 0.25)
        self.assertEqual(metric["valid_only_accuracy"], 0.5)
        self.assertEqual(metric["coverage"], 0.5)
        self.assertAlmostEqual(metric["macro_f1"], 1 / 3)
        self.assertEqual(metric["per_class"]["YES"]["fn"], 3)
        binary = metric["binary"]
        self.assertEqual([binary[k] for k in ("tp", "fp", "tn", "fn")], [1, 1, 1, 1])
        self.assertEqual(binary["fpr"], 0.5)
        self.assertEqual(binary["fnr"], 0.5)
        self.assertEqual(binary["fpr_denominator"], 2)
        self.assertEqual(binary["by_truth"]["positive"]["n_failure"], 1)
        self.assertEqual(binary["by_truth"]["negative"]["n_invalid"], 1)
        self.assertEqual(binary["positive_miss_rate_including_failures"], 0.75)
        self.assertEqual(report["per_request"][3]["response_status"], "missing")
        self.assertEqual(sum(sum(row) for row in metric["confusion_matrix"]["counts"]), 8)
        json.dumps(report, allow_nan=False)

    def test_multiclass_supported_macro(self):
        metric = evaluate(*fixture(["A", "A", "B"], ["A", "C", "B"], ["A", "B", "C"]))["overall"]
        self.assertAlmostEqual(metric["accuracy"], 2 / 3)
        self.assertEqual(metric["macro_f1_classes"], ["A", "B"])
        self.assertAlmostEqual(metric["macro_f1"], (2 / 3 + 1) / 2)
        self.assertIsNone(metric["per_class"]["C"]["recall"])
        self.assertEqual(metric["per_class"]["C"]["f1"], 0)
        self.assertIsNone(metric["binary"])

    def test_zero_denominators_and_all_missing(self):
        metric = evaluate(*fixture(["YES"], []))["overall"]
        self.assertEqual(metric["accuracy"], 0)
        self.assertEqual(metric["macro_f1"], 0)
        self.assertIsNone(metric["binary"]["fpr"])
        self.assertIsNone(metric["binary"]["fnr"])
        self.assertIsNone(metric["per_class"]["NO"]["f1"])

    def test_freeform_is_not_judged(self):
        report = evaluate(*fixture(["description"], ["description"], []))
        self.assertEqual(report["overall"]["classification_status"], "unsupported_scoring")
        self.assertEqual(report["overall"]["n_unsupported_scoring"], 1)
        self.assertIsNone(report["overall"]["accuracy"])
        self.assertIsNone(report["per_request"][0]["correct"])

    def test_text_only_tuple_inputs_and_mock_metadata(self):
        records = fixture(["YES"], ["YES"])
        records[0][0]["image_paths"] = ()
        records[0][0]["allowed_answers"] = ("YES", "NO")
        records[0][0]["modality_paths"] = (("depth", "depth.png"),)
        records[2][0]["mock"] = True
        report = evaluate(*records)
        self.assertEqual(report["overall"]["accuracy"], 1)
        self.assertIs(report["mock"], True)
        self.assertIs(report["per_request"][0]["mock"], True)

    def test_slices_and_incompatible_labels(self):
        requests, targets, predictions = fixture(["YES", "NO"], ["YES", "NO"])
        targets[1]["condition_id"] = "blur"
        requests[1]["allowed_answers"] = ["NO", "MAYBE"]
        report = evaluate(requests, targets, predictions)
        self.assertEqual(report["overall"]["accuracy"], 1)
        self.assertEqual(report["overall"]["classification_status"], "incompatible_label_sets")
        self.assertIsNone(report["overall"]["macro_f1"])
        self.assertEqual(len(report["slices"]), 2)

    def test_incompatible_within_slice_not_pooled(self):
        requests, targets, predictions = fixture(["YES", "NO"], ["YES", "NO"])
        requests[1]["allowed_answers"] = ["NO", "MAYBE"]
        report = evaluate(requests, targets, predictions)
        self.assertEqual(report["slices"][0]["metrics"]["classification_status"], "incompatible_label_sets")

    def test_task_macro_not_pooled_even_when_labels_equal(self):
        requests, targets, predictions = fixture(["YES", "NO"], ["YES", "NO"])
        targets[1]["task_id"] = "other_task"
        report = evaluate(requests, targets, predictions)
        self.assertEqual(report["overall"]["classification_status"], "multiple_tasks_use_slices")
        self.assertIsNone(report["overall"]["macro_f1"])

    def test_duplicate_records_rejected_for_all_inputs(self):
        for index in range(3):
            records = fixture(["YES"], ["YES"])
            records[index].append(copy.deepcopy(records[index][0]))
            with self.subTest(index=index), self.assertRaises(ValueError):
                evaluate(*records)

    def test_missing_stale_targets_and_unknown_predictions(self):
        for index in (1, 2):
            records = fixture(["YES"], ["YES"])
            records[index][0]["request_id"] = "stale"
            with self.subTest(index=index), self.assertRaises(ValueError):
                evaluate(*records)
        requests, _, predictions = fixture(["YES"], ["YES"])
        with self.assertRaises(ValueError):
            evaluate(requests, [], predictions)

    def test_gt_mixing_and_unknown_fields_rejected(self):
        for index, field in ((0, "answer"), (0, "ground_truth"), (2, "answer")):
            records = fixture(["YES"], ["YES"])
            records[index][0][field] = "YES"
            with self.subTest(index=index, field=field), self.assertRaises(ValueError):
                evaluate(*records)

    def test_invalid_target_and_malformed_request_rejected(self):
        records = fixture(["MAYBE"], ["YES"])
        with self.assertRaises(ValueError):
            evaluate(*records)
        for field, value in (("max_new_tokens", True), ("image_paths", "image.png"),
                             ("do_sample", "false"), ("allowed_answers", ["YES", "yes"]),
                             ("modality_paths", [["depth", "a"], ["depth", "b"]])):
            records = fixture(["YES"], ["YES"])
            records[0][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate(*records)

    def test_malformed_predictions_rejected(self):
        for field, value in (("elapsed_seconds", float("nan")), ("elapsed_seconds", -1),
                             ("status", "missing"), ("raw_response", None)):
            records = fixture(["YES"], ["YES"])
            records[2][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate(*records)

    def test_mixed_runs_or_models_rejected(self):
        for field in ("model_id", "run_id"):
            records = fixture(["YES", "NO"], ["YES", "NO"])
            records[2][0][field] = "a"
            records[2][1][field] = "b"
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate(*records)

    def test_unsupported_model_is_failure_not_invalid(self):
        records = fixture(["YES"], ["YES"])
        records[2][0]["status"] = "unsupported"
        metric = evaluate(*records)["overall"]
        self.assertEqual(metric["n_failure"], 1)
        self.assertEqual(metric["n_invalid"], 0)
        self.assertEqual(metric["accuracy"], 0)

    def test_empty_inputs_and_no_mutation(self):
        self.assertEqual(evaluate([], [], [])["overall"]["classification_status"], "empty")
        records = fixture(["YES"], ["YES"])
        before = copy.deepcopy(records)
        evaluate(*records)
        self.assertEqual(records, before)

    def test_request_fingerprint_reordering_and_missing_prediction(self):
        requests, targets, predictions = fixture(["YES", "NO"], ["YES", "NO"])
        fingerprint = records_fingerprint(requests)
        for prediction in predictions:
            prediction["request_manifest_sha256"] = fingerprint
        report = evaluate(requests[::-1], targets, predictions[::-1])
        self.assertEqual(report["request_manifest_sha256"], fingerprint)
        self.assertTrue(report["request_manifest_verified"])
        partial = evaluate(requests, targets, predictions[:1])
        self.assertEqual(partial["overall"]["n_failure"], 1)
        self.assertTrue(partial["request_manifest_verified"])

    def test_request_fingerprint_rejects_stale_prompt_and_labels_with_same_ids(self):
        for field, value in (("prompt", "Changed prompt."),
                             ("allowed_answers", ["YES", "NO", "MAYBE"])):
            records = fixture(["YES"], ["YES"])
            records[2][0]["request_manifest_sha256"] = records_fingerprint(records[0])
            records[0][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "does not match"):
                evaluate(*records)

    def test_request_fingerprint_missing_mixed_and_malformed(self):
        records = fixture(["YES", "NO"], ["YES", "NO"])
        self.assertFalse(evaluate(*records)["request_manifest_verified"])
        records[2][0]["request_manifest_sha256"] = records_fingerprint(records[0])
        with self.assertRaisesRegex(ValueError, "consistent"):
            evaluate(*records)
        for marker in (None, 1, "x" * 64, "a" * 63):
            records = fixture(["YES"], ["YES"])
            records[2][0]["request_manifest_sha256"] = marker
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, "hexadecimal"):
                evaluate(*records)


class BootstrapTests(unittest.TestCase):
    def test_known_cluster_weighting_and_reproducibility(self):
        left = evaluate(*fixture(["YES"] * 3, ["NO"] * 3))["per_request"]
        right = evaluate(*fixture(["YES"] * 3, ["YES", "YES", "NO"]))["per_request"]
        result = paired_accuracy_bootstrap(left, right, seed=7, repetitions=200)
        self.assertEqual(result, paired_accuracy_bootstrap(left[::-1], right, seed=7, repetitions=200))
        self.assertAlmostEqual(result["difference"], 2 / 3)
        self.assertEqual(result["n_groups"], 2)
        self.assertEqual(result["interval"], [0, 1])

    def test_insufficient_groups_and_mismatched_pairs(self):
        rows = evaluate(*fixture(["YES"], ["YES"]))["per_request"]
        result = paired_accuracy_bootstrap(rows, rows, repetitions=10)
        self.assertIsNone(result["interval"])
        self.assertEqual(result["warning"], "insufficient_independent_groups")
        for changed in ([{**rows[0], "answer": "NO"}], [], [{**rows[0], "correct": None}]):
            with self.assertRaises(ValueError):
                paired_accuracy_bootstrap(rows, changed)


if __name__ == "__main__":
    unittest.main()
