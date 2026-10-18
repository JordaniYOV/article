"""Contract checks only: no dataset, model or annotation export is loaded."""
import math
from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AnnotationProtocolV2Tests(unittest.TestCase):
    def setUp(self):
        self.rules = tomllib.loads((ROOT / "configs/crater_annotation_v2.toml").read_text(encoding="utf-8"))

    def test_rules_fixed_does_not_claim_converter_reference_or_model_readiness(self):
        r = self.rules
        self.assertEqual(r["schema_version"], "crater-annotation-v2")
        self.assertTrue(r["annotation_rules_frozen"])
        self.assertTrue(r["test_annotation_ready"])
        for key in ("reference_review_complete", "complete_instances_available", "converter_implemented", "paired_ibm_admitted", "ground_truth_in_inference", "inspect_model_answers_during_annotation"):
            self.assertFalse(r[key])
        self.assertTrue(r["original_images_and_masks_immutable"])
        self.assertTrue((ROOT / r["protocol"]).is_file())

    def test_native_geometry_and_answer_labels_unchanged_from_v1(self):
        v1 = tomllib.loads((ROOT / "configs/crater_annotation_v1.toml").read_text(encoding="utf-8"))
        r = self.rules
        self.assertEqual(r["tasks"], v1["tasks"])
        self.assertEqual(r["regions"], v1["regions"])
        self.assertEqual(r["answers"], v1["answers"])
        for key in ("minimum_short_side_fraction", "minimum_short_side_rounding", "centre", "largest_size_proxy"):
            self.assertEqual(r["objects"][key], v1["objects"][key])
        for dimension, threshold in ((256, 11), (500, 20), (512, 21)):
            self.assertEqual(math.ceil(r["objects"]["minimum_short_side_fraction"] * dimension), threshold)
        self.assertEqual(r["visibility"]["image_border_margin_px"], 1)
        self.assertFalse(r["objects"]["reshape_bbox_to_pass_filters"])
        self.assertFalse(r["objects"]["semantic_connected_components_are_instances"])

    def test_uncertainty_is_question_level_and_not_exact_geometry_or_negative(self):
        r = self.rules
        self.assertFalse(r["visibility"]["uncertain_possible_eligible_object_blocks_all_four_tasks"])
        self.assertTrue(r["visibility"]["uncertain_object_is_not_a_negative"])
        u = r["uncertainty"]
        self.assertEqual(u["policy"], "question_level_answer_invariance")
        self.assertFalse(u["uncertain_boxes_are_exact_geometry_bounds"])
        self.assertFalse(u["remove_uncertain_to_make_reference"])
        self.assertTrue(u["geometry_bounds_require_explicit_review"])
        self.assertFalse(u["ambiguous_question_in_accuracy_denominator"])
        self.assertFalse(u["ambiguous_question_counts_as_model_error"])
        for key in ("q1", "q2", "q3", "q4"):
            self.assertNotIn(u["ambiguous_question_status"], r["answers"][key])

    def test_missing_pixels_and_review_are_not_false_negatives_or_auto_acceptance(self):
        r = self.rules
        for key in ("source_label_in_missing_region_is_confirmed_visual_truth", "uniform_pixels_automatically_mean_no_data", "unknown_region_empty_means_no_craters", "assume_finite_crater_count_without_evidence"):
            self.assertFalse(r["missing_visual_evidence"][key])
        self.assertTrue(r["review"]["second_review_required_for_scoring"])
        self.assertTrue(r["review"]["accepted_image_requires_uncertainty_accounted_for"])
        self.assertFalse(r["review"]["accepted_image_requires_no_uncertain_objects"])
        self.assertTrue(r["review"]["source_instances_required_before_scoring"])
        self.assertIn("unknown_regions", r["review"]["required_image_fields"])
        self.assertIn("reference_status", r["review"]["required_question_fields"])

    def test_cvat_results_preserve_source_and_complete_check_is_not_gt(self):
        r = self.rules["cvat"]
        self.assertEqual(r["default_result"], "not_reviewed")
        self.assertFalse(r["default_complete"])
        self.assertIn("no_eligible_craters", r["result_values"])
        self.assertNotIn("no_eligible_creaters", r["result_values"])
        self.assertTrue(r["unused_unknown_enum_options_are_not_image_annotations"])
        self.assertEqual(r["used_unknown_enum_policy"], "validation_error_no_silent_normalization")
        self.assertTrue(r["source_complete_means_no_supplement_not_no_objects"])
        self.assertTrue(r["needs_review_is_not_automatic_whole_image_exclusion"])

    def test_pairing_and_degradation_keep_answer_independent_admission(self):
        r = self.rules["comparisons"]
        self.assertTrue(r["admission_independent_of_model_answers"])
        self.assertTrue(r["same_question_scene_pool_for_paired_models"])
        self.assertTrue(r["clean_reference_admission_preserved_across_corruptions"])
        self.assertFalse(r["evidence_hidden_by_corruption_relabels_clean_reference"])

    def test_current_study_points_to_v2_and_pilot_exports_not_accepted_truth(self):
        study = tomllib.loads((ROOT / "configs/orbital_study.toml").read_text(encoding="utf-8"))
        self.assertEqual(study["questions"]["crater_annotation"]["config"], "configs/crater_annotation_v2.toml")
        # The frozen v2 file records its initial readiness snapshot; the current
        # study separately records the subsequently implemented candidate converter.
        self.assertTrue(study["questions"]["crater_annotation"]["converter_implemented"])
        self.assertFalse(study["questions"]["crater_annotation"]["complete_instances_available"])
        self.assertEqual(sum(p["source_val_images"] for p in self.rules["pilot_exports"]), 50)
        for p in self.rules["pilot_exports"]:
            self.assertFalse(p["accepted_ground_truth"])
            self.assertEqual(len(p["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
