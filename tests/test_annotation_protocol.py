"""Contract checks for the annotation rule file; no labels or models are loaded."""

import math
from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AnnotationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rules = tomllib.loads((ROOT / "configs" / "crater_annotation_v1.toml").read_text(encoding="utf-8"))

    def test_rule_file_is_pilot_not_an_annotation_or_model_admission_claim(self):
        r = self.rules
        self.assertEqual(r["schema_version"], "crater-annotation-v1")
        self.assertIn("pending", r["status"])
        self.assertFalse(r["converter_implemented"])
        self.assertFalse(r["annotations_acquired_or_manually_completed"])
        self.assertFalse(r["paired_ibm_admitted"])
        self.assertFalse(r["ground_truth_in_inference"])
        self.assertTrue(r["original_images_and_masks_immutable"])
        self.assertTrue(r["test_annotation_requires_frozen_rules"])
        self.assertFalse(r["inspect_model_answers_during_annotation"])
        self.assertTrue((ROOT / r["protocol"]).is_file())

    def test_native_size_threshold_and_regional_boundary_contract(self):
        r = self.rules
        self.assertEqual(r["objects"]["minimum_short_side_fraction"], 0.04)
        for dimension, expected in ((256, 11), (500, 20), (512, 21)):
            self.assertEqual(math.ceil(r["objects"]["minimum_short_side_fraction"] * dimension), expected)
        self.assertEqual(r["objects"]["centre"], "bbox_midpoint_without_rounding")
        self.assertEqual(r["regions"]["numbering"], "row_major_1_to_9")
        self.assertEqual(r["regions"]["midline_assignment"], "right")
        self.assertEqual(r["regions"]["central_cell"], 5)

    def test_unknowns_full_review_and_answer_sets_are_explicit(self):
        r = self.rules
        self.assertTrue(r["visibility"]["fully_within_frame_required_for_all_four_tasks"])
        self.assertTrue(r["visibility"]["uncertain_possible_eligible_object_blocks_all_four_tasks"])
        self.assertFalse(r["objects"]["semantic_connected_components_are_instances"])
        self.assertTrue(r["review"]["second_review_required_for_scoring"])
        self.assertTrue(r["review"]["empty_annotation_is_not_automatically_negative"])
        self.assertEqual(r["answers"]["q3_near_largest_fraction"], 0.05)
        self.assertEqual(r["answers"]["q1"], ["YES", "NO"])
        self.assertEqual(r["answers"]["q2"], ["YES", "NO"])
        self.assertEqual(r["answers"]["q3"], [str(i) for i in range(1, 10)] + ["NONE", "TIE"])
        self.assertEqual(r["answers"]["q4"], ["LEFT", "RIGHT", "TIE"])
        for key in ("q1", "q2", "q3", "q4"):
            self.assertNotIn("UNKNOWN_REFERENCE", r["answers"][key])
        self.assertIn("complete_for_size_rule", r["review"]["required_image_fields"])


if __name__ == "__main__":
    unittest.main()
