"""Offline checks of partner identity and hypothetical geometric coverage."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"), "Optional stereo dependencies")
class PartnerSearchTests(unittest.TestCase):
    def model(self, eye, center):
        return {"product_id": f"0703{eye}0029750010304398E01_DXXX", "C": center, "A": [0, 0, 1], "O": [0, 0, 1], "H": [100, 0, 50], "V": [0, 100, 50], "R": [0, 0, 0], "frame": "fixture", "frame_indices": {"site": "1"}, "frame_solution": "fixture", "axes": {"Sample": 100, "Line": 100}, "start_date_time": "2020-01-01T00:00:00Z"}

    def test_only_release_padding_is_normalized(self):
        from planetary_vlm.datasets._mars.partners import normalized
        self.assertEqual(normalized("0703ML0029750010304398E01_DXXX"), normalized("0703ML0029750010304398E1_DXXX"))
        self.assertNotEqual(normalized("0703ML0029750010304398E01_DXXX"), normalized("0703MR0029750010304398E01_DXXX"))

    def test_aligned_cameras_produce_candidates_not_verified_pairs(self):
        from planetary_vlm.datasets._mars.partners import candidate, rays
        left, right = self.model("ML", [0, 0, 0]), self.model("MR", [.265, 0, 0])
        pair, reason = candidate(left, right, {m["product_id"]: rays(m) for m in (left, right)})
        self.assertIsNone(reason)
        self.assertGreater(pair["predicted_right_coverage"], .8)
        self.assertFalse(pair["image_verified"])
        self.assertFalse(pair["depth_reconstructed"])

    def test_legacy_and_compression_identity_remain_hypotheses(self):
        from planetary_vlm.datasets._mars.partners import identity_hypothesis
        self.assertEqual(identity_hypothesis("0137ML0818012000E1_DXXX", "0137ML0008180120104200E01_DXXX"), "legacy_short_id_sequence_command_hypothesis")
        self.assertEqual(identity_hypothesis("0703MR0029750000402286E02_DXXX", "0703MR0029750000402286E01_DXXX"), "same_capture_other_compression_version")
        self.assertIsNone(identity_hypothesis("0137ML0818012000E1_DXXX", "0137MR0008180120104200E01_DXXX"))

    def test_different_frame_or_unphysical_baseline_rejected(self):
        from planetary_vlm.datasets._mars.partners import candidate
        left, right = self.model("ML", [0, 0, 0]), self.model("MR", [.265, 0, 0])
        right["frame_indices"] = {"site": "2"}
        self.assertEqual(candidate(left, right, {})[1], "coordinate_frame_mismatch")
        right["frame_indices"] = left["frame_indices"]
        right["C"] = [10, 0, 0]
        self.assertEqual(candidate(left, right, {})[1], "baseline_outside_mastcam_search_gate")
