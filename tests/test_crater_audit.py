"""Candidate extraction checks. Components do not imply geological instances."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from planetary_vlm.datasets.crater_audit import components

try:
    import numpy as np
    import cv2
except ImportError:
    np = cv2 = None


@unittest.skipIf(np is None or cv2 is None, "CPU image dependencies required")
class CraterAuditTests(unittest.TestCase):
    def test_extracts_only_selected_class_and_does_not_mutate_mask(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[10:20, 30:45] = 3
        mask[50:90, 50:90] = 2
        original = mask.copy()
        rows = components(mask, 3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mask_bbox_xywh"], [30, 10, 15, 10])
        self.assertEqual(rows[0]["mask_pixels"], 150)
        self.assertEqual(rows[0]["bbox_midpoint_xy_proxy"], [37.5, 15])
        self.assertTrue(rows[0]["mask_bbox_passes_size_proxy"])
        self.assertFalse(rows[0]["reference_admitted"])
        self.assertTrue(rows[0]["needs_instance_and_geometry_review"])
        np.testing.assert_array_equal(mask, original)

    def test_border_small_and_diagonal_components_are_proposals_not_dropped(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[0:8, 0:8] = 3
        mask[20, 20] = mask[21, 21] = 3
        rows = components(mask, 3)
        self.assertEqual(len(rows), 2)  # 8-connectivity joins the diagonal pixels.
        self.assertFalse(rows[0]["mask_bbox_within_frame_margin"])
        self.assertTrue(rows[0]["mask_bbox_passes_size_proxy"])
        self.assertFalse(rows[1]["mask_bbox_passes_size_proxy"])
        self.assertEqual(rows[1]["mask_pixels"], 2)

    def test_contact_is_flagged_but_not_called_occlusion_or_instance_truth(self):
        mask = np.zeros((10, 10), np.uint8)
        mask[2:5, 2:5] = 3
        mask[2:5, 5:8] = 4
        row = components(mask, 3)[0]
        self.assertTrue(row["touches_different_foreground_class"])
        self.assertNotIn("occluded", row)
        self.assertFalse(row["reference_admitted"])
        self.assertEqual(components(np.zeros_like(mask), 3), [])

    def test_rgb_or_float_mask_is_rejected(self):
        for mask in (np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), float)):
            with self.assertRaises(ValueError):
                components(mask, 3)


if __name__ == "__main__":
    unittest.main()
