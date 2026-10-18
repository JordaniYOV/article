"""Display-only palette checks; synthetic arrays, no published labels modified."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from planetary_vlm.datasets.mask_preview import PALETTE, render

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = Image = None


@unittest.skipIf(np is None or Image is None, "CPU image dependencies required")
class MaskPreviewTests(unittest.TestCase):
    def test_palette_and_overlay_preserve_background_and_source_arrays(self):
        pixels = np.full((4, 4, 3), 100, dtype=np.uint8)
        labels = np.zeros((4, 4), dtype=np.uint8)
        labels[1, 1] = 1
        old_pixels, old_labels = pixels.copy(), labels.copy()
        panel = render(Image.fromarray(pixels), Image.fromarray(labels), ["Background", "Crater"], "fixture")
        output = np.asarray(panel)
        x, y = 12, 56
        np.testing.assert_array_equal(output[y:y + 4, x:x + 4], pixels)
        colour_x, overlay_x = x + 4 + 12, x + 2 * (4 + 12)
        np.testing.assert_array_equal(output[y + 1, colour_x + 1], PALETTE[1])
        np.testing.assert_array_equal(output[y, overlay_x], pixels[0, 0])
        expected = np.rint(pixels[1, 1] * .55 + np.array(PALETTE[1]) * .45).astype(np.uint8)
        np.testing.assert_array_equal(output[y + 1, overlay_x + 1], expected)
        np.testing.assert_array_equal(labels, old_labels)
        np.testing.assert_array_equal(pixels, old_pixels)

    def test_unknown_source_class_is_not_silently_coloured_as_crater(self):
        image = Image.fromarray(np.zeros((4, 4, 3), np.uint8))
        mask = Image.fromarray(np.full((4, 4), 2, np.uint8))
        with self.assertRaisesRegex(ValueError, "taxonomy"):
            render(image, mask, ["Background", "Crater"], "fixture")

    def test_shape_mismatch_is_not_resized_or_hidden(self):
        image = Image.fromarray(np.zeros((4, 4, 3), np.uint8))
        mask = Image.fromarray(np.zeros((3, 4), np.uint8))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            render(image, mask, ["Background", "Crater"], "fixture")


if __name__ == "__main__":
    unittest.main()
