"""Reproduce source-specific asset conversions before checking curated hashes."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import hashlib
import unittest
from unittest.mock import Mock, patch

try:
    from PIL import Image
    import pyarrow.parquet
    from planetary_vlm.datasets.orbital import materialize
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Install images and orbital dependencies")
class OrbitalMaterializeTests(unittest.TestCase):
    def encode(self, image, format):
        buffer = BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()

    def restore(self, repo, image_bytes, mask_bytes):
        # Include a different image first to check exact source identity selection.
        records = [
            {"image": {"path": "other.tif", "bytes": b"wrong"}, "mask": {"bytes": b"wrong"}},
            {"image": {"path": "source/selected.tif", "bytes": image_bytes}, "mask": {"bytes": mask_bytes}},
        ]
        parquet = Mock()
        parquet.iter_batches.return_value = [SimpleNamespace(to_pylist=lambda: records)]
        cache = Mock()
        cache.find.return_value = {"path": "data/test-00000-of-00001.parquet"}
        cache.asset.return_value = Path("unused.parquet")
        row = {"source_dataset": repo, "source_revision": "pinned", "source_image_id": "selected.tif", "source_split": "test"}
        with patch("pyarrow.parquet.ParquetFile", return_value=parquet):
            return materialize(row, cache, Path.cwd(), Path("unused-staging"))

    def test_boulder_tiffs_are_reencoded_to_the_recorded_png_format(self):
        image = Image.new("L", (500, 500), 100)
        mask = Image.new("L", (500, 500), 0)
        mask.putpixel((10, 20), 1)
        source_image, source_mask = self.encode(image, "TIFF"), self.encode(mask, "TIFF")
        restored = self.restore("Mirali33/mb-boulder_seg", source_image, source_mask)
        for key, original, source in (("image", image, source_image), ("mask", mask, source_mask)):
            expected = self.encode(original, "PNG")
            self.assertNotEqual(hashlib.sha256(source).digest(), hashlib.sha256(expected).digest())
            self.assertEqual(hashlib.sha256(restored[key]).digest(), hashlib.sha256(expected).digest())
            with Image.open(BytesIO(restored[key])) as decoded:
                self.assertEqual(decoded.format, "PNG")
                self.assertEqual(decoded.mode, original.mode)
                self.assertEqual(decoded.size, original.size)
                self.assertEqual(decoded.tobytes(), original.tobytes())

    def test_other_mars_sources_keep_their_original_bytes(self):
        image = Image.new("L", (16, 16), 100)
        source = self.encode(image, "TIFF")
        restored = self.restore("Mirali33/mb-crater_binary_seg", source, source)
        self.assertEqual(restored, {"image": source, "mask": source})

    def test_boulder_dimensions_must_match_the_original_acquisition_contract(self):
        wrong_size = self.encode(Image.new("L", (16, 16)), "TIFF")
        with self.assertRaisesRegex(ValueError, "Unexpected boulder image size"):
            self.restore("Mirali33/mb-boulder_seg", wrong_size, wrong_size)


if __name__ == "__main__":
    unittest.main()
