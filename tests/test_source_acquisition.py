"""Protect distinctions between missing semantic GT and binary depth validity."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from planetary_vlm.datasets.sources import download
from planetary_vlm.datasets.lune import lunar_candidates


class SourceAcquisitionTests(unittest.TestCase):
    def test_depth_validity_is_not_semantic_ground_truth(self):
        inventory = [{"type": "file", "path": name} for name in
                     ("images/001.png", "depth/001.npy", "masks/001.npy")]
        row, = lunar_candidates(inventory, "pinned-revision")
        self.assertEqual(row["source_depth"], "depth/001.npy")
        self.assertEqual(row["source_depth_validity_mask"], "masks/001.npy")
        self.assertIsNone(row["source_semantic_mask"])
        self.assertFalse(row["benchmark_eligible"])

    def test_unmatched_depth_is_recorded_as_missing(self):
        inventory = [{"type": "file", "path": "images/001.png"},
                     {"type": "file", "path": "depth/002.npy"}]
        row, = lunar_candidates(inventory, "revision")
        self.assertIsNone(row["source_depth"])
        self.assertIsNone(row["source_depth_validity_mask"])

    def test_corrupted_cached_file_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            path.write_bytes(b"bad")
            checksum = "sha256:" + hashlib.sha256(b"good").hexdigest()
            with self.assertRaisesRegex(ValueError, "failed validation"):
                download("https://example.invalid/unreachable", path, checksum=checksum)

    def test_verified_cached_file_requires_no_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            path.write_bytes(b"good")
            checksum = "sha256:" + hashlib.sha256(b"good").hexdigest()
            asset = download("https://example.invalid/unreachable", path, 4, checksum)
            self.assertEqual(asset["sha256"], checksum.removeprefix("sha256:"))


if __name__ == "__main__":
    unittest.main()
