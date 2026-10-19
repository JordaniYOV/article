"""Tiny pinned-Parquet fixtures: no network, Torch, weights or training."""
from contextlib import ExitStack
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from planetary_vlm.datasets import mars_binary as binary


READY = all(importlib.util.find_spec(name) for name in ("PIL", "numpy", "pyarrow"))


class SafetyContracts(unittest.TestCase):
    def test_paths_never_target_other_collections_or_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "data_orbital/existing"
            protected.mkdir(parents=True)
            (protected / "original.txt").write_text("unchanged")
            for value in (".", "data_orbital", "data_rover/new", "../escape", "data_orbital/existing"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    binary.collection_path(root, value)
            self.assertEqual(binary.collection_path(root, binary.DEFAULT_OUTPUT).parent, root / "data_orbital")
            with self.assertRaises(ValueError):
                binary._child(protected, "../../escape")
            self.assertEqual((protected / "original.txt").read_text(), "unchanged")

    def test_changed_asset_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original"
            binary._write(path, b"first")
            binary._write(path, b"first")
            with self.assertRaisesRegex(ValueError, "Immutable"):
                binary._write(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_download_resumes_and_validates_bounded_range(self):
        class Response(BytesIO):
            status = 206
            headers = {"Content-Range": "bytes 3-5/6"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            path.with_name("source.partial").write_bytes(b"abc")
            with patch.object(binary, "urlopen", return_value=Response(b"def")) as request:
                binary._download("https://example.invalid/source", path, 6, hashlib.sha256(b"abcdef").hexdigest())
                self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=3-")
            self.assertEqual(path.read_bytes(), b"abcdef")

    def test_bad_range_or_bad_source_digest_is_rejected(self):
        class Response(BytesIO):
            status = 206
            headers = {"Content-Range": "bytes 2-5/6"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            path.with_name("source.partial").write_bytes(b"abc")
            with patch.object(binary, "urlopen", return_value=Response(b"def")):
                with self.assertRaisesRegex(ValueError, "Content-Range"):
                    binary._download("https://example.invalid/source", path, 6, "f" * 64)
            self.assertFalse(path.exists())

    def test_cli_build_and_verify_are_independent_commands(self):
        from planetary_vlm.cli import main
        with patch.object(binary, "build_collection", return_value={"fixture": True}) as builder:
            self.assertEqual(main(["dataset", "mars-binary-build", "--workspace", "."]), 0)
            self.assertFalse(builder.call_args.kwargs["allow_download"])
        with patch.object(binary, "verify_collection", return_value={"valid": True}) as verifier:
            self.assertEqual(main(["dataset", "mars-binary-verify", "--workspace", "."]), 0)
            verifier.assert_called_once()


@unittest.skipUnless(READY, "Pillow/NumPy/PyArrow optional dataset dependencies")
class CollectionContracts(unittest.TestCase):
    def row(self, name="thm_dir_N00_000_001.png", seed=1, empty=False):
        import numpy as np
        from PIL import Image
        pixels = np.random.default_rng(seed).integers(0, 256, (512, 512), dtype="uint8")
        mask = np.zeros((512, 512), dtype="uint8")
        if not empty:
            mask[100:180, 120:210] = 1
        payload = {}
        for key, array in (("image", pixels), ("mask", mask)):
            stream = BytesIO()
            Image.fromarray(array).save(stream, format="PNG")
            payload[key] = {"bytes": stream.getvalue(), "path": name}
        return {**payload, "width": 512, "height": 512,
                "class_labels": ["Background"] if empty else ["Background", "Crater"]}

    def fixtures(self, root):
        import pyarrow as pa
        import pyarrow.parquet as pq
        destination = root / binary.DEFAULT_OUTPUT
        source = destination / "sources"
        source.mkdir(parents=True)
        binary._write_json(destination / "collection.json", {
            "schema": binary.SCHEMA, "repo": binary.REPO, "revision": binary.REVISION})
        entries = []
        for split, names in (("train", ["thm_dir_N00_000_001.png", "thm_dir_N00_000_002.png"]),
                             ("val", ["thm_dir_N00_030_001.png"]),
                             ("test", ["thm_dir_N00_060_001.png"])):
            filename = split + ".parquet"
            rows = [self.row(name, seed=20 * len(entries) + i, empty=split == "test")
                    for i, name in enumerate(names)]
            pq.write_table(pa.Table.from_pylist(rows), source / filename)
            entries.append((split, filename, (source / filename).stat().st_size, binary.sha(source / filename)))
        card = source / "README.md"
        card.write_text("license: cc-by-4.0\n", encoding="utf-8")
        stack = ExitStack()
        for name, value in {"SOURCES": tuple(entries), "EXPECTED_COUNTS": {"train": 2, "val": 1, "test": 1},
                            "CARD_SIZE": card.stat().st_size, "CARD_SHA": binary.sha(card)}.items():
            stack.enter_context(patch.object(binary, name, value))
        return stack, destination

    def test_roundtrip_native_bytes_separate_targets_and_resume(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack, destination = self.fixtures(root)
            with stack, patch.object(binary, "urlopen", side_effect=AssertionError("Network forbidden")):
                first = binary.build_collection(root)
                self.assertTrue(first["valid"])
                self.assertEqual(first["pairs_verified"], 4)
                before = (destination / "metadata.json").read_bytes()
                self.assertEqual(binary.build_collection(root), first)
                self.assertEqual((destination / "metadata.json").read_bytes(), before)
                inputs = binary._rows(destination / "inputs.jsonl")
                targets = binary._rows(destination / "targets.jsonl")
                self.assertEqual(len(inputs), len(targets))
                for row in inputs:
                    self.assertEqual(set(row), set(binary.INPUT_KEYS))
                    self.assertFalse({"mask", "qa", "classes", "foreground_fraction"} & row.keys())
                    with Image.open(root / row["image"]) as image:
                        self.assertEqual(image.mode, "L")
                audit = json.loads((destination / "audit.json").read_text())
                self.assertEqual(audit["split_statistics"]["test"]["empty_masks"], 1)
                self.assertEqual(audit["cross_split_scene_groups"], [])

    def test_image_mask_and_manifest_tampering_refused(self):
        for field in ("image", "mask", "manifest"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                stack, destination = self.fixtures(root)
                with stack:
                    binary.build_collection(root)
                    row = binary._rows(destination / "manifest.jsonl")[0]
                    path = destination / "manifest.jsonl" if field == "manifest" else root / row[field]
                    path.write_bytes(path.read_bytes() + b"tampered")
                    with self.assertRaises(ValueError):
                        binary.verify_collection(root)

    def test_labels_shape_and_filename_validated(self):
        from PIL import Image
        for problem in ("label", "shape", "path", "mismatch", "class_names"):
            row = self.row()
            if problem in {"label", "shape"}:
                stream = BytesIO()
                Image.new("L", (512, 512) if problem == "label" else (511, 512),
                          255 if problem == "label" else 1).save(stream, format="PNG")
                row["mask"]["bytes"] = stream.getvalue()
            elif problem == "path":
                row["image"]["path"] = "../thm_dir_N00_000_001.png"
            elif problem == "mismatch":
                row["mask"]["path"] = "thm_dir_N00_000_002.png"
            else:
                row["class_labels"] = ["Background"]
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                binary.validate_record(row)

    def test_exact_cross_split_and_prior_overlap_flags_not_silent_filter(self):
        rows = []
        for i, split in enumerate(("train", "test")):
            name, group, pixels, dhash, qa = binary.validate_record(self.row())
            rows.append({"sample_id": f"sample_{i}", "split": split, "source_image_id": name,
                         "scene_group_id": group, "image_sha256": "same", "rgb_pixels_sha256": pixels,
                         "dhash64": dhash, "qa": qa})
        prior = {"records": [{"source_stem": name[:-4], "rgb_pixels_sha256": pixels,
                              "dhash64": dhash, "scene_group_id": group,
                              "prior_manifest": "named_fixture", "sample_id": "old"}]}
        audit = binary.audit_records(rows, prior)
        self.assertEqual(audit["admission"], "review_required")
        self.assertEqual(len(audit["cross_split_scene_groups"]), 1)
        self.assertEqual(len(audit["decoded_image_duplicates"]), 1)
        self.assertEqual(audit["prior_overlap_counts"], {"train": 1, "test": 1})
        self.assertEqual(audit["split_counts"], {"train": 1, "test": 1})

    def test_dhash_search_flags_two_bit_distance_but_not_three(self):
        records = [{"sample_id": str(i), "split": "train", "rgb_pixels_sha256": str(i),
                    "dhash64": f"{number:016x}"} for i, number in enumerate((0, 3, 7))]
        pairs = binary._near_pairs(records)
        self.assertIn(["0", "1"], [r["sample_ids"] for r in pairs])
        self.assertNotIn(["0", "2"], [r["sample_ids"] for r in pairs])


if __name__ == "__main__":
    unittest.main()
