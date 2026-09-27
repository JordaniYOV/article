"""Verify automatic targets cannot silently treat ignored masks as negatives."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from planetary_vlm.datasets.semantic import convert_semantic
from planetary_vlm.io import read_jsonl, write_jsonl


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow optional dependency is not installed")
class SemanticTests(unittest.TestCase):
    def build_inputs(self, directory, pixels, *, rock_classes='["ROCK"]'):
        from PIL import Image
        Image.new("RGB", (2, 2), (80, 80, 80)).save(directory / "image.png")
        mask = Image.new("L", (2, 2))
        mask.putdata(pixels)
        mask.save(directory / "mask.png")
        write_jsonl(directory / "samples.jsonl", [{"sample_id": "s1", "image_path": "image.png",
            "mask_path": "mask.png", "group_id": "scene1", "split": "test",
            "license": "software-fixture", "source_url": "fixture:synthetic"}])
        (directory / "spec.toml").write_text(
            'ignore_labels=[255]\nroi=[0,0,1,1]\nmin_valid_fraction=0.5\n'
            'dominant_min_fraction=0.6\nrock_min_pixels=2\n'
            f'rock_classes={rock_classes}\ntasks=["terrain","rock_presence"]\n'
            '[label_to_class]\n0="SOIL"\n1="ROCK"\n', encoding="utf-8")

    def test_fully_annotated_masks_produce_separate_targets_and_provenance(self):
        with tempfile.TemporaryDirectory() as task_directory:
            directory = Path(task_directory)
            self.build_inputs(directory, [0, 0, 0, 1])
            output = convert_semantic(directory / "samples.jsonl", directory / "spec.toml", directory / "output")
            requests = read_jsonl(output / "requests.jsonl")
            targets = read_jsonl(output / "targets.jsonl")
            self.assertEqual({row["task_id"]: row["answer"] for row in targets},
                             {"terrain": "SOIL", "rock_presence": "NO"})
            self.assertTrue(all("answer" not in row and "mask_path" not in row for row in requests))
            self.assertEqual(read_jsonl(output / "provenance.jsonl")[0]["split"], "test")

    def test_ignored_pixels_do_not_create_negative_or_dominance_targets(self):
        with tempfile.TemporaryDirectory() as task_directory:
            directory = Path(task_directory)
            self.build_inputs(directory, [0, 0, 255, 255])
            output = convert_semantic(directory / "samples.jsonl", directory / "spec.toml", directory / "output")
            self.assertEqual(read_jsonl(output / "targets.jsonl"), [])
            exclusions = read_jsonl(output / "exclusions.jsonl")
            self.assertEqual({row["task_id"] for row in exclusions}, {"terrain", "rock_presence"})

    def test_duplicate_rock_classes_rejected(self):
        with tempfile.TemporaryDirectory() as task_directory:
            directory = Path(task_directory)
            self.build_inputs(directory, [0, 0, 0, 1], rock_classes='["ROCK","ROCK"]')
            with self.assertRaises(ValueError):
                convert_semantic(directory / "samples.jsonl", directory / "spec.toml", directory / "output")
