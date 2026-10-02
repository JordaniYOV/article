"""Integration coverage for GT isolation, run recovery and corrupted inputs."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planetary_vlm.cli import main
from planetary_vlm.contracts import ModelRequest
from planetary_vlm.inference.config import load_config
from planetary_vlm.inference.runner import run
from planetary_vlm.io import read_jsonl, write_jsonl


class PipelineTests(unittest.TestCase):
    def build_fixture(self, directory):
        image = directory / "asset.bin"
        image.write_bytes(b"fixture-image-not-decoded-by-mock")
        request = ModelRequest("r1", (str(image),), "Rock?", ("YES", "NO"))
        write_jsonl(directory / "requests.jsonl", [asdict(request)])
        (directory / "run.toml").write_text(
            '[run]\nrun_id="test"\nrequests="requests.jsonl"\noutput_dir="outputs"\n'
            '[model]\nid="mock"\nbackend="mock"\n[model.options.responses]\nr1="YES"\n', encoding="utf-8")
        return load_config(directory / "run.toml"), image

    def test_run_resume_and_changed_input_rejection(self):
        with tempfile.TemporaryDirectory() as task_directory:
            config, image = self.build_fixture(Path(task_directory))
            destination = run(config)
            rows = read_jsonl(destination / "predictions.jsonl")
            self.assertEqual(rows[0]["raw_response"], "YES")
            self.assertTrue(rows[0]["mock"])
            self.assertNotIn("answer", read_jsonl(destination / "requests.jsonl")[0])
            run(config, resume=True)
            self.assertEqual(len(read_jsonl(destination / "predictions.jsonl")), 1)
            with self.assertRaises(ValueError):
                run(config)
            image.write_bytes(b"changed-input")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run(config, resume=True)

    def test_cli_scores_persisted_responses_without_loading_model(self):
        with tempfile.TemporaryDirectory() as task_directory:
            directory = Path(task_directory)
            config, _ = self.build_fixture(directory)
            destination = run(config)
            write_jsonl(directory / "targets.jsonl", [{"request_id": "r1", "answer": "NO",
                "source_sample_id": "s1", "group_id": "g1", "derivation_version": "fixture"}])
            output = directory / "report.json"
            self.assertEqual(main(["evaluate", "--requests", str(config.requests),
                "--targets", str(directory / "targets.jsonl"), "--predictions",
                str(destination / "predictions.jsonl"), "--output", str(output)]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["overall"]["accuracy"], 0)
            self.assertEqual(report["overall"]["binary"]["fpr"], 1)
            self.assertTrue(report["mock"])

    def test_orbital_prepare_versions_parameters_and_keeps_targets_separate(self):
        try:
            from PIL import Image
            import numpy as np
            import cv2
        except ImportError:
            self.skipTest("Optional image dependencies are not installed")
        from planetary_vlm.inference.full_systems import prepare_inputs
        from planetary_vlm.inference.runner import file_hash
        with tempfile.TemporaryDirectory() as task_directory:
            directory = Path(task_directory)
            orbital = directory / "data_orbital"
            orbital.mkdir()
            image = orbital / "image.png"
            Image.fromarray(np.full((256, 256), 90, dtype=np.uint8)).save(image)
            write_jsonl(orbital / "manifest.jsonl", [{"sample_id": "s1", "planet": "moon",
                "source_dataset": "imp-fixture", "source_split": "test", "mask_semantics": "imp",
                "evaluation_eligible": True, "image": "data_orbital/image.png", "image_sha256": file_hash(image),
                "mask": "never-read.tif", "scene_group_id": "scene", "source_revision": "fixture",
                "source_image_id": "tile", "license_claim": "fixture-only"}])
            config = {"dataset": {"manifest": orbital / "manifest.jsonl", "prepared_dir": orbital / "prepared",
                "source_dataset": "imp-fixture", "splits": ["test"], "conditions": ["clean", "radiation"]},
                "generation": {"seed": 12, "prompt": "Visible IMP evidence?", "max_new_tokens": 32},
                "distortion": {"num_hits": 10, "max_streak_length": 5}}
            prepared, original_metadata = prepare_inputs(config, directory)
            rows = read_jsonl(prepared / "requests.jsonl")
            self.assertNotEqual(rows[0]["request_id"], rows[1]["request_id"])
            self.assertTrue(all("answer" not in row for row in rows))
            self.assertTrue(all(not row["modality_paths"] for row in rows))
            self.assertEqual(prepare_inputs(config, directory)[0], prepared)
            config["distortion"]["num_hits"] = 20
            with self.assertRaisesRegex(ValueError, "Preparation changed"):
                prepare_inputs(config, directory)
            config["dataset"]["prepared_dir"] = orbital / "changed"
            changed, new_metadata = prepare_inputs(config, directory)
            self.assertNotEqual(original_metadata["protocol_sha256"], new_metadata["protocol_sha256"])
            self.assertNotEqual(file_hash(Path(rows[1]["image_paths"][0])),
                file_hash(Path(read_jsonl(changed / "requests.jsonl")[1]["image_paths"][0])))


if __name__ == "__main__":
    unittest.main()
