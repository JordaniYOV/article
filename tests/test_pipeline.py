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

    def test_prepare_reuses_scene_noise_and_keeps_targets_separate(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow optional dependency is not installed")
        from planetary_vlm.transforms.prepare import prepare, expand_targets
        with tempfile.TemporaryDirectory() as task_directory:
            directory = Path(task_directory)
            Image.new("RGB", (8, 8), (90, 90, 90)).save(directory / "image.png")
            write_jsonl(directory / "requests.jsonl", [asdict(ModelRequest(
                key, ("image.png",), "Rock?", ("YES", "NO"))) for key in ("r1", "r2")])
            (directory / "variants.toml").write_text(
                'seed=12\n[[variants]]\nid="noise"\ntransform="noise"\n'
                '[variants.parameters]\nsigma=10\n', encoding="utf-8")
            prepared = prepare(directory / "requests.jsonl", directory / "variants.toml", directory / "prepared")
            rows = read_jsonl(prepared / "requests.jsonl")
            self.assertEqual(rows[0]["image_paths"], rows[1]["image_paths"])
            self.assertNotEqual(rows[0]["request_id"], rows[1]["request_id"])
            self.assertTrue(all("answer" not in row for row in rows))
            write_jsonl(directory / "targets.jsonl", [{"request_id": key, "answer": "YES",
                "source_sample_id": "s1", "group_id": "g1", "derivation_version": "fixture"}
                for key in ("r1", "r2")])
            expand_targets(directory / "targets.jsonl", prepared / "variants.jsonl", directory / "expanded.jsonl")
            targets = read_jsonl(directory / "expanded.jsonl")
            self.assertEqual({row["request_id"] for row in targets}, {row["request_id"] for row in rows})
            self.assertTrue(all(row["condition_id"] == "noise" for row in targets))
            (directory / "variants.toml").write_text(
                'seed=12\n[[variants]]\nid="noise"\ntransform="noise"\n'
                '[variants.parameters]\nsigma=20\n', encoding="utf-8")
            changed = prepare(directory / "requests.jsonl", directory / "variants.toml", directory / "changed")
            self.assertNotEqual(read_jsonl(changed / "requests.jsonl")[0]["request_id"], rows[0]["request_id"])


if __name__ == "__main__":
    unittest.main()
