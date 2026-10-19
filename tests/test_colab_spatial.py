"""CPU/synthetic fixtures only, not SpaceLLaVA GPU performance evidence."""

import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planetary_vlm import colab, colab_spatial as spatial
from planetary_vlm.contracts import ModelRequest, ModelResponse
from planetary_vlm.io import read_jsonl, write_jsonl
from benchmark_services.common.files import atomic_json


class SpatialPilotTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image, ImageDraw
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        inputs = self.root / "data_orbital/colab_inputs"
        inputs.mkdir(parents=True)
        self.source = inputs / "fixture.png"
        image = Image.new("RGB", (256, 256), (128, 128, 128))
        ImageDraw.Draw(image).ellipse((48, 48, 208, 208), fill=(0, 0, 0))
        image.save(self.source)
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        request = ModelRequest(spatial.SOURCE_ID + "__crater_presence_v1__clean",
                               (self.source.relative_to(self.root).as_posix(),), "FROZEN Q1", ("YES", "NO"), 16)
        self.origin = dict(request_id=request.request_id, sample_id=spatial.SOURCE_ID, source_split="val",
                           condition="clean", evaluation_eligible=True, image_sha256=self.sha,
                           scene_group_id="SYNTHETIC_FIXTURE")
        write_jsonl(inputs / "requests.jsonl", [asdict(request)])
        write_jsonl(inputs / "provenance.jsonl", [self.origin])
        self.bundle_patch = patch.object(colab, "verify_bundle", return_value={"bundle_sha256": "fixture"})
        self.hash_patch = patch.object(spatial, "SOURCE_SHA256", self.sha)
        self.bundle_patch.start(); self.hash_patch.start()
        self.addCleanup(self.bundle_patch.stop); self.addCleanup(self.hash_patch.stop)

    def test_six_controls_geometry_mirrors_determinism_and_no_source_edits(self):
        from PIL import Image, ImageOps, ImageChops
        controls = spatial.prepare_controls(self.root)
        self.assertEqual([c["expected_third"] for c in controls["controls"]],
                         ["LEFT", "RIGHT", "CENTER", "CENTER", "RIGHT", "LEFT"])
        for index in (0, 2, 4):
            a, b = controls["controls"][index:index + 2]
            with Image.open(self.root / a["image"]) as original, Image.open(self.root / b["image"]) as mirrored:
                self.assertEqual(original.size, (512, 512))
                self.assertIsNone(ImageChops.difference(ImageOps.mirror(original), mirrored).getbbox())
                x, y = a["placement_xy_before_mirror"]
                self.assertEqual(original.getpixel((x + 96, y + 96)), (0, 0, 0))
                self.assertEqual(original.getpixel((256, 20)), (128, 128, 128))
        self.assertEqual(spatial.prepare_controls(self.root), controls)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.sha)

    def test_changed_source_and_test_source_rejected(self):
        self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            spatial.prepare_controls(self.root)
        self.origin["source_split"] = "test"
        path = self.root / "data_orbital/colab_inputs/provenance.jsonl"
        atomic_json(path, [self.origin], lines=True)
        with self.assertRaisesRegex(ValueError, "val"):
            spatial.prepare_controls(self.root)

    def test_changed_control_never_overwritten(self):
        controls = spatial.prepare_controls(self.root)
        path = self.root / controls["controls"][0]["image"]
        path.write_bytes(b"user change")
        with self.assertRaisesRegex(ValueError, "image changed"):
            spatial.prepare_controls(self.root)
        self.assertEqual(path.read_bytes(), b"user change")

    def test_eighteen_requests_no_evaluator_targets_or_label_filenames(self):
        controls = spatial.prepare_controls(self.root)
        requests, settings = spatial.prepare_requests(self.root, controls)
        self.assertEqual(len(requests), 18)
        self.assertEqual(len({r.request_id for r in requests}), 18)
        self.assertEqual(len({r.image_paths for r in requests}), 6)
        for request in requests:
            self.assertFalse(request.allowed_answers or request.modality_paths)
            self.assertRegex(Path(request.image_paths[0]).name, r"^c\d\d\.png$")
            self.assertNotIn("expected", json.dumps(asdict(request)))
        self.assertTrue(all(v["conversation_template"] == "llava_v1" and v["use_cache"] for v in settings.values()))
        self.assertEqual(sum(v["compare_first_forward"] for v in settings.values()), 2)
        self.assertEqual({r.max_new_tokens for r in requests}, {32, 64})
        self.assertNotIn("LEFT", spatial.PROMPTS[0][1])
        self.assertNotEqual(spatial.PROMPTS[1][1], spatial.PROMPTS[2][1])

    def test_scoring_strict_closed_controls_only_never_benchmark_admission(self):
        controls = spatial.prepare_controls(self.root)
        requests, _ = spatial.prepare_requests(self.root, controls)
        targets = {"spatial_" + c["control_id"]: c["expected_third"] for c in controls["controls"]}
        rows = [asdict(ModelResponse(r.request_id, targets[r.request_id.split("__")[0]], "ok", .1)) for r in requests]
        report = spatial.summarize(rows, {}, controls)
        self.assertEqual(report["development_controls_correct"], 12)
        self.assertEqual(report["development_controls_total"], 12)
        self.assertFalse(report["benchmark_admitted"] or report["scientific_score_available"] or report["free_responses_automatically_scored"])
        rows[1]["raw_response"] = "The answer is LEFT."
        self.assertEqual(spatial.summarize(rows, {}, controls)["development_controls_correct"], 11)
        self.assertFalse(spatial.summarize(rows, {}, controls)["closed_control_checks"][0]["format_valid"])

    def test_cpu_runner_persists_six_images_eighteen_answers_and_keeps_smoke_proof(self):
        proof = self.root / "colab_smoke_passed.json"
        proof.write_text('{"success":false}', encoding="utf-8")
        before = proof.read_bytes()
        class Adapter:
            def __init__(self, options): self.options = options
            def initialize(self): return {"device_map": "{'': 0}"}
            def predict(self, request):
                self.assert_no_targets(request)
                path = Path(self.options["diagnostic_output_dir"]) / "generation_diagnostics.json"
                traces = json.loads(path.read_text()) if path.exists() else {}
                traces[request.request_id] = {"fixture": True, "hit_token_budget": False}
                atomic_json(path, traces)
                return ModelResponse(request.request_id, "LEFT", "ok", .1)
            def assert_no_targets(self, request):
                assert "expected" not in json.dumps(asdict(request))
                assert "expected" not in json.dumps(self.options)
            def close(self): pass
        with patch.object(colab, "environment", return_value={"fixture": True}), \
             patch("planetary_vlm.models.registry.create_adapter", side_effect=lambda _, options: Adapter(options)), \
             patch("benchmark_services.spacellava.diagnostics.diagnose", return_value={"success": True}):
            result = spatial.run_pilot(self.root, self.root / "drive", "spatial", "a" * 32)
        self.assertTrue(result["success"])
        self.assertFalse(result["benchmark_admitted"])
        self.assertEqual(proof.read_bytes(), before)
        durable = Path(result["persistent_dir"])
        self.assertEqual(len(list(durable.glob("*.png"))), 6)
        self.assertEqual(len(read_jsonl(durable / "predictions.jsonl")), 18)
        self.assertTrue((durable / "spatial_report.json").is_file())
        with patch.object(colab, "environment", return_value={}), self.assertRaisesRegex(ValueError, "exists"):
            spatial.run_pilot(self.root, self.root / "drive", "spatial", "a" * 32)

    def test_existing_notebook_spatial_command_safe_defaults_backup_and_syntax(self):
        if not (ROOT / 'notebooks/spacellava_language_colab.ipynb').is_file():
            self.skipTest('Generated legacy notebook intentionally removed')
        notebook = json.loads((ROOT / "notebooks/spacellava_language_colab.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        self.assertIn("spacellava_precision_v12.zip", code)
        self.assertIn("/content/planetary_language_v7", code)
        self.assertIn("'precision-pilot'", code)
        self.assertIn("/content/planetary_fresh_v9", code)
        self.assertIn("RUN_PILOT = False", code)
        self.assertIn("DOWNLOAD_WEIGHTS = False", code)
        self.assertNotIn("step('run'", code)
        for c in notebook["cells"]:
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]))
                self.assertFalse(c["outputs"])

    def test_cli_prepare_has_no_gpu_initialization(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with patch.object(colab, "environment", side_effect=AssertionError("Must be CPU only")), redirect_stdout(StringIO()):
            self.assertEqual(colab.main(["spatial-prepare", "--workspace", str(self.root)]), 0)


if __name__ == "__main__":
    unittest.main()
