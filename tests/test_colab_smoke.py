"""Synthetic CPU fixtures only; no evidence of real model or GPU correctness."""

from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planetary_vlm import colab
from planetary_vlm.colab_smoke import SCHEMA, TASKS, assess, load_controls
from planetary_vlm.contracts import ModelRequest, ModelResponse
from planetary_vlm.inference.runner import file_hash
from planetary_vlm.io import write_json, write_jsonl, read_jsonl

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = Image = None


@unittest.skipIf(Image is None or np is None, "Install image extras")
class SmokeControlsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.requests, self.provenance, controls = [], [], []
        answers = (("YES", "NO"), ("YES", "NO"), tuple(map(str, range(1, 10)))+("NONE", "TIE"), ("LEFT", "RIGHT", "TIE"))
        for number, (sid, target) in enumerate((("positive", "YES"), ("negative", "NO"))):
            path = self.root / "data_orbital" / "colab_inputs" / "images" / sid / "clean.png"
            path.parent.mkdir(parents=True)
            Image.fromarray((np.arange(32*32).reshape(32, 32) + number*10).astype(np.uint8)).save(path)
            checksum = file_hash(path)
            controls.append(dict(sample_id=sid, source_split="val", expected_presence=target,
                image_sha256=checksum, annotation_fingerprint="synthetic-fixture", complete_for_size_rule=True,
                review_basis="Synthetic CPU fixture, not real labels"))
            for task, labels in zip(TASKS, answers):
                request_id = sid+"__"+task+"__clean"
                self.requests.append(ModelRequest(request_id, (path.relative_to(self.root).as_posix(),),
                                                  "Synthetic fixture prompt", labels, 16))
                self.provenance.append(dict(request_id=request_id, sample_id=sid, source_split="val", condition="clean",
                    image_sha256=checksum, scene_group_id="scene-"+sid, evaluation_eligible=True))
        self.control_path = self.root / "evaluator_controls.json"
        write_json(self.control_path, dict(schema=SCHEMA, scientific_ground_truth=False, controls=controls))
        inputs = self.root / "data_orbital" / "colab_inputs"
        write_jsonl(inputs / "requests.jsonl", map(asdict, self.requests))
        write_jsonl(inputs / "provenance.jsonl", self.provenance)
        self.bundle = {"protocol":{"question_type":"candidate_Q1_Q4_clean_unscored", "seed":42},
                       "bundle_sha256":"fixture-bundle", "counts":{}, "files_sha256":{}}
        self.env = {"gpu":"CPU_FIXTURE_NOT_GPU", "dependencies":{"python":"fixture"}}

    def controls(self):
        return load_controls(self.root, self.control_path, self.requests, self.provenance)

    def fake_adapter(self, *, always_yes=False, bad_q3=False):
        class Adapter:
            calls, closed = [], False
            def initialize(self): return {"device_map":"{'': 0}", "checkpoint_sha256":{}}
            def predict(self, request):
                self.calls.append(request)
                if "crater_largest" in request.request_id:
                    answer = "1, 2, 3, 4, 5," if bad_q3 else "NONE"
                elif "crater_half" in request.request_id: answer = "TIE"
                else: answer = "YES" if always_yes or request.request_id.startswith("positive") else "NO"
                return ModelResponse(request.request_id, answer, "ok", .01)
            def close(self): self.closed = True
        return Adapter()

    def run_smoke(self, adapter, *, vision_ok=True):
        report = {"success":vision_ok, "protocol":"spacellava-visual-diagnostic-v1", "fixture":True}
        with patch.object(colab, "verify_bundle", return_value=self.bundle), \
             patch.object(colab, "environment", return_value=self.env), \
             patch("planetary_vlm.models.registry.create_adapter", return_value=adapter), \
             patch("benchmark_services.spacellava.diagnostics.diagnose", return_value=report), \
             redirect_stdout(StringIO()):
            return colab.run_job(self.root, self.root / "drive", "fixture", "d"*32, smoke=True,
                                 smoke_controls=self.control_path, checkpoint_every=1)

    def test_selection_is_two_val_images_ten_requests_and_no_reference_in_model_input(self):
        controls = self.controls()
        self.assertEqual(len(controls["requests"]), 10)
        self.assertEqual(len(controls["expected"]), 4)
        self.assertEqual(len(set(controls["images"])), 2)
        for request in controls["requests"]:
            self.assertNotIn("expected_presence", asdict(request))
            self.assertFalse(request.modality_paths)
            self.assertEqual(request.max_new_tokens, 16)
        # Do not mutate the original benchmark manifest/prompts.
        self.assertEqual(len(self.requests), 8)
        self.assertTrue(all(r.prompt == "Synthetic fixture prompt" for r in self.requests))

    def test_test_split_and_changed_bytes_are_rejected(self):
        self.provenance[0]["source_split"] = "test"
        with self.assertRaisesRegex(ValueError, "val provenance"): self.controls()
        self.provenance[0]["source_split"] = "val"
        (self.root / self.requests[0].image_paths[0]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"): self.controls()

    def test_success_requires_correct_positive_negative_and_q3_format(self):
        controls = self.controls()
        adapter = self.fake_adapter()
        rows = [dict(asdict(adapter.predict(r)), mock=False) for r in controls["requests"]]
        self.assertTrue(assess(rows, controls["requests"], controls["expected"], {"success":True})["success"])
        for change in ("constant_yes", "invalid_q3", "missing", "duplicate", "bad_vision"):
            altered = [dict(row) for row in rows]
            diagnostics = {"success": change != "bad_vision"}
            if change == "constant_yes":
                for row in altered:
                    if "presence" in row["request_id"]: row["raw_response"] = "YES"
            if change == "invalid_q3": altered[2]["raw_response"] = "1, 2, 3, 4, 5,"
            if change == "missing": altered.pop()
            if change == "duplicate": altered[-1] = altered[0]
            self.assertFalse(assess(altered, controls["requests"], controls["expected"], diagnostics)["success"], change)

    def test_actual_runner_loads_once_diagnoses_then_persists_separate_assessment(self):
        adapter = self.fake_adapter()
        result = self.run_smoke(adapter)
        self.assertTrue(result["success"])
        self.assertTrue(adapter.closed)
        self.assertEqual(len(adapter.calls), 10)
        folder = Path(result["output_dir"])
        self.assertTrue((folder / "vision_diagnostics.json").is_file())
        self.assertTrue((Path(result["persistent_dir"]) / "smoke_assessment.json").is_file())
        self.assertFalse(any("expected_presence" in row for row in read_jsonl(folder / "requests.jsonl")))
        # Successful new evidence admits a run, old one-image proofs do not.
        proof_path = self.root / "colab_smoke_passed.json"
        proof = json.loads(proof_path.read_text())
        proof.pop("smoke_protocol")
        colab.atomic_json(proof_path, proof)
        with patch.object(colab,"verify_bundle",return_value=self.bundle), patch.object(colab,"environment",return_value=self.env):
            with self.assertRaisesRegex(ValueError,"two-image"):
                colab.run_job(self.root, self.root/"drive", "benchmark", "d"*32, smoke_controls=self.control_path)

    def test_changed_control_sidecar_or_diagnostic_evidence_invalidates_admission(self):
        result = self.run_smoke(self.fake_adapter())
        with patch.object(colab,"verify_bundle",return_value=self.bundle), patch.object(colab,"environment",return_value=self.env):
            original_bytes = self.control_path.read_bytes()
            sidecar = json.loads(self.control_path.read_text())
            sidecar["new_note"] = "changed after smoke"
            colab.atomic_json(self.control_path, sidecar)
            with self.assertRaisesRegex(ValueError,"two-image"):
                colab.run_job(self.root, self.root/"drive", "benchmark", "d"*32, smoke_controls=self.control_path)
            self.control_path.write_bytes(original_bytes)
            report = Path(result["output_dir"])/"vision_diagnostics.json"
            report.write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,"evidence missing or changed"):
                colab.run_job(self.root, self.root/"drive", "benchmark", "d"*32, smoke_controls=self.control_path)

    def test_visual_failure_stops_before_generation_and_leaves_no_success_proof(self):
        adapter = self.fake_adapter()
        with self.assertRaisesRegex(RuntimeError,"Visual diagnostics failed"):
            self.run_smoke(adapter, vision_ok=False)
        self.assertFalse(adapter.calls)
        self.assertTrue(adapter.closed)
        self.assertFalse(json.loads((self.root/"colab_smoke_passed.json").read_text())["success"])
        saved = self.root/"drive/runs/SpaceLLaVA/fixture_smoke_dddddddd/vision_diagnostics.json"
        self.assertTrue(saved.is_file())

    def test_constant_yes_and_list_q3_cannot_pass_actual_smoke(self):
        result = self.run_smoke(self.fake_adapter(always_yes=True, bad_q3=True))
        self.assertFalse(result["success"])
        assessment = json.loads((Path(result["output_dir"])/"smoke_assessment.json").read_text())
        self.assertEqual(sum(not r["format_valid"] for r in assessment["checks"]), 2)
        self.assertEqual(sum(r["control_correct"] is False for r in assessment["checks"]), 2)


@unittest.skipIf(np is None, "Install NumPy")
class VisualMetricTests(unittest.TestCase):
    def test_all_visual_weight_values_are_checked_not_just_key_names(self):
        from benchmark_services.spacellava.diagnostics import inspect_weights
        class Tensor:
            def __init__(self, values, *, meta=False, dtype="f16"):
                self.values = np.asarray(values, dtype=np.float16)
                self.is_meta, self.dtype = meta, dtype
                self.shape = self.values.shape
            def detach(self): return self
            def cpu(self): return self
            def to(self, *, dtype): return Tensor(self.values, dtype=dtype)
        expected = {"model.vision_tower.vision_tower.test.weight": Tensor([1, 2]),
                    "model.mm_projector.test.weight": Tensor([3, 4])}
        @contextmanager
        def opened(*args, **kwargs):
            yield SimpleNamespace(get_tensor=lambda key: expected[key])
        torch = SimpleNamespace(float16="f16", float32="f32", bfloat16="bf16",
            isfinite=lambda value: np.isfinite(value.values),
            equal=lambda a, b: bool(np.array_equal(a.values, b.values)))
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for failure in (None, "values", "nonfinite", "meta", "quantized", "missing", "extra"):
                state = {key: Tensor(value.values) for key, value in expected.items()}
                first = next(iter(state))
                if failure == "values": state[first] = Tensor([1, 5])
                if failure == "nonfinite": state[first] = Tensor([1, float("nan")])
                if failure == "meta": state[first] = Tensor([1, 2], meta=True)
                if failure == "quantized": state[first] = Tensor([1, 2], dtype="uint8")
                index = {key:"fixture.safetensors" for key in expected}
                if failure == "missing": index.pop(first)
                if failure == "extra": index["model.mm_projector.extra.weight"] = "fixture.safetensors"
                colab.atomic_json(folder/"model.safetensors.index.json", {"weight_map":index})
                model = SimpleNamespace(named_parameters=lambda:state.items(), named_buffers=lambda:iter(()),
                                        state_dict=lambda:state)
                adapter = SimpleNamespace(torch=torch, model=model, options={"model_dir":str(folder)})
                with patch.dict(sys.modules,{"safetensors":SimpleNamespace(safe_open=opened)}):
                    report = inspect_weights(adapter)
                self.assertEqual(report["success"], failure is None, failure)
                if failure is None: self.assertEqual(report["compared_tensors"], 2)

    def test_sensitivity_does_not_depend_on_decoded_argmax_changing(self):
        from benchmark_services.spacellava.diagnostics import compare_arrays
        left, right = np.array([10, 1, 0]), np.array([10, 2, 0])
        self.assertEqual(left.argmax(), right.argmax())
        self.assertTrue(compare_arrays(left, right)["different"])
        self.assertFalse(compare_arrays(left, left)["different"])
        self.assertFalse(compare_arrays(left, np.array([float("nan"), 1, 0]))["finite"])
        self.assertFalse(compare_arrays(left, np.ones((2, 3)))["same_shape"])


if __name__ == "__main__":
    unittest.main()
