"""Offline portable-bundle/restore fixtures, never evidence of GPU compatibility."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planetary_vlm import colab
from planetary_vlm.contracts import ModelResponse
from planetary_vlm.contracts import ModelRequest
from planetary_vlm.inference.runner import file_hash
from planetary_vlm.io import read_jsonl

try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = np = None


@unittest.skipIf(Image is None or np is None, "Install requirements/api.txt")
class ColabTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "source"
        for name in ("src/planetary_vlm/inference/full_systems.py", "src/benchmark_services/__init__.py",
                     "src/api/__init__.py", "src/api/logging.py", "pyproject.toml", "requirements/spacellava.txt"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source fixture\n", encoding="utf-8")
        self.collection = self.root / "data_orbital" / "orbital_300_v3"
        for planet in ("moon", "mars"):
            for track in ("segmentation", "height", "train"):
                sid = f"{planet}_{track}_001"
                path = self.collection / planet / track / "image.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                pixels = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
                Image.fromarray(pixels).save(path)
                row = {"sample_id": sid, "planet": planet, "track": track,
                    "source_split": "train" if track == "train" else "val", "source_dataset": "fixture",
                    "source_revision": "fixture-revision", "source_image_id": sid, "scene_group_id": sid,
                    "license_claim": "synthetic fixture", "evaluation_eligible": track != "train",
                    "image": path.relative_to(self.root).as_posix(), "image_sha256": file_hash(path),
                    "answer": "PRIVATE_GROUND_TRUTH", "mask": "NEVER_READ_MASK.png",
                    "height_map": "NEVER_READ_DEPTH.npy"}
                (self.collection / planet / (track + "_manifest.jsonl")).write_text(json.dumps(row) + "\n", encoding="utf-8")
        (self.collection / "audit.json").write_text(json.dumps({"total_images": 6}), encoding="utf-8")
        self.zip_path = self.root / "outputs" / "colab" / "bundle.zip"

    def bundle(self, *, conditions=("clean",)):
        with redirect_stdout(StringIO()):
            result = colab.export_bundle(self.root, self.zip_path, conditions=conditions)
        target = Path(self.temporary.name) / "colab"
        # Execute the actual notebook bootstrap offline (no Colab/Drive/GPU).
        notebook = json.loads((ROOT / "notebooks" / "spacellava_colab.ipynb").read_text(encoding="utf-8"))
        source = "".join(next(c for c in notebook["cells"] if c["id"] == "unpack")["source"])
        import hashlib, stat
        context = dict(Path=Path, zipfile=zipfile, json=json, hashlib=hashlib, stat=stat,
                       BUNDLE_ZIP=self.zip_path, WORKSPACE=target)
        with redirect_stdout(StringIO()):
            exec(compile(source, "colab-unpack-cell", "exec"), context)
        return target, result

    def test_bundle_keeps_eval_splits_and_has_no_ground_truth_or_train_images(self):
        target, summary = self.bundle(conditions=colab.CONDITIONS)
        self.assertEqual(summary["selected_images"], 4)
        self.assertEqual(summary["requests"], 24)
        self.assertEqual(summary["excluded"], {"publisher_train_or_ineligible": 2})
        metadata = colab.verify_bundle(target)
        rows = read_jsonl(target / "data_orbital" / "colab_inputs" / "requests.jsonl")
        self.assertTrue(all(set(r) <= {"request_id", "image_paths", "prompt", "allowed_answers", "max_new_tokens", "do_sample", "modality_paths"} for r in rows))
        self.assertTrue(all(not r["modality_paths"] for r in rows))
        self.assertNotIn("PRIVATE_GROUND_TRUTH", json.dumps(rows))
        with zipfile.ZipFile(self.zip_path) as archive:
            self.assertFalse(any("mask" in name or name.endswith(".npy") or "/train/" in name for name in archive.namelist()))
            provenance = archive.read("data_orbital/colab_inputs/provenance.jsonl").decode()
            self.assertNotIn("PRIVATE_GROUND_TRUTH", provenance)
        self.assertEqual(metadata["collection_images"], 6)

    def test_changed_image_and_unsafe_paths_fail_before_export(self):
        for value in ("../escape", "/tmp/image", "data_rover/image.png", "a\\image.png", "C:/image.png", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                colab.safe_relative(value)
        image = self.collection / "moon" / "height" / "image.png"
        image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed source image"):
            colab.export_bundle(self.root, self.zip_path)
        self.assertFalse(self.zip_path.exists())

    def test_crater_bundle_is_clean_candidate_inputs_without_answers(self):
        from planetary_vlm.datasets.crater_reference import TASKS, prompt_for
        manifests = {}
        requests = []
        for planet in ('moon', 'mars'):
            for track in ('segmentation', 'train'):
                path = self.collection / planet / (track + '_manifest.jsonl')
                manifests[path.relative_to(self.root).as_posix()] = file_hash(path)
            row = read_jsonl(self.collection / planet / 'segmentation_manifest.jsonl')[0]
            for task in TASKS:
                requests.append(ModelRequest(row['sample_id'] + '__' + task + '__clean',
                    (row['image'],), prompt_for(task, 32, 32), ('YES', 'NO'), max_new_tokens=16))
        record = dict(source_manifests_sha256=manifests, bundle_sha256='fixture-hash',
                      rules={'version':'fixture', 'sha256':'fixture'},
                      question_type='candidate_Q1_Q4_clean_unscored', reference_ready_questions=0)
        with patch('planetary_vlm.datasets.crater_annotations.load_crater_requests', return_value=(requests, record)):
            with self.assertRaisesRegex(ValueError, 'clean only'):
                colab.export_bundle(self.root, self.zip_path, crater_questions='fixture')
            with self.assertRaisesRegex(ValueError, 'not mixed'):
                colab.export_bundle(self.root, self.zip_path, questions='height', crater_questions='crater')
            with redirect_stdout(StringIO()):
                result = colab.export_bundle(self.root, self.zip_path, conditions=('clean',), crater_questions='fixture')
        self.assertEqual((result['selected_images'], result['requests']), (2, 8))
        self.assertEqual(result['excluded']['no_admitted_question'], 2)
        target = Path(self.temporary.name) / 'crater-inputs'
        with zipfile.ZipFile(self.zip_path) as archive:
            archive.extractall(target)  # Exclusively our synthetic trusted fixture.
            self.assertFalse(any(n.endswith('.npy') or 'targets.jsonl' in n or 'questions.jsonl' in n
                                 or '/train/' in n or 'mask' in n for n in archive.namelist()))
        bundle = colab.verify_bundle(target)
        self.assertEqual(bundle['protocol']['question_type'], 'candidate_Q1_Q4_clean_unscored')
        inputs = read_jsonl(target / 'data_orbital/colab_inputs/requests.jsonl')
        self.assertEqual(len({r['request_id'] for r in inputs}), 8)
        self.assertTrue(all(not r['modality_paths'] and 'answer' not in r and 'candidate_answer' not in r for r in inputs))
        # Q1-Q4 can no longer pass on a lone arbitrary Q1 response.
        env = {'gpu':'MOCK_CPU_TEST_NOT_GPU', 'dependencies':{'python':'fixture'}}
        with patch('planetary_vlm.colab.environment', return_value=env), \
                patch('planetary_vlm.models.registry.create_adapter', return_value=self.fake_adapter()), \
                redirect_stdout(StringIO()):
            with self.assertRaisesRegex(ValueError, '--smoke-controls'):
                colab.run_job(target, Path(self.temporary.name) / 'crater-drive',
                              'candidate-fixture', 'c' * 32, smoke=True)

    def test_export_never_overwrites_and_verify_rejects_changed_input(self):
        target, _ = self.bundle()
        with self.assertRaises(FileExistsError):
            colab.export_bundle(self.root, self.zip_path, conditions=("clean",))
        row = read_jsonl(target / "data_orbital" / "colab_inputs" / "requests.jsonl")[0]
        (target / row["image_paths"][0]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            colab.verify_bundle(target)

    def fake_adapter(self, *, interrupt_after=None):
        class Adapter:
            calls = []
            closed = False
            def initialize(self): return {"device_map": "{'': 0}", "checkpoint_sha256": {"fixture": "hash"}}
            def predict(self, request):
                self.calls.append(request.request_id)
                if interrupt_after and len(self.calls) == interrupt_after: raise KeyboardInterrupt()
                return ModelResponse(request.request_id, "SYNTHETIC CPU FIXTURE, NOT SCIENTIFIC", "ok", .01)
            def close(self): self.closed = True
        return Adapter()

    def test_smoke_gate_and_real_runner_checkpoint_restore_skip_saved_requests(self):
        target, _ = self.bundle()
        persistent = Path(self.temporary.name) / "drive"
        session_id = "a" * 32
        env = {"gpu": "MOCK_CPU_TEST_NOT_GPU", "dependencies": {"python": "fixture"}}
        with patch("planetary_vlm.colab.environment", return_value=env), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(ValueError, "smoke test"):
                colab.run_job(target, persistent, "benchmark", session_id)
            smoke = self.fake_adapter()
            with patch("planetary_vlm.models.registry.create_adapter", return_value=smoke):
                result = colab.run_job(target, persistent, "benchmark", session_id, smoke=True, checkpoint_every=1)
            self.assertTrue(result["success"])
            self.assertEqual(len(smoke.calls), 1)
            adapter = self.fake_adapter(interrupt_after=3)
            with patch("planetary_vlm.models.registry.create_adapter", return_value=adapter), self.assertRaises(KeyboardInterrupt):
                colab.run_job(target, persistent, "benchmark", session_id, checkpoint_every=1)
            self.assertTrue(adapter.closed)
            local = target / "outputs" / "SpaceLLaVA" / "benchmark"
            self.assertEqual(len(read_jsonl(local / "predictions.jsonl")), 2)
            # A move inside our temporary test folder simulates loss of local results.
            self.assertTrue(local.resolve().is_relative_to(Path(self.temporary.name).resolve()))
            local.rename(Path(self.temporary.name) / "discarded_local_fixture")
            resumed = self.fake_adapter()
            with patch("planetary_vlm.models.registry.create_adapter", return_value=resumed):
                final = colab.run_job(target, persistent, "benchmark", session_id, resume=True, checkpoint_every=1)
            self.assertTrue(final["success"])
            self.assertEqual(len(resumed.calls), 2)
            self.assertEqual(len(read_jsonl(local / "predictions.jsonl")), 4)

    def test_notebook_schema_code_and_safe_defaults(self):
        # The current notebook is v8; its previous user-owned v7 outputs are
        # preserved separately before updating this same notebook file.
        notebook = json.loads((ROOT / "notebooks" / "spacellava_language_colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])
        parameters = "".join(next(c for c in notebook["cells"] if c["id"] == "parameters")["source"])
        self.assertIn("DOWNLOAD_WEIGHTS = False", parameters)
        self.assertIn("RUN_PILOT = False", parameters)
        try:
            import nbformat
        except ImportError:
            return
        nbformat.validate(nbformat.from_dict(notebook))

    def test_checkpoint_permission_retry_is_bounded_and_preserves_atomic_replace(self):
        local = self.root / "checkpoint"
        local.mkdir()
        (local / "run.json").write_text('{"fixture":true}', encoding="utf-8")
        persistent = self.root / "drive_checkpoint"
        replace = colab.os.replace
        attempts = []
        def briefly_locked(source, target):
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError("synthetic temporary file lock")
            replace(source, target)
        with patch.object(colab.os, "replace", side_effect=briefly_locked), patch.object(colab.time, "sleep") as sleep:
            colab.checkpoint(local, persistent)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual((persistent / "run.json").read_bytes(), (local / "run.json").read_bytes())
        with patch.object(colab.os, "replace", side_effect=PermissionError("persistent denial")) as locked, \
             patch.object(colab.time, "sleep") as sleep, self.assertRaises(PermissionError):
            colab.checkpoint(local, persistent)
        self.assertEqual(locked.call_count, 6)
        self.assertEqual(sleep.call_count, 5)


if __name__ == "__main__":
    unittest.main()
