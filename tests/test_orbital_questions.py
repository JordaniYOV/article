"""Synthetic CPU fixtures for reference derivation and inference/GT separation."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planetary_vlm.datasets import orbital_questions as questions
from planetary_vlm.io import read_jsonl

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = Image = None


@unittest.skipIf(np is None or Image is None, "Install the CPU image requirements")
class OrbitalQuestionTests(unittest.TestCase):
    def test_height_gradient_known_answers_and_affine_unit_invariance(self):
        h = np.repeat(np.arange(9).reshape(3, 3), 3, axis=0).repeat(3, axis=1).astype(float)
        v = np.ones_like(h, dtype=bool)
        for height in (h, h * 150 + 800, h / 8):
            answers, detail = questions.height_answers(height, v, (9, 9))
            self.assertEqual(answers, {"height_pair_v1": "B", "height_lowest_v1": "1"})
            self.assertEqual(detail["valid_fractions"], [1.0] * 9)

    def test_constant_and_multiple_lowest_cells_are_ties(self):
        answers, _ = questions.height_answers(np.zeros((9, 9)), np.ones((9, 9), bool), (9, 9))
        self.assertEqual(answers, {"height_pair_v1": "TIE", "height_lowest_v1": "TIE"})
        h = np.ones((9, 9))
        h[:3, :6] = 0
        answers, _ = questions.height_answers(h, np.ones_like(h, bool), (9, 9))
        self.assertEqual(answers["height_lowest_v1"], "TIE")
        self.assertEqual(answers["height_pair_v1"], "B")

    def test_nan_nodata_and_coverage_do_not_become_lowest_height(self):
        h = np.ones((9, 9))
        h[3:6, 3:6] = np.nan
        answers, detail = questions.height_answers(h, np.ones_like(h, bool), (9, 9))
        self.assertEqual(answers, {"height_pair_v1": "TIE"})
        self.assertEqual(detail["valid_fractions"][4], 0)
        answers, _ = questions.height_answers(h, np.zeros_like(h, bool), (9, 9))
        self.assertEqual(answers, {})
        h[:3, :3] = np.nan
        answers, _ = questions.height_answers(h, np.ones_like(h, bool), (9, 9))
        self.assertEqual(answers, {})

    def test_invalid_shape_and_nonbinary_validity_are_blocked(self):
        for h, v, size in ((np.ones((9, 9)), np.ones((8, 9)), (9, 9)),
                           (np.ones((9, 9)), np.ones((9, 9)), (10, 9)),
                           (np.ones((9, 9)), np.full((9, 9), 2), (9, 9)),
                           (np.ones((2, 2)), np.ones((2, 2)), (2, 2))):
            with self.subTest(size=size):
                self.assertEqual(questions.height_answers(h, v, size)[0], {})

    def test_uneven_grid_covers_every_pixel_once(self):
        h = np.arange(128 * 128).reshape(128, 128)
        answers, detail = questions.height_answers(h, np.ones_like(h, bool), (128, 128))
        self.assertEqual(answers["height_pair_v1"], "B")
        self.assertEqual(answers["height_lowest_v1"], "1")
        self.assertAlmostEqual(detail["region_means"][8], float(h[85:, 85:].mean()))

    def make_source(self, root):
        collection = root / "data_orbital" / "orbital_300_v3"
        for planet in ("moon", "mars"):
            for track in ("segmentation", "height", "train"):
                sid = f"{planet}_{track}_001"
                folder = collection / planet / track / "001"
                folder.mkdir(parents=True)
                image = folder / "image.png"
                Image.fromarray(np.zeros((9, 9, 3), dtype=np.uint8)).save(image)
                row = {"sample_id": sid, "planet": planet, "track": track,
                    "source_split": "train" if track == "train" else "val",
                    "evaluation_eligible": track != "train", "source_dataset": "synthetic_fixture",
                    "scene_group_id": sid, "source_revision": "synthetic", "final_test_eligible": False,
                    "image": image.relative_to(root).as_posix(), "image_sha256": questions.file_hash(image),
                    "mask": "DO_NOT_READ_A_MASK_AS_INSTANCES.png"}
                if track == "height":
                    for field, data in (("height_map", np.arange(81).reshape(9, 9)),
                                        ("height_valid_mask", np.ones((9, 9), bool))):
                        path = folder / (field + ".npy")
                        np.save(path, data)
                        row[field] = path.relative_to(root).as_posix()
                        row[field + "_sha256"] = questions.file_hash(path)
                (collection / planet / (track + "_manifest.jsonl")).write_text(json.dumps(row) + "\n", encoding="utf-8")
        (collection / "audit.json").write_text(json.dumps({"total_images": 6}), encoding="utf-8")
        for name in ("src/planetary_vlm/inference/full_systems.py", "src/benchmark_services/__init__.py",
                     "src/api/__init__.py", "src/api/logging.py", "pyproject.toml", "requirements/spacellava.txt"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic fixture\n", encoding="utf-8")

    def test_build_separates_targets_keeps_all_templates_and_excludes_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_source(root)
            folder = root / "data_orbital" / "questions_v1"
            summary = questions.build_questions(root, folder)
            self.assertEqual(summary["requests"], 4)
            self.assertEqual(summary["selected_images"], 2)
            requests, record = questions.load_question_requests(root, folder)
            self.assertEqual(record["bundle_sha256"], summary["bundle_sha256"])
            self.assertEqual(len(requests), 4)
            self.assertTrue(all(not r.modality_paths and len(r.image_paths) == 1 for r in requests))
            self.assertTrue(all(r.allowed_answers and "height_map" not in r.prompt for r in requests))
            self.assertTrue(all("train" not in r.request_id for r in requests))
            targets = read_jsonl(folder / "targets.jsonl")
            self.assertEqual({r["request_id"] for r in targets}, {r.request_id for r in requests})
            admitted = read_jsonl(folder / "admission.jsonl")
            self.assertEqual(len(admitted), 4 * 8)
            self.assertTrue(all(not r["paired_ibm_admitted"] for r in admitted))
            self.assertTrue(all(not r["reference_admitted"] for r in admitted if r["task_id"].startswith("crater_")))
            with self.assertRaises(FileExistsError):
                questions.build_questions(root, folder)
            (folder / "targets.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                questions.load_question_requests(root, folder)

    def test_colab_question_bundle_contains_no_reference_targets_and_rejects_geometry_edits(self):
        from planetary_vlm.colab import export_bundle
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_source(root)
            folder = root / "data_orbital" / "questions_v1"
            questions.build_questions(root, folder)
            archive = root / "outputs" / "colab" / "questions.zip"
            with self.assertRaisesRegex(ValueError, "clean only"):
                export_bundle(root, archive, questions=folder)
            self.assertFalse(archive.exists())
            with redirect_stdout(StringIO()):
                summary = export_bundle(root, archive, conditions=("clean",), questions=folder)
            self.assertEqual(summary["selected_images"], 2)
            self.assertEqual(summary["requests"], 4)
            with zipfile.ZipFile(archive) as bundle:
                self.assertFalse(any(name.endswith("targets.jsonl") or name.endswith(".npy") for name in bundle.namelist()))
                rows = json.loads(bundle.read("colab_bundle.json"))
                self.assertEqual(rows["protocol"]["question_type"], "reference_ready_Q5_Q6_clean_pilot")
                requests = [json.loads(line) for line in bundle.read("data_orbital/colab_inputs/requests.jsonl").decode().splitlines()]
                self.assertEqual({r["request_id"] for r in requests},
                                 {r["request_id"] for r in read_jsonl(folder / "targets.jsonl")})
                for r in requests:
                    self.assertNotIn("answer", r)
                    self.assertEqual(r["modality_paths"], [])
                    self.assertIn(r["image_paths"][0], bundle.namelist())

    def test_changed_reference_and_protected_path_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_source(root)
            path = root / "data_orbital" / "orbital_300_v3" / "moon" / "height" / "001" / "height_map.npy"
            path.write_bytes(b"tampered")
            folder = root / "data_orbital" / "questions_v1"
            with self.assertRaisesRegex(ValueError, "Changed reference"):
                questions.build_questions(root, folder)
            self.assertFalse(folder.exists())
            for name in ("data_rover/image.png", "../outside.npy", "data_orbital/../data_rover/a",
                         "data_orbital/a\\b", "C:/data_orbital/a"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    questions.orbital_path(root, name)

    def test_question_smoke_rejects_freeform_and_accepts_exact_label_without_using_targets(self):
        from planetary_vlm import colab
        from planetary_vlm.contracts import ModelResponse
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            self.make_source(root)
            folder = root / "data_orbital" / "questions_v1"
            questions.build_questions(root, folder)
            archive = root / "outputs" / "colab" / "questions.zip"
            target = Path(temporary) / "colab"
            with redirect_stdout(StringIO()):
                colab.export_bundle(root, archive, conditions=("clean",), questions=folder)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(target)  # Our synthetic trusted fixture, not an uploaded archive.
            class Adapter:
                response = "I think A because it looks bright"
                def initialize(self): return {"device_map": "{'': 0}"}
                def close(self): pass
                def predict(self, request):
                    return ModelResponse(request.request_id, self.response, "ok", 0.01)
            adapter = Adapter()
            env = {"gpu": "SYNTHETIC_CPU_FIXTURE_NOT_GPU", "dependencies": {"python": "fixture"}}
            with patch("planetary_vlm.colab.environment", return_value=env), \
                    patch("planetary_vlm.models.registry.create_adapter", return_value=adapter), \
                    redirect_stdout(StringIO()):
                result = colab.run_job(target, Path(temporary) / "drive", "questions", "a" * 32, smoke=True)
                self.assertFalse(result["success"])
                with self.assertRaisesRegex(ValueError, "smoke test"):
                    colab.run_job(target, Path(temporary) / "drive", "questions", "a" * 32)
                adapter.response = "A"
                result = colab.run_job(target, Path(temporary) / "drive", "questions", "b" * 32, smoke=True)
                self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
