"""Trained-head API/worker contracts with explicitly synthetic CPU maps only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    from PIL import Image
    from fastapi.testclient import TestClient
    from api.main import create_app
    from benchmark_services.common.settings import Settings
    from benchmark_services.common.files import atomic_json, file_hash
    from benchmark_services.common.http_app import create_service
    from benchmark_services.common.preparation import prepare_bundle, load_bundle
    from benchmark_services.common.execution import run_selected
    from benchmark_services.common.store import Run
    from sqlmodel import Session
    from planetary_vlm.heads_contract import CLASSES, INPUT_PROTOCOL, TASK, pixel_hash
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "Install requirements/api.txt")
class FrozenHeadsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patches = [patch("benchmark_services.common.settings.WORKSPACE_ROOT", self.root), patch("api.main.WORKSPACE_ROOT", self.root)]
        for patcher in self.patches:
            patcher.start()
        self.settings = Settings(database_path=self.root / ".state/test.sqlite3", service_token="synthetic-heads-test-token", service_mock=True,
                                 worker_poll_seconds=.02, _env_file=None)
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.folder = self.root / "outputs/deployment"
        self.folder.mkdir(parents=True)
        self.inputs, self.targets = [], []
        data = self.root / "data_orbital/fixture"
        data.mkdir(parents=True)
        for i, split in enumerate(("val", "test", "test")):
            image, mask, native = data / f"image{i}.png", data / f"mask{i}.png", data / f"native{i}.tif"
            Image.new("RGB", (512, 512), (i, 10, 20)).save(image)
            target = np.zeros((512, 512), dtype=np.uint8)
            if i != 1:
                target[100:200, 100:200] = 1
            Image.fromarray(target).save(mask)
            native.write_bytes(f"synthetic native fixture only {i}".encode())
            common = {"sample_id": f"fixture_{i}", "scene_group_id": f"group{i}", "source_dataset": "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection"}
            self.inputs.append(common | {"split": split, "rgb_pixels_sha256": pixel_hash(image), "native_vis_sha256": file_hash(native)})
            self.targets.append(common | {"source_split": split, "image": str(image), "image_sha256": file_hash(image),
                "native_vis": str(native), "native_vis_sha256": file_hash(native), "mask": str(mask), "mask_sha256": file_hash(mask),
                "classes": CLASSES, "mask_semantics": "catalog_crater_polygon", "reference_status": "ready", "evaluation_eligible": True})
        self.manifest = data / "manifest.jsonl"
        atomic_json(self.manifest, self.targets, lines=True)
        (self.folder / "head.pt").write_bytes(b"CPU fixture ONLY: not real trained weights")
        self.write_deployment()

    def write_deployment(self):
        atomic_json(self.folder / "encoder_inputs.jsonl", self.inputs, lines=True)
        atomic_json(self.folder / "deployment.json", {"schema": "frozen-crater-deployment-v1", "task": TASK, "input_protocol": INPUT_PROTOCOL,
            "encoder": "ibm", "family": "spatial", "seed": 13, "threshold": .5,
            "files_sha256": {name: file_hash(self.folder / name) for name in ("head.pt", "encoder_inputs.jsonl")},
            "benchmark_manifest": str(self.manifest), "benchmark_sha256": file_hash(self.manifest)})
        self.pointer = {"deployment_path": str(self.folder / "deployment.json"), "sha256": file_hash(self.folder / "deployment.json"),
                        "name": "synthetic-head-ibm", "kind": "frozen_crater_deployment"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temporary.cleanup()

    def imported(self):
        response = self.client.post("/head-deployments", json=self.pointer)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def payload(self, **overrides):
        ids = self.imported()
        return {"model_name": self.pointer["name"], "dataset_id": ids["dataset_id"], "image_count": 2, "split": "test",
                "include_clean": True, "distortions": [], "prompt": "Segment publisher catalog crater polygons.", "allowed_answers": [], **overrides}

    def completed(self, **overrides):
        response = self.client.post("/vit/runs", json=self.payload(**overrides))
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()["id"]
        with Session(self.app.state.engine) as session:
            prepare_bundle(session.get(Run, run_id))
        self.assertTrue(run_selected(self.settings, run_id, ("frozen_crater",), mock_mode=True))
        self.assertEqual(self.client.get(f"/runs/{run_id}").json()["status"], "completed")
        return run_id

    def test_import_is_atomic_idempotent_and_no_training(self):
        first, second = self.imported(), self.imported()
        self.assertEqual(first, second)
        self.assertFalse(first["training_started"])
        model = self.client.get(f"/model-configs/{first['model_id']}").json()
        self.assertEqual(model["backend"], "frozen_crater")
        self.assertEqual(self.client.get("/runs").json(), [])
        self.assertEqual(len(self.client.get("/dataset-configs").json()), 1)

    def test_changed_deployment_and_weights_rejected(self):
        (self.folder / "head.pt").write_bytes(b"corrupted")
        response = self.client.post("/head-deployments", json=self.pointer)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.get("/model-configs").json(), [])

    def test_ground_truth_or_training_index_rejected(self):
        for change in ({"mask": "forbidden"}, {"split": "train"}):
            with self.subTest(change=change):
                original = self.inputs[0].copy()
                self.inputs[0].update(change)
                self.write_deployment()
                response = self.client.post("/head-deployments", json=self.pointer)
                self.assertEqual(response.status_code, 422, response.text)
                self.inputs[0] = original

    def test_questions_corruptions_and_other_tasks_rejected(self):
        for change in ({"distortions": [["shear"]]}, {"allowed_answers": ["YES", "NO"]}, {"include_clean": False, "distortions": [["radiation"]]}):
            response = self.client.post("/vit/runs", json=self.payload(**change))
            self.assertEqual(response.status_code, 422, response.text)

    def test_image_and_native_hashes_checked_before_inference(self):
        Path(self.targets[1]["native_vis"]).write_bytes(b"changed native")
        response = self.client.post("/vit/runs", json=self.payload())
        # Planning or preparation must catch altered paired inputs, never use RGB as VIS5.
        if response.status_code == 202:
            with Session(self.app.state.engine) as session:
                with self.assertRaisesRegex(ValueError, "native VIS"):
                    prepare_bundle(session.get(Run, response.json()["id"]))
        else:
            self.assertEqual(response.status_code, 422)

    def test_real_request_contract_has_no_targets_and_contains_native_input(self):
        response = self.client.post("/vit/runs", json=self.payload())
        self.assertEqual(response.status_code, 202, response.text)
        with Session(self.app.state.engine) as session:
            requests, records, _ = prepare_bundle(session.get(Run, response.json()["id"]))
        for request in requests:
            self.assertEqual(request.modality_paths[0][0], "vis")
            self.assertNotIn("mask", json.dumps(request.__dict__))
            self.assertNotIn("reference_status", json.dumps(request.__dict__))
            self.assertEqual(request.allowed_answers, ())

    def test_mock_queue_results_maps_and_scene_metrics(self):
        run_id = self.completed()
        rows = self.client.get(f"/runs/{run_id}/results").json()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["mock"] and r["map_url"] and r["output_type"] == TASK for r in rows))
        self.assertEqual(self.client.get(rows[0]["map_url"]).status_code, 200)
        response = self.client.post(f"/runs/{run_id}/metrics")
        self.assertEqual(response.status_code, 200, response.text)
        condition = response.json()["metrics"]["conditions"]["clean"]
        self.assertEqual(condition["metric_family"], TASK)
        self.assertAlmostEqual(condition["quality"]["crater_iou"], .5)
        self.assertAlmostEqual(condition["quality"]["crater_dice"], .5)
        self.assertNotIn("accuracy", condition["quality"])

    def test_map_tampering_refuses_cached_evaluation(self):
        run_id = self.completed()
        self.assertEqual(self.client.post(f"/runs/{run_id}/metrics").status_code, 200)
        row = self.client.get(f"/runs/{run_id}/results").json()[0]
        Path(row["artifacts"]["map_directory"], row["artifacts"]["label_png"]).write_bytes(b"changed")
        self.assertEqual(self.client.post(f"/runs/{run_id}/metrics").status_code, 422)

    def test_matched_mask_runs_have_paired_scene_interval_without_qa_metrics(self):
        first = self.completed()
        self.pointer["name"] = "synthetic-head-clip"
        second = self.completed(reuse_run_id=first)
        for run in (first, second):
            self.assertEqual(self.client.post(f"/runs/{run}/metrics").status_code, 200)
        response = self.client.post("/comparisons", json={"run_ids": [first, second]})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["mock"])
        self.assertTrue(result["quality_comparable"])
        self.assertEqual(result["contrasts"][0]["metric"], "crater_iou")
        self.assertEqual(result["contrasts"][0]["paired_interval"]["interval"], [0., 0.])

    def test_private_heads_service_runs_selected_mock_job(self):
        response = self.client.post("/vit/runs", json=self.payload())
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()["id"]
        with Session(self.app.state.engine) as session:
            prepare_bundle(session.get(Run, run_id))
        headers = {"Authorization": "Bearer " + self.settings.service_token}
        with TestClient(create_service("heads", self.settings)) as service:
            health = service.get("/health", headers=headers)
            self.assertEqual(health.json()["backends"], ["frozen_crater"])
            submit = service.post("/jobs", headers=headers, json={"run_id": run_id, "protocol_sha256": response.json()["protocol_sha256"]})
            self.assertEqual(submit.status_code, 202, submit.text)
            for _ in range(150):
                state = service.get(f"/jobs/{run_id}", headers=headers).json()
                if state["status"] in {"completed", "blocked", "failed"}:
                    break
                time.sleep(.05)
            self.assertEqual(state["status"], "completed", state)

    def test_palette_preserves_class_ids_and_validation_threshold(self):
        from benchmark_services.common.segmentation import save_crater_maps
        # Sigmoid(0)=.5; threshold .6 must yield background, NOT implicit .5/argmax.
        result = save_crater_maps(self.folder / "maps", "r", np.zeros((512, 512), np.float32), .6)
        with Image.open(Path(result["map_directory"]) / result["label_png"]) as image:
            self.assertEqual(image.mode, "P")
            self.assertEqual(set(np.unique(np.asarray(image))), {0})

    def test_api_and_service_imports_do_not_load_torch_or_heads(self):
        script = "import api.main, benchmark_services.heads.app,sys; assert 'torch' not in sys.modules; assert 'benchmark_services.heads.adapter' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
