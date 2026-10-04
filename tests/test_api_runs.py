"""HTTP -> queue -> fixture adapters -> SQLite/files -> metric integration."""

from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    from PIL import Image
    from fastapi.testclient import TestClient
    from sqlmodel import Session, select
    from api.main import create_app
    from api.models import Run, ModelResult
    from api.settings import Settings
    from api.worker import run_once, gpu_lock
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "Install api and api-test extras")
class RunApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.patch_paths = patch("api.settings.WORKSPACE_ROOT", self.workspace)
        self.patch_paths.start()
        self.settings = Settings(database_path=self.workspace / ".state" / "api.sqlite3", _env_file=None)
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.patch_paths.stop()
        self.temporary.cleanup()

    def archive(self, *, labels=True, imp=False):
        stream = BytesIO()
        rows = []
        with zipfile.ZipFile(stream, "w") as archive:
            for i in range(3):
                size = 256 if imp else 16
                pixels = np.arange(size * size, dtype=np.uint8).reshape(size, size)
                image = BytesIO()
                Image.fromarray(pixels).save(image, format="PNG")
                name = f"images/{i}.png"
                archive.writestr(name, image.getvalue())
                row = {"sample_id": f"s{i}", "image": name, "source_split": "test",
                    "scene_group_id": f"scene{i}"}
                if labels:
                    row["answer"] = "YES"
                if imp:
                    mask = BytesIO()
                    Image.fromarray(np.zeros((size, size), dtype=np.uint8)).save(mask, format="PNG")
                    archive.writestr(f"masks/{i}.png", mask.getvalue())
                    row.update(mask=f"masks/{i}.png", classes=["Background", "IMP"], mask_semantics="imp")
                rows.append(row)
            archive.writestr("manifest.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
        return stream.getvalue()

    def dataset(self, *, labels=True, imp=False):
        metadata = {"name": "fixture", "planet": "moon", "task_id": "imp_segmentation" if imp else "presence"}
        if imp:
            metadata["preprocessing"] = {"input_protocol": "curated_imp_png_reflectance_0_0.12_v1"}
        response = self.client.post("/datasets/upload", files={"file": ("dataset.zip", self.archive(labels=labels, imp=imp), "application/zip")},
            data={"metadata": json.dumps(metadata)})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def model(self, name="SpaceLLaVA", *, backend="mock", enabled=True, options=None):
        response = self.client.post("/model-configs", json={"name": name, "model_id": "fixture/model",
            "backend": backend, "enabled": enabled, "options": options or {"default_response": "YES"}})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def start(self, dataset, *, name="SpaceLLaVA", endpoint="vlm", **kwargs):
        body = {"model_name": name, "dataset_id": dataset["id"], "image_count": 2,
            "split": "test", "seed": 123, "prompt": "Are features present? Answer YES or NO.",
            "allowed_answers": ["YES", "NO"], "include_clean": True,
            "distortions": [["shear"], ["radiation"], ["lens_flare"], ["hard_shadows"],
                ["shear", "hard_shadows", "lens_flare", "radiation"]], **kwargs}
        response = self.client.post(f"/{endpoint}/runs", json=body)
        self.assertEqual(response.status_code, 202, response.text)
        return response.json(), body

    def process(self, run):
        self.assertTrue(run_once(self.settings))
        status = self.client.get(run["status_url"]).json()
        self.assertEqual(status["status"], "completed", status)
        return status

    def test_full_pipeline_all_effects_report_and_comparison(self):
        dataset = self.dataset()
        first_model = self.model()
        first, body = self.start(dataset)
        self.assertEqual(first["total"], 12)
        self.assertEqual(first["completed"], 0)
        self.assertEqual(self.client.post(f"/runs/{first['id']}/metrics").status_code, 409)
        # A queued configuration is already frozen, before the first response.
        model_payload = {key: value for key, value in first_model.items() if key not in {"id", "created_at", "updated_at", "config_sha256"}}
        self.assertEqual(self.client.put(f"/model-configs/{first_model['id']}", json=model_payload).status_code, 409)
        self.process(first)
        results = self.client.get(first["results_url"]).json()
        self.assertEqual(len(results), 12)
        self.assertTrue(all(row["mock"] for row in results))
        self.assertEqual(len(self.client.get(first["results_url"], params={"condition": "shear"}).json()), 2)
        self.assertEqual(self.client.get(results[0]["image_url"]).status_code, 200)
        requests = [json.loads(line) for line in (Path(first["output_dir"]) / "requests.jsonl").read_text().splitlines()]
        self.assertTrue(all("answer" not in row and "mask" not in row for row in requests))
        self.assertEqual(len({row["source_sample_id"] for row in results}), 2)
        scored = self.client.post(f"/runs/{first['id']}/metrics")
        self.assertEqual(scored.status_code, 200, scored.text)
        self.assertTrue(scored.json()["mock"])
        self.assertEqual(scored.json()["metrics"]["conditions"]["clean"]["quality"]["accuracy"], 1)
        self.assertEqual(self.client.post(f"/runs/{first['id']}/metrics").json()["id"], scored.json()["id"])
        self.assertEqual(self.client.get(first["report_url"]).json()["metrics_status"], "calculated")
        self.assertTrue((Path(first["output_dir"]) / "metrics.json").is_file())
        self.assertEqual(self.client.delete(f"/model-results/{results[0]['id']}").status_code, 409)
        self.assertEqual(self.client.post("/model-results", json={k: v for k, v in results[0].items()
            if k not in {"id", "created_at", "image_url", "map_url"}}).status_code, 409)
        self.model("NASA-IBM-Lunar-Foundation-Model", options={"default_response": "NO"})
        second, _ = self.start(dataset, name="NASA-IBM-Lunar-Foundation-Model", reuse_run_id=first["id"])
        self.process(second)
        self.client.post(f"/runs/{second['id']}/metrics")
        comparison = self.client.post("/comparisons", json={"run_ids": [first["id"], second["id"]]})
        self.assertEqual(comparison.status_code, 200, comparison.text)
        self.assertTrue(comparison.json()["quality_comparable"])
        self.assertEqual(comparison.json()["contrasts"][0]["right_minus_left"], -1)
        self.assertEqual(comparison.json()["contrasts"][0]["paired_interval"]["n_groups"], 2)
        changed = self.client.post("/vlm/runs", json={**body, "seed": 99, "reuse_run_id": first["id"]})
        self.assertEqual(changed.status_code, 422)
        self.assertEqual(self.client.get("/datasets").json()[0]["id"], dataset["id"])

    def test_cancel_resume_recovery_and_no_duplicate_responses(self):
        dataset = self.dataset()
        self.model()
        run, _ = self.start(dataset, distortions=[])
        self.assertEqual(self.client.post(f"/runs/{run['id']}/cancel").json()["status"], "cancelled")
        self.assertFalse(run_once(self.settings))
        self.assertEqual(self.client.post(f"/runs/{run['id']}/resume").json()["status"], "queued")
        from planetary_vlm.models.mock import MockAdapter
        original = MockAdapter.predict
        calls = []
        def stop_after_first(adapter, request):
            calls.append(request.request_id)
            response = original(adapter, request)
            self.client.post(f"/runs/{run['id']}/cancel")
            return response
        with patch.object(MockAdapter, "predict", stop_after_first):
            self.assertTrue(run_once(self.settings))
        status = self.client.get(run["status_url"]).json()
        self.assertEqual(status["status"], "cancelled")
        self.assertEqual(status["completed"], 1)
        # Simulate a worker crash after committing SQLite but before export.
        (Path(run["output_dir"]) / "predictions.jsonl").write_text("")
        self.client.post(f"/runs/{run['id']}/resume")
        self.process(run)
        results = self.client.get(run["results_url"]).json()
        self.assertEqual(len(results), 2)
        self.assertEqual(len({row["request_id"] for row in results}), 2)
        self.assertEqual(len((Path(run["output_dir"]) / "predictions.jsonl").read_text().splitlines()), 2)
        with Session(self.app.state.engine) as session:
            saved = session.get(Run, run["id"])
            saved.status = "running"
            session.add(saved)
            session.commit()
        self.assertFalse(run_once(self.settings))
        self.assertEqual(self.client.get(run["status_url"]).json()["status"], "interrupted")

    def test_resume_refuses_changed_inputs_and_gpu_lock_is_exclusive(self):
        dataset = self.dataset()
        self.model()
        run, _ = self.start(dataset, distortions=[])
        from planetary_vlm.models.mock import MockAdapter
        original = MockAdapter.predict
        def stop(adapter, request):
            self.client.post(f"/runs/{run['id']}/cancel")
            return original(adapter, request)
        with patch.object(MockAdapter, "predict", stop):
            run_once(self.settings)
        with Session(self.app.state.engine) as session:
            source = session.get(Run, run["id"]).samples[0]
        Path(source["image"]).write_bytes(b"changed")
        self.client.post(f"/runs/{run['id']}/resume")
        run_once(self.settings)
        self.assertEqual(self.client.get(run["status_url"]).json()["status"], "blocked")
        self.assertEqual(len(self.client.get(run["results_url"]).json()), 1)
        with gpu_lock(self.settings):
            with self.assertRaises(OSError):
                run_once(self.settings)

    def test_unlabelled_images_and_invalid_outputs_are_not_scientific_scores(self):
        dataset = self.dataset(labels=False)
        self.model(options={"default_response": "YES but prose"})
        run, _ = self.start(dataset, distortions=[])
        self.process(run)
        score = self.client.post(f"/runs/{run['id']}/metrics").json()["metrics"]
        self.assertIsNone(score["conditions"]["clean"]["quality"])
        self.assertEqual(score["conditions"]["clean"]["runtime"]["n_ok"], 2)

    def test_vit_native_maps_and_aligned_geometry(self):
        dataset = self.dataset(imp=True)
        self.model("NASA-IBM-Lunar-Foundation-Model", backend="ibm_imp",
            options={"input_protocol": "curated_imp_png_reflectance_0_0.12_v1"})
        from planetary_vlm.contracts import ModelResponse
        from planetary_vlm.models.ibm_imp import save_maps
        class FixtureHead:
            def __init__(self, options):
                self.options = options
            def initialize(self):
                return {"fixture": True}
            def predict(self, request):
                assert request.modality_paths == ()
                logits = np.zeros((2, 256, 256), dtype=np.float32)
                logits[0] = 2
                artifacts = save_maps(self.options["artifact_dir"], request.request_id, logits)
                return ModelResponse(request.request_id, json.dumps(artifacts), "ok", .1)
            def close(self):
                pass
        run, _ = self.start(dataset, endpoint="vit", name="NASA-IBM-Lunar-Foundation-Model", distortions=[["shear"]])
        with patch("api.worker.create_adapter", side_effect=lambda backend, options: FixtureHead(options)):
            self.process(run)
        response = self.client.post(f"/runs/{run['id']}/metrics")
        self.assertEqual(response.status_code, 200, response.text)
        conditions = response.json()["metrics"]["conditions"]
        self.assertEqual(conditions["shear"]["quality"]["pixel_accuracy"], 1)
        self.assertLess(conditions["shear"]["quality"]["n_evaluated_pixels"], 2 * 256 * 256)
        row = self.client.get(run["results_url"]).json()[0]
        self.assertEqual(self.client.get(row["map_url"]).status_code, 200)
        # Tampered maps must be detected on a repeated metric calculation.
        artifacts = json.loads(row["raw_response"])
        (Path(artifacts["map_directory"]) / artifacts["label_png"]).write_bytes(b"changed")
        self.assertEqual(self.client.post(f"/runs/{run['id']}/metrics").status_code, 422)

    def test_upload_rejections_and_url_import_without_real_download(self):
        metadata = {"name": "bad", "planet": "moon"}
        for name in ("../escape.png", "data_rover/x.png", "C:/escape.png", "script.py", "CON.png"):
            stream = BytesIO()
            with zipfile.ZipFile(stream, "w") as archive:
                archive.writestr(name, b"invalid")
            response = self.client.post("/datasets/upload", files={"file": ("bad.zip", stream.getvalue())},
                data={"metadata": json.dumps(metadata)})
            self.assertEqual(response.status_code, 422, (name, response.text))
        self.assertFalse(list((self.workspace / "data_orbital" / "uploads").iterdir()))
        self.assertEqual(self.client.post("/datasets/from-url", json={**metadata, "url": "http://127.0.0.1/a.zip"}).status_code, 422)
        with patch("api.main.download_zip", return_value=BytesIO(self.archive())):
            response = self.client.post("/datasets/from-url", json={**metadata, "url": "https://example.org/archive.zip"})
        self.assertEqual(response.status_code, 201, response.text)
        response = self.client.post("/datasets/upload", files={"file": ("data.zip", self.archive())},
            data={"metadata": json.dumps(metadata)})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(len(list((self.workspace / "data_orbital" / "uploads").iterdir())), 1)

    def test_protocol_validation_model_kinds_and_missing_dependencies(self):
        dataset = self.dataset()
        self.model(backend="spacellava", options={"local_files_only": True})
        run, body = self.start(dataset, distortions=[])
        run_once(self.settings)
        self.assertEqual(self.client.get(run["status_url"]).json()["status"], "blocked")
        self.assertEqual(self.client.get(run["results_url"]).json(), [])
        for changes in ({"image_count": 100}, {"distortions": [["unknown"]]},
            {"parameters": {"shear": {"max_shift_px": -1}}}, {"include_clean": False, "distortions": []},
            {"distortions": [["shear", "shear"]]}, {"split": "train"}):
            self.assertEqual(self.client.post("/vlm/runs", json={**body, **changes}).status_code, 422)
        self.assertEqual(self.client.post("/vit/runs", json=body).status_code, 422)
        self.assertEqual(self.client.get("/runs/missing").status_code, 404)
        self.assertEqual(self.client.get("/runs/missing/results").status_code, 404)


if __name__ == "__main__":
    unittest.main()
