"""HTTP and SQLite integration using temporary databases and fixture outputs."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from fastapi.testclient import TestClient
    from sqlalchemy import inspect
    from api.main import create_app
    from api.settings import Settings
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "Install the api and api-test extras")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.settings = Settings(database_path=Path(self.directory.name) / "api.sqlite3", _env_file=None)
        self.application = create_app(self.settings)
        self.client = TestClient(self.application)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.directory.cleanup()

    def configs(self):
        model_payload = {"name": "SpaceLLaVA", "model_id": "fixture/SpaceLLaVA", "backend": "spacellava"}
        dataset_payload = {"name": "fixture", "planet": "moon", "task_id": "imp_segmentation",
            "manifest_path": "data_orbital/fixture/manifest.jsonl"}
        model = self.client.post("/model-configs", json=model_payload)
        dataset = self.client.post("/dataset-configs", json=dataset_payload)
        self.assertEqual(model.status_code, 201, model.text)
        self.assertEqual(dataset.status_code, 201, dataset.text)
        return model.json(), dataset.json(), model_payload, dataset_payload

    def result(self, model, dataset, *, request_id="r1", **overrides):
        payload = {"model_config_id": model["id"], "dataset_config_id": dataset["id"],
            "run_id": "fixture", "request_id": request_id, "status": "ok", "raw_response": "fixture answer",
            "elapsed_seconds": .2, "mock": True, **overrides}
        response = self.client.post("/model-results", json=payload)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def analysis(self, model, dataset, result_ids, **overrides):
        return self.client.post("/metric-analyses", json={"model_config_id": model["id"],
            "dataset_config_id": dataset["id"], "run_id": "fixture", "task_id": "imp_segmentation",
            "name": "fixture-runtime", "method_version": "fixture-v1", "result_ids": result_ids,
            "metrics": {"n": len(result_ids)}, **overrides})

    def test_health_five_tables_and_persistence(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.assertEqual(set(inspect(self.application.state.engine).get_table_names()),
                         {"model_configs", "dataset_configs", "model_results", "metric_analyses", "runs"})
        model, _, _, _ = self.configs()
        with TestClient(create_app(self.settings)) as other:
            self.assertEqual(other.get(f"/model-configs/{model['id']}").json()["model_id"], model["model_id"])

    def test_config_crud_duplicates_and_freezing(self):
        model, dataset, model_payload, dataset_payload = self.configs()
        self.assertEqual(self.client.post("/model-configs", json=model_payload).status_code, 409)
        changed = self.client.put(f"/model-configs/{model['id']}", json={**model_payload, "seed": 123})
        self.assertEqual(changed.status_code, 200, changed.text)
        self.assertNotEqual(changed.json()["config_sha256"], model["config_sha256"])
        self.result(model, dataset)
        self.assertEqual(self.client.put(f"/model-configs/{model['id']}", json=model_payload).status_code, 409)
        self.assertEqual(self.client.delete(f"/model-configs/{model['id']}").status_code, 409)
        self.assertEqual(self.client.put(f"/dataset-configs/{dataset['id']}", json=dataset_payload).status_code, 409)
        self.assertEqual(self.client.delete(f"/dataset-configs/{dataset['id']}").status_code, 409)
        fresh = self.client.post("/model-configs", json={**model_payload, "version": "v2"}).json()
        self.assertEqual(self.client.delete(f"/model-configs/{fresh['id']}").status_code, 204)

    def test_validation_foreign_keys_paths_and_result_uniqueness(self):
        model, dataset, _, dataset_payload = self.configs()
        invalid = {"model_config_id": 999, "dataset_config_id": dataset["id"], "run_id": "fixture",
            "request_id": "r", "status": "ok", "elapsed_seconds": .1}
        self.assertEqual(self.client.post("/model-results", json=invalid).status_code, 404)
        invalid["model_config_id"] = model["id"]
        self.assertEqual(self.client.post("/model-results", json={**invalid, "elapsed_seconds": "nan"}).status_code, 422)
        self.assertEqual(self.client.post("/model-results", json={**invalid, "elapsed_seconds": True}).status_code, 422)
        self.assertEqual(self.client.post("/model-results", json={**invalid, "condition_id": "unknown"}).status_code, 422)
        self.assertEqual(self.client.post("/model-results", json={**invalid, "answer": "leaked"}).status_code, 422)
        self.assertEqual(self.client.post("/dataset-configs", json={**dataset_payload,
            "manifest_path": "data_rover/forbidden.jsonl"}).status_code, 422)
        result = self.result(model, dataset)
        duplicate = {key: value for key, value in result.items() if key not in {"id", "created_at"}}
        self.assertEqual(self.client.post("/model-results", json=duplicate).status_code, 409)
        self.assertEqual(len(self.client.get("/model-results", params={"run_id": "fixture"}).json()), 1)
        self.assertEqual(self.client.get("/model-results", params={"limit": 501}).status_code, 422)

    def test_analysis_scope_and_result_delete_protection(self):
        model, dataset, _, _ = self.configs()
        first = self.result(model, dataset)
        second = self.result(model, dataset, request_id="r2", mock=False)
        self.assertEqual(self.analysis(model, dataset, [first["id"], second["id"]]).status_code, 422)
        self.assertEqual(self.analysis(model, dataset, [first["id"]], run_id="other").status_code, 422)
        self.assertEqual(self.analysis(model, dataset, [999]).status_code, 422)
        response = self.analysis(model, dataset, [first["id"]])
        self.assertEqual(response.status_code, 201, response.text)
        saved = response.json()
        self.assertTrue(saved["mock"])
        self.assertEqual(len(saved["results_sha256"]), 64)
        self.assertEqual(self.client.delete(f"/model-results/{first['id']}").status_code, 409)
        self.assertEqual(self.client.delete(f"/metric-analyses/{saved['id']}").status_code, 204)
        self.assertEqual(self.client.delete(f"/model-results/{first['id']}").status_code, 204)

    def test_default_configuration_import_is_idempotent_and_does_not_infer(self):
        response = self.client.post("/configurations/import-defaults")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["model_config_ids"]), 2)
        self.assertEqual(response.json(), self.client.post("/configurations/import-defaults").json())
        self.assertEqual(self.client.get("/model-results").json(), [])

    def test_saved_run_import_is_atomic_idempotent_and_bound_to_dataset(self):
        from planetary_vlm.contracts import ModelRequest
        from planetary_vlm.io import write_json, write_jsonl, records_fingerprint
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            inputs = workspace / "data_orbital" / "inputs"
            requests = [asdict(ModelRequest("s1__clean", (str(inputs / "image.png"),), "Q?", (), 128))]
            digest = records_fingerprint(requests)
            write_jsonl(inputs / "requests.jsonl", requests)
            write_jsonl(inputs / "provenance.jsonl", [{"request_id": "s1__clean", "source_sample_id": "s1",
                "scene_group_id": "scene", "source_split": "test", "condition": "clean"}])
            folder = workspace / "outputs" / "SpaceLLaVA" / "fixture"
            write_json(folder / "run.json", {"model_id": "fixture/SpaceLLaVA", "backend": "spacellava",
                "run_id": "fixture", "seed": 42, "options": {}, "mock": True, "request_manifest_sha256": digest})
            write_jsonl(folder / "requests.jsonl", requests)
            predictions = [{"model_id": "fixture/SpaceLLaVA", "run_id": "fixture", "request_id": "s1__clean",
                "status": "ok", "raw_response": "fixture answer", "elapsed_seconds": .2, "mock": True,
                "request_manifest_sha256": digest}]
            write_jsonl(folder / "predictions.jsonl", predictions)
            with patch("api.main.WORKSPACE_ROOT", workspace), \
                 patch("api.settings.WORKSPACE_ROOT", workspace):
                model = self.client.post("/model-configs", json={"name": "SpaceLLaVA",
                    "model_id": "fixture/SpaceLLaVA", "backend": "spacellava"}).json()
                dataset = self.client.post("/dataset-configs", json={"name": "fixture", "planet": "moon",
                    "task_id": "imp_segmentation", "manifest_path": "data_orbital/manifest.jsonl",
                    "requests_path": "data_orbital/inputs/requests.jsonl"}).json()
                payload = {"model_config_id": model["id"], "dataset_config_id": dataset["id"], "run_id": "fixture"}
                first = self.client.post("/imports/results", json=payload)
                self.assertEqual(first.status_code, 200, first.text)
                self.assertEqual(first.json()["inserted"], 1)
                second = self.client.post("/imports/results", json=payload)
                self.assertEqual(second.status_code, 200, second.text)
                self.assertEqual(second.json()["skipped"], 1)
                # A later invalid row must roll back the earlier pending insert.
                session_results = self.client.get("/model-results").json()
                self.assertEqual(self.client.delete(f"/model-results/{session_results[0]['id']}").status_code, 204)
                bad = {**predictions[0], "request_id": "foreign-request"}
                (folder / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in [*predictions, bad]))
                self.assertEqual(self.client.post("/imports/results", json=payload).status_code, 422)
                self.assertEqual(self.client.get("/model-results").json(), [])


if __name__ == "__main__":
    unittest.main()
