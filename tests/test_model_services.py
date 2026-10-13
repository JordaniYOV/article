"""CPU-only API -> service -> real job subprocess -> persisted fixture results."""

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
    from sqlmodel import Session
    from api.main import create_app
    from api.dispatcher import Dispatcher
    from api.service_client import ServiceClient, ServiceRejected, ServiceUnavailable
    from benchmark_services.common.http_app import create_service
    from benchmark_services.common.settings import Settings
    from benchmark_services.common.store import Run
    from benchmark_services.common.files import file_hash
    from benchmark_services.common.files import atomic_json
    from benchmark_services.common.execution import run_selected
    from benchmark_services.common.preparation import prepare_bundle
    from benchmark_services.common.locks import gpu_busy
except ImportError:
    TestClient = None


class InProcessClient(ServiceClient if TestClient is not None else object):
    def __init__(self, settings, clients):
        super().__init__(settings)
        self.clients, self.submissions = clients, []

    def request(self, service, method, path, payload=None):
        if service not in self.clients:
            raise ServiceUnavailable("Fixture service offline")
        if method == "POST" and path == "/jobs":
            self.submissions.append((service, payload))
        response = self.clients[service].request(method, path, json=payload,
            headers={"Authorization": "Bearer " + self.settings.service_token})
        if response.status_code >= 400:
            raise ServiceRejected(response.status_code, response.text)
        return response.json()


@unittest.skipIf(TestClient is None, "Install requirements/api.txt")
class ModelServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.patches = [patch("benchmark_services.common.settings.WORKSPACE_ROOT", self.workspace),
                        patch("api.main.WORKSPACE_ROOT", self.workspace)]
        for value in self.patches:
            value.start()
        self.settings = Settings(database_path=self.workspace / ".state" / "services.sqlite3",
            service_token="cpu-fixture-token-not-a-secret", service_mock=True,
            worker_poll_seconds=.02, _env_file=None)
        self.clients, self.apps = {}, {}
        for name in ("spacellava", "ibm"):
            app = create_service(name, self.settings)
            self.apps[name] = app
            self.clients[name] = TestClient(app)
            self.clients[name].__enter__()
        self.api_app = create_app(self.settings)
        self.api = TestClient(self.api_app)
        self.api.__enter__()
        self.transport = InProcessClient(self.settings, self.clients)
        self.dispatcher = Dispatcher(self.api_app.state.engine, self.settings, self.transport)
        folder = self.workspace / "data_orbital"
        folder.mkdir()
        image, mask = folder / "image.png", folder / "mask.png"
        Image.fromarray(np.full((256, 256), 64, dtype=np.uint8)).save(image)
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(mask)
        manifest = folder / "manifest.jsonl"
        manifest.write_text(json.dumps({"sample_id": "fixture", "image": str(image),
            "image_sha256": file_hash(image), "source_split": "test", "scene_group_id": "fixture-scene",
            "mask": str(mask), "mask_sha256": file_hash(mask), "classes": ["Background", "IMP"],
            "mask_semantics": "imp", "answer": "YES"}) + "\n", encoding="utf-8")
        response = self.api.post("/dataset-configs", json={"name": "fixture", "planet": "moon",
            "task_id": "imp_segmentation", "manifest_path": str(manifest),
            "preprocessing": {"input_protocol": "curated_imp_png_reflectance_0_0.12_v1"}})
        self.assertEqual(response.status_code, 201, response.text)
        self.dataset = response.json()
        for name, backend in (("SpaceLLaVA", "spacellava"), ("IBM", "ibm_imp")):
            response = self.api.post("/model-configs", json={"name": name, "model_id": "fixture",
                "backend": backend, "enabled": True,
                "options": {"input_protocol": "curated_imp_png_reflectance_0_0.12_v1"}})
            self.assertEqual(response.status_code, 201, response.text)

    def tearDown(self):
        self.dispatcher.close()
        self.api.__exit__(None, None, None)
        for client in self.clients.values():
            client.__exit__(None, None, None)
        for value in reversed(self.patches):
            value.stop()
        self.temporary.cleanup()

    def start(self, model="SpaceLLaVA"):
        response = self.api.post("/vit/runs" if model == "IBM" else "/vlm/runs", json={
            "model_name": model, "dataset_id": self.dataset["id"], "image_count": 1,
            "distortions": [], "prompt": "Describe visible evidence; never use labels.",
            "allowed_answers": ["YES", "NO"]})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()

    def submit(self, service, run):
        with Session(self.api_app.state.engine) as session:
            prepare_bundle(session.get(Run, run["id"]))
        return self.transport.request(service, "POST", "/jobs",
            {"run_id": run["id"], "protocol_sha256": run["protocol_sha256"]})

    def wait_job(self, service, run):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = self.transport.request(service, "GET", "/jobs/" + run["id"])
            if not status["worker_alive"] and status["status"] in {"completed", "blocked", "failed", "interrupted", "cancelled"}:
                log = Path(run["output_dir"]) / "service-worker.log"
                self.assertEqual(status["status"], "completed",
                    (status, log.read_text(errors="replace") if log.exists() else "no log"))
                return status
            time.sleep(.02)
        self.fail("Fixture subprocess did not finish")

    def test_both_models_route_through_http_and_fresh_processes_without_gpu(self):
        first, second = self.start(), self.start("IBM")
        self.assertTrue(self.dispatcher.tick())
        self.assertEqual(self.transport.submissions[0][0], "spacellava")
        self.assertEqual(self.api.get(second["status_url"]).json()["status"], "queued")
        self.wait_job("spacellava", first)
        self.assertFalse(gpu_busy())
        self.assertTrue(self.dispatcher.tick())
        self.wait_job("ibm", second)
        for run in (first, second):
            rows = self.api.get(run["results_url"]).json()
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["mock"])
            metadata = json.loads((Path(run["output_dir"]) / "run.json").read_text())
            self.assertTrue(metadata["mock"])
            self.assertEqual(metadata["execution_mode"], "mock")
            requests = (Path(run["output_dir"]) / "requests.jsonl").read_text()
            self.assertNotIn('"mask"', requests)
            self.assertNotIn('"answer"', requests)
            score = self.api.post(f"/runs/{run['id']}/metrics")
            self.assertEqual(score.status_code, 200, score.text)
            self.assertTrue(score.json()["mock"])
            self.assertTrue(self.api.get(run["report_url"]).json()["mock"])
        for _, payload in self.transport.submissions:
            self.assertEqual(set(payload), {"run_id", "protocol_sha256"})
        # Different model environments must receive the very same prepared bytes.
        a = self.api.get(first["results_url"]).json()[0]
        b = self.api.get(second["results_url"]).json()[0]
        self.assertEqual(a["image_paths"], b["image_paths"])
        self.assertEqual(a["run_metadata"]["input_sha256"], b["run_metadata"]["input_sha256"])

    def test_authentication_routing_and_payload_are_fail_closed(self):
        run = self.start("IBM")
        self.assertEqual(self.clients["spacellava"].get("/health").status_code, 401)
        with self.assertRaises(ServiceRejected) as caught:
            self.submit("spacellava", run)
        self.assertEqual(caught.exception.status, 422)
        headers = {"Authorization": "Bearer " + self.settings.service_token}
        response = self.clients["ibm"].post("/jobs", headers=headers, json={
            "run_id": run["id"], "protocol_sha256": run["protocol_sha256"], "answer": "YES"})
        self.assertEqual(response.status_code, 422)
        self.assertFalse(gpu_busy())

    def test_duplicate_delivery_does_not_launch_twice(self):
        run = self.start()
        original = subprocess.Popen
        with patch("benchmark_services.common.controller.subprocess.Popen", wraps=original) as popen:
            self.submit("spacellava", run)
            self.submit("spacellava", run)
            self.wait_job("spacellava", run)
            self.submit("spacellava", run)
            self.assertEqual(popen.call_count, 1)
        self.assertEqual(len(self.api.get(run["results_url"]).json()), 1)

    def test_mock_results_cannot_resume_as_real_without_new_run(self):
        run = self.start()
        self.submit("spacellava", run)
        self.wait_job("spacellava", run)
        with Session(self.api_app.state.engine) as session:
            row = session.get(Run, run["id"])
            row.status = "interrupted"
            session.add(row)
            session.commit()
        self.assertEqual(self.api.post(f"/runs/{run['id']}/resume").status_code, 202)
        with patch("benchmark_services.common.execution.create_adapter") as create:
            run_selected(self.settings, run["id"], ("spacellava",), mock_mode=False)
            create.assert_not_called()
        status = self.api.get(run["status_url"]).json()
        self.assertEqual(status["status"], "blocked")
        self.assertIn("resume refused", status["error_message"])

    def test_unavailable_model_does_not_block_the_other_queue(self):
        first, second = self.start(), self.start("IBM")
        self.transport.clients = {"ibm": self.clients["ibm"]}
        self.assertTrue(self.dispatcher.tick())
        self.wait_job("ibm", second)
        self.assertEqual(self.api.get(first["status_url"]).json()["phase"], "waiting_service")

    def test_cross_service_reservation_and_cancel_release_on_process_exit(self):
        first, second = self.start(), self.start("IBM")

        class HoldingProcess:
            code = None
            def poll(self): return self.code
            def terminate(self): self.code = -15
            def kill(self): self.code = -9
            def wait(self, timeout=None): return self.code

        with patch("benchmark_services.common.controller.subprocess.Popen", return_value=HoldingProcess()):
            self.submit("spacellava", first)
            self.assertTrue(gpu_busy())
            with self.assertRaises(ServiceRejected) as caught:
                self.submit("ibm", second)
            self.assertEqual(caught.exception.status, 409)
            self.transport.request("spacellava", "POST", f"/jobs/{first['id']}/cancel")
            self.assertTrue(self.api.get(first["status_url"]).json()["cancel_requested"])
            self.apps["spacellava"].state.controller.terminate("Fixture stopped")
            self.assertFalse(gpu_busy())

    def test_worker_exit_preserves_recoverable_state(self):
        run = self.start()

        class ExitedProcess:
            def poll(self): return 17

        with patch("benchmark_services.common.controller.subprocess.Popen", return_value=ExitedProcess()):
            self.submit("spacellava", run)
        status = self.api.get(run["status_url"]).json()
        self.assertEqual(status["status"], "blocked")
        self.assertIn("exited (17)", status["error_message"])
        self.assertFalse(gpu_busy())

    def test_timeout_kills_only_owned_process_and_releases_reservation(self):
        run = self.start()

        class HoldingProcess:
            code = None
            def poll(self): return self.code
            def terminate(self): self.code = -15
            def kill(self): self.code = -9
            def wait(self, timeout=None): return self.code

        controller = self.apps["spacellava"].state.controller
        with patch("benchmark_services.common.controller.subprocess.Popen", return_value=HoldingProcess()):
            self.submit("spacellava", run)
            self.settings.service_max_job_seconds = .01
            controller.started = time.monotonic() - 1
            controller.reap()
        status = self.api.get(run["status_url"]).json()
        self.assertEqual(status["status"], "interrupted")
        self.assertIn("timeout", status["error_message"])
        self.assertFalse(gpu_busy())

    def test_service_identity_mismatch_is_rejected(self):
        with patch.object(self.transport, "request", return_value={"service": "spacellava", "store_id": "wrong"}):
            with self.assertRaises(ServiceUnavailable):
                self.transport.health("spacellava")

    def test_fast_resume_is_not_overwritten_when_the_previous_child_exits(self):
        run = self.start()

        class HoldingProcess:
            code = None
            def poll(self): return self.code
            def terminate(self): self.code = -15
            def kill(self): self.code = -9
            def wait(self, timeout=None): return self.code

        process = HoldingProcess()
        controller = self.apps["spacellava"].state.controller
        with patch("benchmark_services.common.controller.subprocess.Popen", return_value=process):
            self.submit("spacellava", run)
        with controller.guard:
            atomic_json(Path(run["output_dir"]) / "service-attempt.json",
                {"attempt_id": controller.attempt_id, "claimed": True})
            with Session(self.api_app.state.engine) as session:
                row = session.get(Run, run["id"])
                row.status = "blocked"
                session.add(row)
                session.commit()
            self.assertEqual(self.api.post(f"/runs/{run['id']}/resume").status_code, 202)
            process.code = 0
            controller.reap()
        self.assertEqual(self.api.get(run["status_url"]).json()["status"], "queued")
        self.assertFalse(gpu_busy())

    def test_selected_worker_consumes_bundle_without_rendering_in_its_environment(self):
        run = self.start()
        with Session(self.api_app.state.engine) as session:
            prepare_bundle(session.get(Run, run["id"]))
        with patch("benchmark_services.common.execution.prepare_bundle", side_effect=AssertionError("Do not render here")):
            run_selected(self.settings, run["id"], ("spacellava",), mock_mode=True)
        self.assertEqual(self.api.get(run["status_url"]).json()["status"], "completed")

    def test_changed_shared_image_blocks_before_model_loading(self):
        run = self.start()
        with Session(self.api_app.state.engine) as session:
            requests, _, _ = prepare_bundle(session.get(Run, run["id"]))
        Path(requests[0].image_paths[0]).write_bytes(b"changed fixture input")
        with patch("benchmark_services.common.execution.create_adapter") as create:
            run_selected(self.settings, run["id"], ("spacellava",), mock_mode=True)
            create.assert_not_called()
        status = self.api.get(run["status_url"]).json()
        self.assertEqual(status["status"], "blocked")
        self.assertIn("Prepared image changed", status["error_message"])


class ArchitectureTests(unittest.TestCase):
    def test_api_and_service_imports_do_not_load_models(self):
        if TestClient is None:
            self.skipTest("Install requirements/api.txt")
        command = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
            "import api.main, benchmark_services.spacellava.app, benchmark_services.ibm.app; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
            "assert 'benchmark_services.ibm.adapter' not in sys.modules; "
            "assert 'benchmark_services.spacellava.adapter' not in sys.modules")
        result = subprocess.run([sys.executable, "-I", "-c", command], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
