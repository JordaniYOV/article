"""Real loopback HTTP, three server processes and CPU-only mock job workers."""

import json
import gc
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    from PIL import Image
    import uvicorn
    from benchmark_services.common.files import file_hash
except ImportError:
    uvicorn = None


def free_port():
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return stream.getsockname()[1]


@contextmanager
def server_workspace():
    temporary = tempfile.TemporaryDirectory(prefix="planetary-http-test-")
    try:
        yield temporary.name
    finally:
        # Windows may briefly retain a SQLite/AV handle after terminated servers.
        # Retry only this test-owned directory, with a strict deadline; never ignore
        # a persistent lock or remove anything outside the generated temp root.
        for attempt in range(30):
            try:
                temporary.cleanup()
                break
            except PermissionError:
                if attempt == 29:
                    raise
                gc.collect()
                time.sleep(.1)


@unittest.skipIf(uvicorn is None, "Install requirements/api.txt")
class RealHTTPTests(unittest.TestCase):
    def test_api_dispatches_to_both_servers_over_real_http_without_gpu(self):
        with server_workspace() as temporary:
            root = Path(temporary)
            servers, logs = [], []
            ports = {name: free_port() for name in ("api", "spacellava", "ibm")}
            self.assertEqual(len(set(ports.values())), 3)
            env = dict(os.environ)
            env.update(PLANETARY_WORKSPACE_ROOT=str(root),
                PLANETARY_API_DATABASE_PATH=str(root / ".state" / "http.sqlite3"),
                PLANETARY_API_SERVICE_TOKEN="real-http-fixture-token-not-a-secret",
                PLANETARY_API_DISPATCHER_ENABLED="true", PLANETARY_API_SERVICE_MOCK="true",
                PLANETARY_API_WORKER_POLL_SECONDS="0.05", PLANETARY_API_SERVICE_MAX_JOB_SECONDS="0",
                PLANETARY_API_SPACELLAVA_SERVICE_URL=f"http://127.0.0.1:{ports['spacellava']}",
                PLANETARY_API_IBM_SERVICE_URL=f"http://127.0.0.1:{ports['ibm']}")

            def request(method, path, payload=None, server="api"):
                data = json.dumps(payload).encode() if payload is not None else None
                headers = {"Content-Type": "application/json"}
                if server != "api":
                    headers["Authorization"] = "Bearer " + env["PLANETARY_API_SERVICE_TOKEN"]
                req = Request(f"http://127.0.0.1:{ports[server]}" + path, data=data, headers=headers, method=method)
                with urlopen(req, timeout=3) as response:
                    return json.load(response)

            def wait_until(operation):
                deadline = time.monotonic() + 20
                last = None
                while time.monotonic() < deadline:
                    try:
                        result = operation()
                        if result:
                            return result
                    except (URLError, HTTPError, TimeoutError, ConnectionError) as error:
                        last = error
                    time.sleep(.05)
                details = "\n".join(path.read_text(errors="replace") for path in root.glob("*.log"))
                self.fail(f"HTTP fixture timed out: {last}\n{details}")

            try:
                for name in ("spacellava", "ibm", "api"):
                    target = "api.main:app" if name == "api" else f"benchmark_services.{name}.app:app"
                    log = (root / (name + ".log")).open("wb")
                    logs.append(log)
                    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "--app-dir", str(ROOT / "src"),
                        target, "--host", "127.0.0.1", "--port", str(ports[name]), "--log-level", "warning"],
                        cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
                    servers.append(process)
                    wait_until(lambda: request("GET", "/health", server=name))
                health = request("GET", "/services")
                self.assertTrue(health["dispatcher_enabled"])
                self.assertTrue(all(value["mock"] for value in health["services"].values()))
                data_root = root / "data_orbital"
                data_root.mkdir()
                image, mask = data_root / "image.png", data_root / "mask.png"
                Image.fromarray(np.full((256, 256), 64, dtype=np.uint8)).save(image)
                Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(mask)
                manifest = data_root / "manifest.jsonl"
                manifest.write_text(json.dumps({"sample_id": "http-fixture", "image": str(image),
                    "image_sha256": file_hash(image), "source_split": "test", "scene_group_id": "http-scene",
                    "mask": str(mask), "mask_sha256": file_hash(mask), "classes": ["Background", "IMP"],
                    "mask_semantics": "imp", "answer": "YES"}) + "\n", encoding="utf-8")
                dataset = request("POST", "/dataset-configs", {"name": "http-fixture", "planet": "moon",
                    "task_id": "imp_segmentation", "manifest_path": str(manifest),
                    "preprocessing": {"input_protocol": "curated_imp_png_reflectance_0_0.12_v1"}})
                runs = []
                for name, backend, endpoint in (("SpaceLLaVA", "spacellava", "vlm"), ("IBM", "ibm_imp", "vit")):
                    request("POST", "/model-configs", {"name": name, "model_id": "http-fixture", "backend": backend,
                        "enabled": True, "options": {"input_protocol": "curated_imp_png_reflectance_0_0.12_v1"}})
                    runs.append(request("POST", f"/{endpoint}/runs", {"model_name": name,
                        "dataset_id": dataset["id"], "image_count": 1, "distortions": [],
                        "prompt": "Describe visible evidence."}))
                for run in runs:
                    def finished():
                        value = request("GET", run["status_url"])
                        if value["status"] in {"blocked", "failed", "interrupted"}:
                            self.fail(str(value))
                        return value if value["status"] == "completed" else None
                    wait_until(finished)
                    rows = request("GET", run["results_url"])
                    self.assertEqual(len(rows), 1)
                    self.assertTrue(rows[0]["mock"])
                    self.assertTrue(request("GET", run["report_url"])["mock"])
                wait_until(lambda: all(not value["gpu_busy"] and value["active_run_id"] is None
                    for value in request("GET", "/services")["services"].values()))
                left, right = (request("GET", run["results_url"])[0] for run in runs)
                self.assertEqual(left["image_paths"], right["image_paths"])
            finally:
                for server in reversed(servers):
                    if server.poll() is None:
                        server.terminate()
                    try:
                        server.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
                for log in logs:
                    log.close()
                servers.clear()
                process = server = None
                gc.collect()


if __name__ == "__main__":
    unittest.main()
