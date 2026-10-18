"""Real CPU child processes with mock models; never scientific GPU results."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planetary_vlm.contracts import ModelRequest
from planetary_vlm.inference.config import RunConfig
from planetary_vlm.inference.fresh_process import child, run_fresh, SCHEMA
from planetary_vlm.io import read_jsonl, write_jsonl
from benchmark_services.common.files import atomic_json


class FreshProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        image = self.root / "synthetic.png"
        image.write_bytes(b"CPU fixture, not an actual model image")
        self.requests = [ModelRequest("fixture_" + str(i), (str(image),), "Fixture prompt " + str(i), (), 4) for i in range(2)]
        manifest = self.root / "requests.jsonl"
        write_jsonl(manifest, map(asdict, self.requests))
        self.config = RunConfig("parent", manifest, self.root / "outputs", "CPU_FIXTURE", "mock",
            {"responses": {"fixture_0": "FIRST_FIXTURE", "fixture_1": "SECOND_FIXTURE"}},
            42, "fixture-signature", self.root / "config.json")

    def test_real_subprocesses_have_one_predict_and_distinct_process_nonces(self):
        progress = []
        result = run_fresh(self.config, on_response=lambda folder, n, total: progress.append((n, total)))
        rows = read_jsonl(result / "predictions.jsonl")
        self.assertEqual([r["raw_response"] for r in rows], ["FIRST_FIXTURE", "SECOND_FIXTURE"])
        self.assertTrue(all(r["mock"] for r in rows))
        proof = json.loads((result / "isolation_report.json").read_text())
        workers = proof["workers"]
        self.assertEqual(len(workers), 2)
        self.assertEqual(len({w["process_nonce"] for w in workers}), 2)
        self.assertTrue(all(w["pid"] != os.getpid() and w["prior_predict_calls"] == 0 and
                            w["predict_calls_started"] == w["predict_calls_finished"] == 1 and
                            w["preliminary_inference"] is False for w in workers))
        self.assertEqual(progress, [(1, 2), (2, 2)])
        self.assertFalse(proof["scientific_result"] or proof["benchmark_admitted"])
        self.assertTrue((result / "logs/q000.log").is_file())
        for index in range(2):
            self.assertEqual(len(read_jsonl(result / "children" / f"q{index:03}" / "requests.jsonl")), 1)
        with self.assertRaisesRegex(ValueError, "exists"):
            run_fresh(self.config)

    def test_worker_rejects_multiple_requests_before_loading_a_model(self):
        job = self.root / "job.json"
        atomic_json(job, {"schema": SCHEMA, "job_nonce": "fixture", "config": asdict(self.config)})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            child(job)

    def test_preliminary_forward_probe_rejected(self):
        options = {**self.config.options, "diagnostic_request_settings": {
            self.requests[0].request_id: {"compare_first_forward": True}}}
        with self.assertRaisesRegex(ValueError, "preliminary"):
            run_fresh(replace(self.config, options=options))
        self.assertFalse((self.config.output_dir / self.config.run_id).exists())

    def test_failed_worker_stops_and_keeps_error_log_without_reusing_answer(self):
        config = replace(self.config, backend="missing_synthetic_backend")
        with self.assertRaisesRegex(RuntimeError, "exited"):
            run_fresh(config)
        result = config.output_dir / config.run_id
        self.assertIn("Unknown model backend", (result / "logs/q000.log").read_text(encoding="utf-8"))
        self.assertFalse((result / "predictions.jsonl").exists())
        self.assertFalse((result / "jobs/q001.json").exists())

    def test_unsafe_output_run_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            run_fresh(replace(self.config, run_id="../outside"))


class FreshSpatialPilotTests(unittest.TestCase):
    def setUp(self):
        from test_colab_spatial import SpatialPilotTests
        SpatialPilotTests.setUp(self)

    def test_four_fixed_v8_questions_persist_nested_worker_proofs_and_keep_gate(self):
        from planetary_vlm import colab, colab_spatial as spatial
        from planetary_vlm.inference import fresh_process
        proof_path = self.root / "colab_smoke_passed.json"
        proof_path.write_text('{"success":false}', encoding="utf-8")
        before = proof_path.read_bytes()
        actual_run_fresh = fresh_process.run_fresh
        captured = []
        def fixture_run(config, **kwargs):
            selected = read_jsonl(config.requests)
            captured.extend(selected)
            self.assertEqual([r["request_id"] for r in selected], list(spatial.FRESH_REQUEST_IDS))
            self.assertFalse(any(v["compare_first_forward"] for v in config.options["diagnostic_request_settings"].values()))
            return actual_run_fresh(replace(config, backend="mock", options={"default_response": "LEFT"}), **kwargs)
        with patch.object(colab, "environment", return_value={"fixture": True}), \
             patch.object(fresh_process, "run_fresh", side_effect=fixture_run), \
             patch("benchmark_services.spacellava.diagnostics.diagnose", side_effect=AssertionError("No preliminary inference")):
            result = spatial.run_pilot(self.root, self.root / "drive", "cold", "b" * 32, fresh_process=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["requests"], 4)
        self.assertFalse(result["benchmark_admitted"])
        durable = Path(result["persistent_dir"])
        self.assertEqual(len(read_jsonl(durable / "predictions.jsonl")), 4)
        self.assertTrue((durable / "children/q003/isolation_proof.json").is_file())
        self.assertTrue((durable / "logs/q003.log").is_file())
        self.assertTrue((durable / "jobs/q003.json").is_file())
        report = json.loads((durable / "spatial_report.json").read_text())
        self.assertEqual(report["development_controls_total"], 4)
        self.assertEqual(len(report["controls"]), 2)
        self.assertFalse(any("free" in c["answers"] for c in report["controls"]))
        self.assertEqual(proof_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
