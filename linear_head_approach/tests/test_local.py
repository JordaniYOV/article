"""CPU-only local orchestration checks; no weights, real training or GPU jobs."""
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from linear_head_approach.local import pipeline
from linear_head_approach.common import gpu_reservation


class LocalTests(unittest.TestCase):
    def test_pipeline_is_serial_and_never_scores_test_or_downloads(self):
        commands = []

        class Child:
            def __init__(self, command, **kwargs):
                commands.append(command)
                self.stdout = StringIO("synthetic stage fixture only\n")
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def wait(self, **kwargs): return 0

        with tempfile.TemporaryDirectory() as tmp, patch("linear_head_approach.local.subprocess.Popen", Child):
            result = pipeline("ibm", Path(tmp), Path(tmp) / "assets", "fixture", resume=True)
        stages = [c[c.index("linear_head_approach") + 1] for c in commands]
        self.assertEqual(stages[:5], ["audit", "assets", "extract", "extract", "head-smoke"])
        self.assertIn("--smoke", commands[2])
        self.assertEqual(stages.count("fit"), 6)
        self.assertTrue(all("--resume" in c for c in commands if "fit" in c))
        self.assertFalse(any("--allow-download" in c or "--allow-test" in c or "test" in c for c in commands))
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(result["test_target_usage"], "integrity_audit_only")

    def test_cli_and_ui_share_the_same_gpu_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with gpu_reservation(root):
                with self.assertRaisesRegex(RuntimeError, "GPU is reserved"):
                    with gpu_reservation(root):
                        pass
            with gpu_reservation(root):
                self.assertTrue((root / ".state/api-worker.lock").exists())
