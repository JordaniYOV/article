"""Diagnostics do not alter either scientific identities or prepared image bundles."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_adaptation import fixture
from test_robustness import trusted_runs
from planetary_vlm.adaptation.common.io import atomic, environment, read, sha
from planetary_vlm.identity_diagnostics import difference, main, package_delta, report
from planetary_vlm.robustness.bundle import implementation


class IdentityDiagnosticsTests(unittest.TestCase):
    def make_run(self, root):
        cfg, _ = fixture(root)
        clean = trusted_runs(root, cfg)
        run = clean / "linear_probe/seed_13"
        identity = read(run / "identity.json")
        identity["environment"] = {"python": "3.12.14", "platform": "fixture-platform",
                                   "packages": ["numpy==2.2.6", "torch==2.8.0+cu128"]}
        atomic(run / "identity.json", identity)
        complete = read(run / "complete.json")
        complete["identity_sha256"] = sha(run / "identity.json")
        atomic(run / "complete.json", complete)
        return run, identity

    def test_added_opencv_and_numpy_upgrade_are_explicit_not_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, identity = self.make_run(root)
            current = {**identity["environment"], "packages": ["numpy==2.4.0", "torch==2.8.0+cu128", "opencv-python-headless==4.13.0.92"]}
            before = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
            with patch("planetary_vlm.identity_diagnostics.environment", return_value=current):
                result = report(root, run)
            self.assertFalse(result["fast_fields_match"])
            change = next(r for r in result["differences"] if r["field"] == "environment.packages")
            self.assertEqual(change["details"]["added"], {"opencv-python-headless": ["4.13.0.92"]})
            self.assertEqual(change["details"]["changed_versions"]["numpy"]["training"], ["2.2.6"])
            self.assertEqual(before, {str(p): sha(p) for p in root.rglob("*") if p.is_file()})
            self.assertFalse(result["model_loaded"])
            self.assertIsNone(result["full_identity_match"])

    def test_matching_fast_fields_are_not_misrepresented_as_full_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, identity = self.make_run(root)
            with patch("planetary_vlm.identity_diagnostics.environment", return_value=identity["environment"]):
                result = report(root, run)
            self.assertTrue(result["fast_fields_match"])
            self.assertEqual(result["differences"], [])
            self.assertIn("hardware", result["not_checked"])
            self.assertIsNone(result["full_identity_match"])

    def test_bad_checkpoint_seal_and_platform_difference_remain_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, identity = self.make_run(root)
            (run / "best.pt").write_bytes(b"changed")
            with patch("planetary_vlm.identity_diagnostics.environment", return_value={**identity["environment"], "platform": "different-host"}):
                result = report(root, run)
            self.assertFalse(result["checkpoint_seals"]["checkpoint_sha256_matches"])
            self.assertTrue(any(r["field"] == "environment.platform" for r in result["differences"]))

    def test_cli_reports_without_torch_and_cannot_read_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, identity = self.make_run(root)
            with patch("planetary_vlm.identity_diagnostics.environment", return_value=identity["environment"]):
                with redirect_stdout(io.StringIO()) as out:
                    result = main(["--workspace", str(root), "--run", str(run)])
            self.assertEqual(result, 0)
            self.assertIn('"files_modified": false', out.getvalue())
            with self.assertRaisesRegex(ValueError, "outputs"):
                report(root, "data_rover/do-not-read")

    def test_package_canonicalization_does_not_silently_ignore_raw_changes(self):
        result = package_delta(["NumPy==2", "pillow==12"], ["numpy==2", "new_pkg==1"])
        self.assertEqual(result["changed_versions"], {})
        self.assertEqual(result["missing"], {"pillow": ["12"]})
        self.assertIn("NumPy==2", result["raw_missing_entries"])
        self.assertIn("numpy==2", result["raw_added_entries"])
        self.assertEqual(difference({"a": 1}, {"a": 1, "b": 2})[0]["status"], "added")

    def test_diagnostic_is_outside_both_frozen_code_collections(self):
        from planetary_vlm.adaptation.common.io import code_identity
        self.assertFalse(any("identity_diagnostics" in p for p in code_identity()))
        self.assertFalse(any("identity_diagnostics" in p for p in implementation()))


if __name__ == "__main__":
    unittest.main()
