"""Legacy export recovery: synthetic files/checkpoints only, no Torch or GPU."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from linear_head_approach.common import ROOT, atomic_json, environment, fingerprint, read_json, sha
from linear_head_approach.compatibility import code_compatibility, code_manifest, verify_deployment_code


def options(files):
    return {name: [hashlib.sha256(data).hexdigest(), hashlib.sha256(data.replace(b"\n", b"\r\n")).hexdigest()]
            for name, data in files.items()}


def historical_hash(files, crlf=()):
    return fingerprint({name: hashes[int(name in crlf)] for name, hashes in options(files).items()})


def identity(code, encoder="ibm"):
    return {"code": code, "feature_identity": {"code": code, "encoder": encoder}}


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name)
        self.before = {"heads.py": b"HEAD = 'same'\n", "encoders.py": b"IBM = 'same'\nCLIP = 'before'\n"}
        self.after = self.before | {"encoders.py": b"IBM = 'same'\nCLIP = 'fixed'\n"}
        runtime = self.after | {"compatibility.py": b"AUDIT = True\n"}
        for name, content in runtime.items():
            (self.package / name).write_bytes(content)
        self.policy = {"schema": "frozen-heads-code-compatibility-v1", "reason": "synthetic equivalence fixture only",
                       "runtime_sha256": code_manifest(self.package, True), "profiles": [
            {"id": "before", "git_revision": "synthetic-before", "encoders": ["ibm"], "file_sha256": options(self.before)},
            {"id": "after", "git_revision": "synthetic-after", "encoders": ["ibm", "spacellava"], "file_sha256": options(self.after)}]}
        atomic_json(self.package / "compatibility_revisions.json", self.policy)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_identity_needs_no_legacy_policy(self):
        (self.package / "compatibility_revisions.json").unlink()
        checked = code_compatibility(identity(fingerprint(code_manifest(self.package))), package=self.package)
        self.assertEqual(checked["mode"], "exact")

    def test_ibm_legacy_is_attested_without_rewriting_identity(self):
        saved = identity(historical_hash(self.before))
        original = deepcopy(saved)
        checked = code_compatibility(saved, package=self.package)
        self.assertEqual(checked["mode"], "audited_compatibility")
        self.assertEqual(checked["profile"], "before")
        self.assertEqual(saved, original)

    def test_only_post_fix_spacellava_is_allowed(self):
        with self.assertRaisesRegex(ValueError, "not an audited compatible"):
            code_compatibility(identity(historical_hash(self.before), "spacellava"), package=self.package)
        checked = code_compatibility(identity(historical_hash(self.after), "spacellava"), package=self.package)
        self.assertEqual(checked["profile"], "after")

    def test_crlf_and_mixed_checkout_preserve_historical_evidence(self):
        for crlf in (tuple(self.before), ("encoders.py",)):
            with self.subTest(crlf=crlf):
                checked = code_compatibility(identity(historical_hash(self.before, crlf)), package=self.package)
                self.assertEqual(checked["profile"], "before")

    def test_unaudited_runtime_change_is_blocked(self):
        (self.package / "heads.py").write_bytes(b"HEAD = 'different'\n")
        with self.assertRaisesRegex(ValueError, "differs from the audited"):
            code_compatibility(identity(historical_hash(self.before)), package=self.package)

    def test_unknown_training_revision_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "not an audited compatible"):
            code_compatibility(identity("f" * 64), package=self.package)

    def test_feature_training_revision_mismatch_is_blocked(self):
        saved = identity(fingerprint(code_manifest(self.package)))
        saved["feature_identity"]["code"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "identities disagree"):
            code_compatibility(saved, package=self.package)

    def test_worker_requires_identical_recorded_legacy_attestation(self):
        saved = identity(historical_hash(self.before))
        checked = code_compatibility(saved, package=self.package)
        with patch("linear_head_approach.compatibility.code_compatibility", return_value=checked):
            self.assertEqual(verify_deployment_code({"training_identity": saved, "code_compatibility": checked}), checked)
            for recorded in (None, checked | {"profile": "wrong"}):
                with self.subTest(recorded=recorded), self.assertRaisesRegex(ValueError, "attestation differs"):
                    verify_deployment_code({"training_identity": saved, "code_compatibility": recorded})

    def test_shipped_policy_matches_runtime_and_admits_known_snapshots(self):
        package = ROOT / "linear_head_approach"
        policy = read_json(package / "compatibility_revisions.json")
        self.assertEqual(code_manifest(package, True), policy["runtime_sha256"])
        for profile in policy["profiles"]:
            for encoder in profile["encoders"]:
                for newline_index in (0, 1):
                    code = fingerprint({name: values[newline_index] for name, values in profile["file_sha256"].items()})
                    checked = code_compatibility(identity(code, encoder))
                    self.assertEqual(checked["profile"], profile["id"])

    def test_publish_preserves_legacy_checkpoint_and_identity_bytes(self):
        from test_pipeline import fixture
        from linear_head_approach.deployment import publish
        from linear_head_approach.data import DATA
        policy = read_json(ROOT / "linear_head_approach/compatibility_revisions.json")
        profile = policy["profiles"][0]
        code = fingerprint({name: hashes[0] for name, hashes in profile["file_sha256"].items()})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            run = root / "run"
            run.mkdir()
            saved_identity = identity(code) | {"environment": environment(), "targets_sha256": sha(root / DATA / "targets.jsonl"),
                "config": {"hidden_channels": 64}, "family": "spatial", "seed": 13}
            saved_identity["feature_identity"]["assets"] = {"fixture": True}
            saved_identity["feature_identity"]["inputs_sha256"] = sha(root / DATA / "inputs.jsonl")
            atomic_json(run / "identity.json", saved_identity)
            atomic_json(run / "fit_summary.json", {"epochs": 1, "encoder_trained": False, "test_used_for_selection": False})
            (run / "best.pt").write_bytes(b"synthetic checkpoint fixture; not trained weights")
            before = {name: sha(run / name) for name in ("identity.json", "best.pt")}
            checkpoint = {"identity": saved_identity, "channels": 768, "threshold": .4, "epoch": 0, "parameters": 1}
            torch = ModuleType("torch")
            torch.load = Mock(return_value=checkpoint)
            with patch.dict(sys.modules, {"torch": torch}), patch("linear_head_approach.encoders.assets", return_value={"fixture": True}):
                result = publish(run, root / "assets", root / "outputs/deploy", "synthetic-legacy-head", root)
            self.assertEqual(before, {name: sha(run / name) for name in before})
            deployment = read_json(result["deployment"])
            self.assertEqual(deployment["training_identity"], saved_identity)
            self.assertEqual(deployment["threshold"], .4)
            self.assertEqual(sha(Path(result["deployment"]).parent / "head.pt"), before["best.pt"])
            self.assertEqual(verify_deployment_code(deployment)["profile"], profile["id"])
            self.assertFalse(result["training_started"])
            self.assertFalse(result["test_evaluated"])
            torch.load.assert_called_once_with(run / "best.pt", map_location="cpu", weights_only=True)

    def test_publish_does_not_waive_environment_or_checkpoint_mismatch(self):
        from linear_head_approach.deployment import publish
        saved_identity = identity("f" * 64) | {"environment": {"python": "different", "packages": []}}
        for checkpoint_identity, message in ((saved_identity, "Python/dependency"), ({}, "Checkpoint identity")):
            with self.subTest(message=message):
                torch = ModuleType("torch")
                torch.load = Mock(return_value={"identity": checkpoint_identity})
                def read(path):
                    return ({"epochs": 1, "encoder_trained": False, "test_used_for_selection": False}
                            if Path(path).name == "fit_summary.json" else saved_identity)
                with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"torch": torch}), \
                        patch("linear_head_approach.deployment.read_json", side_effect=read), \
                        patch("linear_head_approach.deployment.code_compatibility") as compatibility:
                    root = Path(tmp)
                    with self.assertRaisesRegex(ValueError, message):
                        publish(root / "run", root / "assets", root / "outputs/deploy", "fixture", root)
                    compatibility.assert_not_called()
                    self.assertFalse((root / "outputs/deploy").exists())

    def test_publish_still_rejects_changed_encoder_inputs(self):
        from linear_head_approach.deployment import publish
        saved_identity = identity("f" * 64) | {"environment": environment(), "targets_sha256": "same-targets"}
        saved_identity["feature_identity"]["inputs_sha256"] = "original-inputs"
        torch = ModuleType("torch")
        torch.load = Mock(return_value={"identity": saved_identity})
        def read(path):
            return ({"epochs": 1, "encoder_trained": False, "test_used_for_selection": False}
                    if Path(path).name == "fit_summary.json" else saved_identity)
        def digest(path):
            return "same-targets" if Path(path).name == "targets.jsonl" else "changed-inputs"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"torch": torch}), \
                patch("linear_head_approach.deployment.read_json", side_effect=read), \
                patch("linear_head_approach.deployment.sha", side_effect=digest), \
                patch("linear_head_approach.deployment.code_compatibility", return_value={"mode": "audited_compatibility"}), \
                patch("linear_head_approach.deployment.audit") as audit:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Encoder inputs changed"):
                publish(root / "run", root / "assets", root / "outputs/deploy", "fixture", root)
            audit.assert_not_called()
            self.assertFalse((root / "outputs/deploy").exists())
