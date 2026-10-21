"""CPU fixtures only: no downloaded weights, training or scientific result claims."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from test_adaptation import fixture
from planetary_vlm.adaptation.common.config import METHODS
from planetary_vlm.adaptation.common.data import Index, prepare as train_stats
from planetary_vlm.adaptation.common.io import atomic, code_identity, fingerprint, read, rows, sha
from planetary_vlm.robustness import bundle
from planetary_vlm.robustness.__main__ import coordinate, main, serial
from planetary_vlm.robustness.evaluation import counts, score_condition, smoke
from planetary_vlm.robustness.protocol import dedicated, discover, job_id, load_protocol, selection
from planetary_vlm.robustness.reporting import distribution, summarize


def small_protocol():
    protocol = load_protocol()
    protocol["conditions"] = [c for c in protocol["conditions"] if c["id"] in ("clean", "shear_2", "radiation_1")]
    return protocol


def trusted_runs(root, cfg):
    """Sealed dummy bytes, never unpickled. CPU orchestrator contract only."""
    clean = root / "outputs/clean"
    stats = train_stats(Index(root, cfg), clean)
    for method in METHODS:
        current = deepcopy(cfg)
        current["method"] = method
        base = {"config": current, "dataset": Index(root, current).identity, "code": code_identity()}
        if method == "sam_baseline":
            atomic(clean / method / "smoke.json", {"identity": base})
            continue
        for seed in current["seeds"]:
            folder = clean / method / f"seed_{seed}"
            folder.mkdir(parents=True)
            atomic(folder / "identity.json", {**base, "stats": stats, "seed": seed})
            (folder / "best.pt").write_bytes(b"not a real model; never load with Torch")
            atomic(folder / "complete.json", {"method": method, "seed": seed,
                   "identity_sha256": sha(folder / "identity.json"), "best_checkpoint_sha256": sha(folder / "best.pt")})
    return clean


class ProtocolTests(unittest.TestCase):
    def test_default_conditions_and_all_three_levels(self):
        cfg = load_protocol()
        self.assertEqual(len(cfg["conditions"]), 13)
        self.assertEqual(cfg["conditions"][0]["id"], "clean")
        self.assertEqual(cfg, load_protocol())
        for effect in ("shear", "radiation", "lens_flare", "hard_shadows"):
            self.assertEqual(sum(c["chain"] == [effect] for c in cfg["conditions"]), 3)

    def test_bad_protocol_fails_without_transform_execution(self):
        for content in ('schema="wrong"',
                        'schema="mars-robustness-v1"\nseed=1\n[[conditions]]\nid="../bad"\nchain=[]',
                        'schema="mars-robustness-v1"\nseed=1\n[[conditions]]\nid="clean"\nchain=[]\n[[conditions]]\nid="x"\nchain=["unknown"]'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.toml"
                path.write_text(content)
                with self.assertRaises(ValueError):
                    load_protocol(path)

    def test_discovery_checks_seals_all_seeds_and_canonical_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            clean = trusted_runs(root, cfg)
            _, _, jobs = discover(root, clean, METHODS[::-1])
            self.assertEqual(len(jobs), 10)
            self.assertEqual([j["method"] for j in jobs], [m for m in METHODS[:3] for _ in range(3)] + ["sam_baseline"])
            self.assertEqual([j["seed"] for j in jobs[:3]], [13, 37, 73])
            (clean / "linear_probe/seed_37/best.pt").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "changed"):
                discover(root, clean)

    def test_plan_read_only_and_test_guard_before_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            trusted_runs(root, cfg)
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            with redirect_stdout(io.StringIO()) as out:
                status = main(["plan", "--workspace", str(root), "--clean-output", "outputs/clean"])
            self.assertEqual(status, 0, out.getvalue())
            self.assertIn('"GPU_started": false', out.getvalue())
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob("*")))
            with redirect_stderr(io.StringIO()) as errors, patch("planetary_vlm.robustness.__main__.discover") as lookup:
                self.assertEqual(main(["run", "--workspace", str(root), "--split", "test"]), 2)
            lookup.assert_not_called()
            self.assertIn("--allow-test", errors.getvalue())

    def test_paths_cannot_write_to_archives_or_outputs_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ("data_rover/a", "data_orbital/../data_rover/a", "data_orbital", "../escape"):
                with self.assertRaises(ValueError):
                    dedicated(root, path, "data_orbital")
            with self.assertRaises(ValueError):
                dedicated(root, "outputs", "outputs")

    def test_selection_is_blind_repeatable_and_does_not_touch_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            index = Index(root, cfg)
            with patch.object(index, "mask", side_effect=AssertionError("No labels for selection")):
                self.assertEqual(selection(index, "val", 1, 42), selection(index, "val", 1, 42))
                self.assertEqual({r["split"] for r in selection(index, "val", 0, 42)}, {"val"})


class PreparationTests(unittest.TestCase):
    def test_shared_files_match_existing_transforms_and_targets(self):
        from benchmark_services.common.transforms import apply_chain
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, original = fixture(root)
            index, protocol = Index(root, cfg), load_protocol()
            selected, folder = index.split("val"), root / "data_orbital/corruptions"
            source_hashes = {p.name: sha(p) for p in original.iterdir() if p.is_file()}
            bundle.prepare(root, folder, index, protocol, selected, "val")
            definition = bundle.definition(index, protocol, selected, "val")
            inputs, targets = bundle.verify(root, folder, definition)
            self.assertEqual(len(inputs), 39)
            self.assertEqual(sum(r["role"] == "control" for r in inputs), 26)
            sid = selected[0]["sample_id"]
            for condition in protocol["conditions"]:
                row = next(r for r in inputs if r["sample_id"] == sid and r["condition"] == condition["id"])
                with Image.open(root / selected[0]["image"]) as image:
                    pixels = np.asarray(image).copy()
                expected_image, expected_mask = apply_chain(pixels, condition["chain"], condition["parameters"],
                                                           protocol["seed"], sid, mask=index.mask(sid))
                with Image.open(root / row["image"]) as image:
                    np.testing.assert_array_equal(np.asarray(image), expected_image)
                actual = bundle.read_target(root, targets[(condition["id"], sid)])
                np.testing.assert_array_equal(actual, expected_mask)
                if condition["id"] == "shear_2":
                    self.assertGreater(int((actual == 255).sum()), 0)
                if condition["id"] == "clean":
                    self.assertEqual(row["image_sha256"], selected[0]["image_sha256"])
                self.assertNotIn("mask", row)
            self.assertEqual(source_hashes, {p.name: sha(p) for p in original.iterdir() if p.is_file()})

    def test_complete_resume_never_calls_transforms_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            index, protocol = Index(root, cfg), small_protocol()
            folder, selected = root / "data_orbital/corruptions", index.split("val")
            bundle.prepare(root, folder, index, protocol, selected, "val")
            pngs = list(folder.glob("images/*/*.png"))
            times = {str(p): p.stat().st_mtime_ns for p in pngs}
            with patch("benchmark_services.common.transforms.apply_chain", side_effect=AssertionError("Do not regenerate")):
                result = bundle.prepare(root, folder, index, protocol, selected, "val")
            self.assertTrue(result["reused_complete_bundle"])
            self.assertEqual(times, {str(p): p.stat().st_mtime_ns for p in pngs})
            pngs[0].write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                bundle.prepare(root, folder, index, protocol, selected, "val")

    def test_interrupted_preparation_resumes_completed_records(self):
        from benchmark_services.common.transforms import apply_chain
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            index, protocol = Index(root, cfg), small_protocol()
            folder, selected = root / "data_orbital/corruptions", index.split("val")
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                if args[0].shape == (512, 512):
                    calls += 1
                    if calls == 3:
                        raise RuntimeError("interrupted")
                return apply_chain(*args, **kwargs)
            with patch("benchmark_services.common.transforms.apply_chain", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    bundle.prepare(root, folder, index, protocol, selected, "val")
            done = list(folder.glob("records/*/*.json"))
            self.assertEqual(len(done), 2)
            hashes = {str(p): sha(p) for p in done}
            bundle.prepare(root, folder, index, protocol, selected, "val")
            self.assertEqual(hashes, {str(p): sha(p) for p in done})
            bundle.verify(root, folder, bundle.definition(index, protocol, selected, "val"))

    def test_protocol_change_and_original_directory_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, original = fixture(root)
            index, protocol = Index(root, cfg), small_protocol()
            folder = root / "data_orbital/corruptions"
            bundle.prepare(root, folder, index, protocol, index.split("val"), "val")
            protocol["seed"] += 1
            with self.assertRaises(ValueError):
                bundle.prepare(root, folder, index, protocol, index.split("val"), "val")
            with self.assertRaises(ValueError):
                bundle.prepare(root, original / "child", index, protocol, index.split("val"), "val")


class EvaluationTests(unittest.TestCase):
    def test_ignore_pixels_are_neither_false_positives_nor_craters(self):
        target = np.array([[0, 1, 255]], dtype=np.uint8)
        result = counts(np.array([[0, 1, 1]], dtype=np.uint8), target)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["iou"]), (1, 0, 0, 1.))
        self.assertEqual((result["valid_pixels"], result["ignored_pixels"]), (2, 1))

    def test_prediction_resume_verifies_pixels_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            index, protocol = Index(root, cfg), small_protocol()
            folder = root / "data_orbital/corruptions"
            bundle.prepare(root, folder, index, protocol, index.split("val"), "val")
            inputs, targets = bundle.verify(root, folder, bundle.definition(index, protocol, index.split("val"), "val"))
            def predictor(row):
                self.assertNotIn("mask", row)
                self.assertNotIn("target", row)
                self.assertEqual(row["role"], "evaluation")
                return np.zeros((512, 512), dtype=np.uint8), {}
            log = SimpleNamespace(info=lambda *args: None)
            output = root / "outputs/new/clean"
            first = score_condition(root, predictor, inputs, targets, "clean", output, {"fake": True}, log)
            with patch("builtins.print"):
                second = score_condition(root, lambda row: self.fail("Repeated inference"), inputs, targets, "clean", output, {"fake": True}, log)
            self.assertEqual(first, second)
            self.assertEqual(first["metrics"]["images"], 1)
            mask = next((output / "masks").glob("*.png"))
            mask.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                score_condition(root, predictor, inputs, targets, "clean", output, {"fake": True}, log)

    def test_smoke_only_train_controls_without_accuracy_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def predictor(row):
                calls.append(row)
                return np.zeros((512, 512), dtype=np.uint8), {}
            inputs = [{"sample_id": f"train_{i}", "condition": "clean", "role": "control"} for i in (0, 1)]
            inputs.append({"sample_id": "test_0", "role": "evaluation"})
            marker = smoke(predictor, inputs, Path(directory))
            self.assertEqual(len(calls), 4)
            self.assertFalse(marker["quality_claim"])
            self.assertEqual({r["sample_id"] for r in calls}, {"train_0", "train_1"})

    def test_summary_keeps_paired_seeds_and_sam_no_fake_std(self):
        self.assertEqual(distribution([.1, .2, .3])["mean"], .2)
        self.assertIsNone(distribution([.2])["sample_std"])
        self.assertIsNone(distribution([None, .2])["mean"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            clean = trusted_runs(root, cfg)
            _, _, jobs = discover(root, clean)
            output = root / "outputs/corrupt_eval"
            definition = {"protocol": small_protocol(), "selected_ids": ["x"], "split": "test"}
            for job in jobs:
                folder = output / job_id(job)
                identity = {"job": job, "bundle_complete_sha256": "bundle"}
                atomic(folder / "identity.json", identity)
                hashes = {}
                for condition in definition["protocol"]["conditions"]:
                    cid = condition["id"]
                    value = .6 if cid == "clean" else .4
                    metrics = {name: value for name in ("scene_macro_iou", "image_macro_iou", "image_macro_dice", "foreground_iou", "precision", "recall")}
                    atomic(folder / cid / "report.json", {"identity": {**identity, "condition": cid},
                           "records": [{"sample_id": "x"}], "metrics": metrics})
                    hashes[cid] = sha(folder / cid / "report.json")
                atomic(folder / "complete.json", {"identity_sha256": fingerprint(identity), "reports": hashes})
            summarize(output, jobs, definition, "bundle")
            result = read(output / "summary.json")
            self.assertEqual(len(result["rows"]), 12)
            self.assertAlmostEqual(result["rows"][1]["clean_minus_condition"]["scene_macro_iou"]["mean"], .2)
            self.assertIsNone(result["rows"][-1]["metrics"]["scene_macro_iou"]["sample_std"])


class CoordinatorTests(unittest.TestCase):
    def test_prepare_completes_before_all_serial_jobs_and_no_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            clean = trusted_runs(root, cfg)
            _, index, jobs = discover(root, clean)
            protocol = small_protocol()
            selected = index.split("val")
            folder, output = root / "data_orbital/corruptions", root / "outputs/evaluation"
            definition = bundle.definition(index, protocol, selected, "val")
            args = SimpleNamespace(split="val", limit=0, device="cuda", methods=list(METHODS), config=None, allow_test=False)
            commands = []
            def child(command, root, log_path):
                bundle.verify(root, folder, definition)
                commands.append(command)
            with patch("planetary_vlm.robustness.__main__.serial", side_effect=child), \
                    patch("planetary_vlm.robustness.reporting.summarize", return_value={"complete": True}):
                with redirect_stdout(io.StringIO()):
                    coordinate(args, root, clean, output, folder, index, protocol, selected, jobs, definition)
            self.assertEqual(len(commands), 10)
            self.assertEqual([int(c[c.index("--job") + 1]) for c in commands], list(range(10)))
            self.assertTrue(all("worker" in c and "fit" not in c and "train" not in c for c in commands))
            self.assertTrue(all(str(folder) in c for c in commands))

    def test_child_failure_stops_and_closes_stdout(self):
        process = SimpleNamespace(stdout=io.StringIO("model failed\n"), returncode=2,
                                  wait=lambda **kwargs: 2, poll=lambda: 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("planetary_vlm.robustness.__main__.subprocess.Popen", return_value=process):
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "resume"):
                    serial(["fixture"], root, root / "logs/failure.log")
            self.assertTrue(process.stdout.closed)
            self.assertIn("model failed", (root / "logs/failure.log").read_text())


if __name__ == "__main__":
    unittest.main()
