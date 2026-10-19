"""New pipelines: CPU data/CLI contracts; optional genuine Torch fixture checks."""
import ast
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from planetary_vlm.adaptation.common.config import METHODS, load, validate
from planetary_vlm.adaptation.common.data import Index, prepare, cache_identity
from planetary_vlm.adaptation.common.io import atomic, code_identity, experiment, guard, orbital, read, sha, within
from planetary_vlm.adaptation.common.metrics import binary_counts, summarize, select_threshold
from planetary_vlm.datasets.mars_binary import REPO, REVISION

ROOT = Path(__file__).resolve().parents[1]
TORCH = bool(importlib.util.find_spec("torch"))


def fixture(root):
    cfg = load("linear_probe")
    cfg["counts"] = {"train": 2, "val": 1, "test": 1}
    folder = root / cfg["dataset"]
    folder.mkdir(parents=True)
    inputs, targets = [], []
    for number, split in enumerate(("train", "train", "val", "test")):
        sid = f"fixture_{number}"
        image, mask = folder / f"{sid}.png", folder / f"{sid}_mask.png"
        values = np.random.default_rng(number).integers(20, 180, (512, 512), dtype=np.uint8)
        labels = np.zeros((512, 512), dtype=np.uint8)
        if number != 0:
            labels[100:210, 100:240] = 1
        Image.fromarray(values).save(image)
        Image.fromarray(labels).save(mask)
        inputs.append({"sample_id": sid, "split": split, "scene_group_id": "group_" + split,
                       "width": 512, "height": 512, "channels": 1, "planet": "mars",
                       "image": image.relative_to(root).as_posix(), "image_sha256": sha(image)})
        targets.append({"sample_id": sid, "mask": mask.relative_to(root).as_posix(), "mask_sha256": sha(mask)})
    atomic(folder / "inputs.jsonl", inputs, lines=True)
    atomic(folder / "targets.jsonl", targets, lines=True)
    atomic(folder / "metadata.json", {"schema": "mars-crater-binary-v1", "status": "complete",
           "source_dataset": REPO, "source_revision": REVISION,
           "files_sha256": {name: sha(folder / name) for name in ("inputs.jsonl", "targets.jsonl")}})
    return cfg, folder


class ConfigurationTests(unittest.TestCase):
    def test_every_method_has_independent_entrypoint_and_docs(self):
        base = ROOT / "src/planetary_vlm/adaptation"
        for method in METHODS:
            cfg = load(method)
            self.assertEqual(cfg["method"], method)
            self.assertEqual(cfg["seeds"], [13, 37, 73])
            for name in ("pipeline.py", "__main__.py", "README.md", "config.toml"):
                self.assertTrue((base / method / name).is_file(), (method, name))

    def test_b_c_share_decoder_and_controlled_input_budget(self):
        b, c = load("frozen_decoder"), load("lora_decoder")
        for key in ("dataset", "counts", "encoder", "training", "decoder", "seeds"):
            self.assertEqual(b[key], c[key])
        source = (ROOT / "src/planetary_vlm/adaptation/lora_decoder/model.py").read_text()
        self.assertIn("from ..frozen_decoder.model import PyramidDecoder", source)

    def test_invalid_or_misspelled_config_not_silently_ignored(self):
        cfg = load("linear_probe")
        cfg["encoder"]["modality"] = "vis"
        with self.assertRaises(ValueError):
            validate(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.toml"
            path.write_text('method="linear_probe"\nmisspelled_lr=1\n')
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load("linear_probe", path)

    def test_all_new_python_sources_parse_without_torch(self):
        for path in (ROOT / "src/planetary_vlm/adaptation").rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.assertGreater(len(code_identity()), 20)

    def test_strict_linear_architecture_is_not_old_spatial_head(self):
        source = (ROOT / "src/planetary_vlm/adaptation/linear_probe/model.py").read_text()
        self.assertIn("nn.Conv2d(channels, 1, 1)", source)
        for name in ("GELU", "GroupNorm", "BatchNorm", "ConvTranspose"):
            self.assertNotIn(name, source)


class DataContracts(unittest.TestCase):
    def test_stats_use_train_only_and_preserve_source_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            index = Index(root, cfg)
            output = experiment(root, "outputs/fixture")
            accessed_images, accessed_masks = [], []
            image_fn, mask_fn = index.image, index.mask
            def image(row, *args, **kwargs):
                accessed_images.append(row["split"])
                return image_fn(row, *args, **kwargs)
            def mask(sid):
                accessed_masks.append(sid)
                return mask_fn(sid)
            with patch.object(index, "image", side_effect=image), patch.object(index, "mask", side_effect=mask):
                stats = prepare(index, output)
            self.assertEqual(set(accessed_images), {"train"})
            self.assertTrue(set(accessed_masks) <= {r["sample_id"] for r in index.split("train")})
            self.assertEqual(stats["controls"], ["fixture_0", "fixture_1"])
            self.assertFalse(stats["test_used"])
            self.assertGreater(stats["std"], 0)
            self.assertEqual(prepare(index, output), stats)

    def test_cross_split_groups_and_target_leakage_fail_closed(self):
        for problem in ("group", "target", "hash"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cfg, folder = fixture(root)
                records = [json.loads(line) for line in (folder / "inputs.jsonl").read_text().splitlines()]
                if problem == "group":
                    records[-1]["scene_group_id"] = records[0]["scene_group_id"]
                elif problem == "target":
                    records[0]["mask"] = "forbidden"
                else:
                    (folder / "inputs.jsonl").write_text("changed")
                if problem != "hash":
                    atomic(folder / "inputs.jsonl", records, lines=True)
                    meta = read(folder / "metadata.json")
                    meta["files_sha256"]["inputs.jsonl"] = sha(folder / "inputs.jsonl")
                    atomic(folder / "metadata.json", meta)
                with self.assertRaises(ValueError):
                    Index(root, cfg)

    def test_a_b_cache_identity_does_not_depend_on_head_or_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            index = Index(root, cfg)
            stats = {"mean": .3, "std": .2}
            a = cache_identity(index, stats, {"assets": 1}, {"code": 1}, {"environment": 1})
            index.cfg["method"] = "frozen_decoder"
            index.cfg["training"]["head_lr"] = .5
            b = cache_identity(index, stats, {"assets": 1}, {"code": 1}, {"environment": 1})
            self.assertEqual(a, b)

    def test_paths_and_identity_never_overwrite_old_runs_or_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "outputs/legacy"
            old.mkdir(parents=True)
            (old / "old.pt").write_bytes(b"preserve")
            for output in ("outputs", "data_orbital/new", "../outside", "outputs/legacy"):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    experiment(root, output)
            for value in ("data_rover/file", "data_orbital/../data_rover/file", "../outside"):
                with self.assertRaises(ValueError):
                    orbital(root, value)
            with self.assertRaises(ValueError):
                within(old, "../escape")
            self.assertEqual((old / "old.pt").read_bytes(), b"preserve")
            marker = root / "outputs/new/identity.json"
            guard(marker, {"version": 1})
            with self.assertRaises(ValueError):
                guard(marker, {"version": 2})


class MetricsAndSAMTests(unittest.TestCase):
    def test_empty_cases_do_not_disappear_from_metrics(self):
        blank = np.zeros((8, 8), dtype=np.uint8)
        self.assertEqual(binary_counts(blank, blank)["iou"], 1)
        false = binary_counts(np.ones_like(blank), blank)
        report = summarize([{"scene_group_id": "a", **false}])
        self.assertEqual(report["empty_target_false_positive_images"], 1)
        self.assertEqual(report["foreground_iou"], 0)

    def test_scene_macro_weights_regions_not_only_tiles(self):
        records = [{"scene_group_id": group, "iou": score, "dice": score, "tp": int(score),
                    "fp": 0, "fn": int(not score), "empty_target": False}
                   for group, score in (("a", 1), ("a", 1), ("b", 0))]
        self.assertAlmostEqual(summarize(records)["scene_macro_iou"], .5)
        self.assertAlmostEqual(summarize(records)["image_macro_iou"], 2 / 3)
        self.assertEqual(select_threshold({.6: records, .2: records}), .2)

    def test_sam_filter_uses_geometry_only_and_preserves_pixel_grid(self):
        from planetary_vlm.adaptation.sam_baseline.filter import union_candidates
        yy, xx = np.mgrid[:64, :64]
        disk = (xx - 20) ** 2 + (yy - 20) ** 2 < 8 ** 2
        strip = np.zeros((64, 64), dtype=bool)
        strip[40:42, :] = True
        cfg = load("sam_baseline")["sam"]
        result, accepted = union_candidates([{"segmentation": disk}, {"segmentation": strip}], disk.shape, cfg)
        np.testing.assert_array_equal(result, disk)
        self.assertEqual(accepted, 1)
        with self.assertRaises(ValueError):
            union_candidates([{"segmentation": np.zeros((32, 32))}], disk.shape, cfg)


class CLIContracts(unittest.TestCase):
    def test_plan_is_read_only_and_does_not_load_tensor_dependencies(self):
        from planetary_vlm.adaptation.common.cli import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            config = root / "fixture.toml"
            config.write_text('method="linear_probe"\n[counts]\ntrain=2\nval=1\ntest=1\n')
            with redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main("linear_probe", ["plan", "--workspace", str(root), "--config", str(config)]), 0)
            value = json.loads(stream.getvalue())
            self.assertFalse(value["training_started"])
            self.assertFalse((root / "outputs").exists())

    def test_test_scoring_fails_before_importing_model_backend(self):
        from planetary_vlm.adaptation.common.cli import main
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stderr(io.StringIO()) as errors:
                result = main("linear_probe", ["evaluate", "--workspace", directory, "--split", "test"])
            self.assertEqual(result, 2)
            self.assertIn("--allow-test", errors.getvalue())

    def test_train_coordinator_is_serial_and_never_evaluates_test(self):
        from planetary_vlm.adaptation.common.cli import pipeline
        class Process:
            def __init__(self, *args, **kwargs):
                self.stdout, self.returncode = iter(["fixture stage\n"]), 0
            def poll(self): return 0
            def wait(self, *args, **kwargs): return 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = experiment(root, "outputs/new")
            args = SimpleNamespace(workspace=str(root), resume=True, config=None, device="cuda")
            with patch("planetary_vlm.adaptation.common.cli.subprocess.Popen", side_effect=Process) as processes:
                with redirect_stdout(io.StringIO()):
                    result = pipeline(args, load("linear_probe"), output)
            commands = [call.args[0] for call in processes.call_args_list]
            self.assertFalse(result["test_evaluated"])
            self.assertEqual(sum("fit" in command for command in commands), 3)
            self.assertEqual(sum("evaluate" in command for command in commands), 3)
            self.assertTrue(all("test" not in command and "--allow-test" not in command for command in commands))
            self.assertTrue(all("--resume" in command for command in commands if "fit" in command))


@unittest.skipUnless(TORCH, "PyTorch absent: run these synthetic CPU contracts in requirements/adaptation.txt environment")
class TorchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_linear_is_affine_and_decoder_forward_backward(self):
        import torch
        from planetary_vlm.adaptation.linear_probe.model import LinearProbe
        from planetary_vlm.adaptation.frozen_decoder.model import PyramidDecoder
        features = [torch.randn(1, 8, 4, 4) for _ in range(4)]
        head = LinearProbe(8)
        zeros = [torch.zeros_like(x) for x in features]
        doubled = [2 * x for x in features]
        torch.testing.assert_close(head(doubled, 16) - head(zeros, 16), 2 * (head(features, 16) - head(zeros, 16)))
        decoder = PyramidDecoder(8, 8)
        out = decoder(features, 32)
        self.assertEqual(tuple(out.shape), (1, 1, 32, 32))
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in decoder.parameters()))

    def test_lora_zero_init_and_frozen_base_gradients(self):
        import torch
        from planetary_vlm.adaptation.common.encoder import LowRankLinear, CheckpointBlock
        base = torch.nn.Linear(8, 24)
        adapter = LowRankLinear(base, 4, 8., 0.)
        x = torch.randn(1, 4, 8)
        torch.testing.assert_close(adapter(x), base(x))
        adapter(x).square().mean().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(float(adapter.b.grad.abs().sum()), 0)

    def test_checkpoint_block_handles_inputs_without_grad(self):
        import torch
        from planetary_vlm.adaptation.common.encoder import LowRankLinear, CheckpointBlock
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = LowRankLinear(torch.nn.Linear(8, 8), 2, 4., 0.)
            def forward(self, x, mask=None): return self.projection(x)
        block = CheckpointBlock(Block())
        block(torch.randn(1, 4, 8)).square().mean().backward()
        self.assertIsNotNone(block.block.projection.b.grad)

    def test_synthetic_fit_resume_and_sealed_test(self):
        import torch
        from planetary_vlm.adaptation.common import engine
        from planetary_vlm.adaptation.common.model import System
        from planetary_vlm.adaptation.linear_probe.model import LinearProbe
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            cfg["seeds"] = [13]
            cfg["training"].update(max_epochs=1, batch_size=1, accumulate=1, thresholds=[.5])
            index = Index(root, cfg)
            output = experiment(root, "outputs/new")
            stats = prepare(index, output)
            cache = output / "cache/fixture"
            cache.mkdir(parents=True)
            for row in index.inputs:
                np.save(cache / (row["sample_id"] + ".npy"), np.ones((4, 8, 4, 4), dtype=np.float16))
            context = (index, stats, {"fixture": True}, cache, {"fixture": True})
            def model(*args, **kwargs): return System(LinearProbe(8))
            with patch.object(engine, "context", return_value=context), patch.object(engine, "require_smoke"), \
                    patch.object(engine, "check_cache"), patch.object(engine, "create_system", side_effect=model):
                first = engine.fit(root, cfg, output, 13, "cpu")
                second = engine.fit(root, cfg, output, 13, "cpu", resume=True)
                self.assertEqual(first["epochs"], 1)
                self.assertEqual(second["epochs"], 1)
                self.assertFalse(second["test_used_for_selection"])
                with self.assertRaises(ValueError):
                    engine.evaluate(root, cfg, output, 13, "test", False, "cpu")

    def test_online_lora_fit_does_not_update_base_weights(self):
        import torch
        from planetary_vlm.adaptation.common import engine
        from planetary_vlm.adaptation.common.encoder import LowRankLinear
        from planetary_vlm.adaptation.common.model import System
        from planetary_vlm.adaptation.frozen_decoder.model import PyramidDecoder
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Conv2d(1, 8, 1).requires_grad_(False)
                self.attention = LowRankLinear(torch.nn.Linear(8, 8), 2, 4., 0.)
            def forward(self, x):
                x = self.embedding(torch.nn.functional.adaptive_avg_pool2d(x, 4))
                x = self.attention(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                return [x, x, x, x]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            cfg["method"], cfg["seeds"] = "lora_decoder", [13]
            cfg["training"].update(max_epochs=1, batch_size=1, accumulate=1, thresholds=[.5])
            index = Index(root, cfg)
            output = experiment(root, "outputs/new")
            stats = prepare(index, output)
            encoder = TinyEncoder()
            before = encoder.embedding.weight.detach().clone()
            before_attention = encoder.attention.base.weight.detach().clone()
            system = System(PyramidDecoder(8, 8), encoder, adapted=True)
            with patch.object(engine, "context", return_value=(index, stats, {"fixture": True}, None, {})), \
                    patch.object(engine, "require_smoke"), patch.object(engine, "create_system", return_value=system):
                result = engine.fit(root, cfg, output, 13, "cpu")
            self.assertEqual(result["epochs"], 1)
            torch.testing.assert_close(before, encoder.embedding.weight)
            torch.testing.assert_close(before_attention, encoder.attention.base.weight)
            self.assertGreater(float(encoder.attention.b.detach().abs().sum()), 0)

    def test_scoring_resume_reuses_verified_masks_not_model_calls(self):
        import torch
        from planetary_vlm.adaptation.common.engine import Samples, predictions
        from planetary_vlm.adaptation.common.model import System
        from planetary_vlm.adaptation.linear_probe.model import LinearProbe
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = fixture(root)
            index = Index(root, cfg)
            stats = {"mean": .3, "std": .2}
            cache = root / "outputs/cache"
            cache.mkdir(parents=True)
            for row in index.split("val"):
                np.save(cache / (row["sample_id"] + ".npy"), np.ones((4, 8, 4, 4), dtype=np.float16))
            data = Samples(index, index.split("val"), stats, cache)
            system = System(LinearProbe(8))
            save = root / "outputs/evaluation/masks"
            first = predictions(system, data, "cpu", [.5], save)
            with patch.object(system, "forward", side_effect=AssertionError("Already completed")):
                second = predictions(system, data, "cpu", [.5], save)
            self.assertEqual(first, second)
            Path(first[.5][0]["prediction_mask"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                predictions(system, data, "cpu", [.5], save)


if __name__ == "__main__":
    unittest.main()
