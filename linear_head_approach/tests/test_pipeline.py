"""CPU fixtures only. No network, pretrained weights or real-data training."""
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from linear_head_approach.common import ROOT, atomic_json, code_hash, config, environment, identity_guard, read_json, safe_asset, sha, write_rows
from linear_head_approach.data import DATA, audit, bootstrap, metrics, scene_macro, validate_inputs


def metadata():
    result = []
    for i in range(120):
        split = "train" if i < 50 else "val" if i < 89 else "test"
        result.append({"sample_id": f"tile_{i:03d}", "split": split, "scene_group_id": f"scene_{i:03d}",
                       "source_dataset": "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection",
                       "source_revision": "20f800becb64a0c7ad314652a683f97f0281fe67", "source_image_id": i,
                       "rgb_image": f"data_orbital/fixture/rgb_{i}.png", "rgb_image_sha256": f"{i:064x}",
                       "native_vis": f"data_orbital/fixture/native_{i}.tif", "native_vis_sha256": f"{i+1000:064x}",
                       "display_transform": {"method": "fixture"}, "license_claim": "synthetic fixture only"})
    return result


def fixture(root):
    base = root / DATA
    base.mkdir(parents=True)
    inputs, targets = metadata(), []
    folder = root / "data_orbital/fixture"
    folder.mkdir(parents=True)
    y = np.zeros((512, 512), dtype=np.uint8)
    y[128:384, 128:384] = 1
    mask = folder / "mask.png"
    Image.fromarray(y).save(mask)
    for i, row in enumerate(inputs):
        path = root / row["rgb_image"]
        Image.new("RGB", (512, 512), (i, 0, 0)).save(path)
        row["rgb_image_sha256"] = sha(path)
        path = root / row["native_vis"]
        path.write_bytes(f"synthetic native fixture {i}".encode())
        row["native_vis_sha256"] = sha(path)
        targets.append({k: row[k] for k in ("sample_id", "split", "scene_group_id")}
                       | {"mask": "data_orbital/fixture/mask.png", "mask_sha256": sha(mask), "target": "catalog_crater_polygon"})
    write_rows(base / "inputs.jsonl", inputs)
    write_rows(base / "targets.jsonl", targets)
    return inputs, targets


class ContractTests(unittest.TestCase):
    def test_config_is_fixed(self):
        cfg = config()
        self.assertEqual(cfg["seeds"], [13, 37, 73])
        self.assertEqual(cfg["common_grid"], 24)
        self.assertFalse(cfg["feature_augmentation"])

    def test_input_contract(self):
        self.assertEqual(len(validate_inputs(metadata())), 120)

    def test_labels_forbidden_in_inputs(self):
        data = metadata()
        data[0]["mask"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "target information"):
            validate_inputs(data)

    def test_cross_split_groups_rejected(self):
        data = metadata()
        data[50]["scene_group_id"] = data[0]["scene_group_id"]
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_inputs(data)

    def test_duplicate_images_rejected(self):
        data = metadata()
        data[50]["rgb_image_sha256"] = data[0]["rgb_image_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate RGB"):
            validate_inputs(data)

    def test_source_revision_rejected(self):
        data = metadata()
        data[0]["source_revision"] = "untracked"
        with self.assertRaisesRegex(ValueError, "source/revision"):
            validate_inputs(data)

    def test_unsafe_ids_rejected(self):
        for unsafe in ("../outside", "x/y", "x\\y", "C:foo"):
            data = metadata()
            data[0]["sample_id"] = unsafe
            with self.assertRaisesRegex(ValueError, "Unsafe sample"):
                validate_inputs(data)

    def test_count_change_rejected(self):
        with self.assertRaisesRegex(ValueError, "cohort"):
            validate_inputs(metadata()[:-1])

    def test_identity_resume_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "identity.json"
            identity_guard(path, {"seed": 13})
            identity_guard(path, {"seed": 13})
            with self.assertRaisesRegex(ValueError, "Identity changed"):
                identity_guard(path, {"seed": 37})

    def test_safe_asset_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ("../outside", "data_orbital/../outside", "other/file", "/absolute"):
                with self.assertRaises(ValueError):
                    safe_asset(Path(tmp), bad)

    def test_metrics_empty_and_overlap(self):
        z = np.zeros((4, 4), bool)
        self.assertEqual(metrics(z, z)["iou"], 1)
        self.assertEqual(metrics(~z, z)["iou"], 0)
        self.assertEqual(metrics(~z, ~z)["dice"], 1)
        with self.assertRaisesRegex(ValueError, "shapes"):
            metrics(z, np.zeros((3, 3)))

    def test_group_macro_not_pixel_or_image_macro(self):
        data = [{"scene_group_id": "A", "iou": 1.}] * 9 + [{"scene_group_id": "B", "iou": 0.}]
        self.assertEqual(scene_macro(data), .5)
        self.assertEqual(bootstrap(data), bootstrap(data))
        self.assertEqual(bootstrap(data)["groups"], 2)

    def test_manifest_and_asset_hash_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, _ = fixture(root)
            self.assertEqual(audit(root)["split_counts"], {"train": 50, "val": 39, "test": 31})
            (root / inputs[0]["rgb_image"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                audit(root)

    def test_mask_unknown_class_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, targets = fixture(root)
            mask = root / targets[0]["mask"]
            Image.new("L", (512, 512), 2).save(mask)
            for row in targets:
                row["mask_sha256"] = sha(mask)
            write_rows(root / DATA / "targets.jsonl", targets)
            with self.assertRaisesRegex(ValueError, "binary"):
                audit(root)

    def test_zip_traversal_rejected(self):
        from linear_head_approach.bundle import unpack
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../outside", "payload")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
                unpack(path, root / "target", sha(path))
            self.assertFalse((root / "outside").exists())

    def test_zip_checksum_rejected(self):
        from linear_head_approach.bundle import unpack
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.zip"
            path.write_bytes(b"not a zip")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                unpack(path, Path(tmp) / "target", "0" * 64)

    def test_compare_requires_matching_seeds(self):
        from linear_head_approach.reporting import compare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for encoder, seed in (("ibm", 13), ("spacellava", 37)):
                atomic_json(root / encoder / "linear" / f"seed_{seed}" / "evaluation_test.json", {
                    "encoder": encoder, "seed": seed, "family": "linear", "protocol_sha256": "same",
                    "targets_sha256": "same", "head_parameters": 10, "scene_iou": {"mean": .5},
                    "rows": [{"sample_id": "a", "scene_group_id": "A", "iou": .5}]})
            with self.assertRaisesRegex(ValueError, "matching seeds"):
                compare(root)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch absent: run these synthetic software tests in the Colab venv")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def test_both_head_families_backward(self):
        from linear_head_approach.training import smoke
        with tempfile.TemporaryDirectory() as tmp:
            result = smoke(tmp, "cpu")
            self.assertFalse(result["scientific_result"])
            self.assertIn("spatial_1024", result)

    def test_mask_not_forward_argument(self):
        import inspect
        from linear_head_approach.heads import CraterHead
        self.assertEqual(list(inspect.signature(CraterHead.forward).parameters), ["self", "features", "output_size"])

    def test_deterministic_resize_matches_bilinear_geometry_and_gradients(self):
        import torch
        from torch.nn import functional as F
        from linear_head_approach.heads import resize_logits
        x = torch.randn(2, 1, 4, 4, dtype=torch.float64, requires_grad=True)
        actual = resize_logits(x, 11)
        expected = F.interpolate(x, (11, 11), mode="bilinear", align_corners=False)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
        ga = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
        gb = torch.autograd.grad(expected.square().sum(), x)[0]
        self.assertTrue(torch.allclose(ga, gb, atol=1e-10, rtol=1e-10))

    def test_interrupted_resume_matches_continuous_and_test_is_sealed(self):
        import torch
        import linear_head_approach.training as training
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, targets = fixture(root)
            cache = root / "features"
            cache.mkdir()
            records = []
            rng = np.random.default_rng(13)
            for row in inputs:
                feature = rng.normal(size=(4, 24, 24)).astype(np.float32)
                feature[0, 6:18, 6:18] += 2
                path = cache / f"{row['sample_id']}.npy"
                np.save(path, feature)
                records.append({k: row[k] for k in ("sample_id", "split", "scene_group_id")}
                               | {"file": path.name, "sha256": sha(path), "shape": list(feature.shape)})
            write_rows(cache / "features.jsonl", records)
            atomic_json(cache / "identity.json", {"encoder": "ibm", "assets": {"fixture": "no pretrained weights"}, "inputs_sha256": sha(root / DATA / "inputs.jsonl"),
                        "code": code_hash(), "environment": environment()})
            cfg = root / "fixture.toml"
            original_cfg = (ROOT / "linear_head_approach/config.toml").read_text()
            cfg.write_text(original_cfg.replace("max_epochs = 80", "max_epochs = 2").replace("patience = 12", "patience = 2"))
            continuous, resumed = root / "continuous", root / "resumed"
            training.fit(cache, continuous, "linear", 13, root=root, cfg_path=cfg, device="cpu")
            real_save = training.atomic_checkpoint

            def interrupted(path, state):
                real_save(path, state)
                if Path(path).name == "last.pt":
                    raise InterruptedError("Simulated disconnect after a completed epoch")

            with patch.object(training, "atomic_checkpoint", interrupted):
                with self.assertRaises(InterruptedError):
                    training.fit(cache, resumed, "linear", 13, root=root, cfg_path=cfg, device="cpu")
            training.fit(cache, resumed, "linear", 13, resume=True, root=root, cfg_path=cfg, device="cpu")
            a = torch.load(continuous / "last.pt", weights_only=False)
            b = torch.load(resumed / "last.pt", weights_only=False)
            self.assertEqual(a["epoch"], b["epoch"])
            for key in a["head"]:
                self.assertTrue(torch.equal(a["head"][key], b["head"][key]), key)
            self.assertEqual([r["validation_scene_iou"] for r in a["history"]], [r["validation_scene_iou"] for r in b["history"]])
            with self.assertRaisesRegex(ValueError, "Test sealed"):
                training.evaluate(cache, resumed, "test", root=root, device="cpu")
            evaluation = training.evaluate(cache, resumed, "val", root=root, device="cpu")
            self.assertEqual(evaluation["images"], 39)
            from linear_head_approach.deployment import publish
            with patch("linear_head_approach.encoders.assets", return_value={"fixture": "no pretrained weights"}):
                exported = publish(resumed, root / "mock_assets", root / "outputs/fixture_deployment", "synthetic-head", root)
            self.assertFalse(exported["test_evaluated"])
            self.assertEqual(sha(Path(exported["deployment"]).parent / "head.pt"), sha(resumed / "best.pt"))
            admitted = [__import__('json').loads(line) for line in (Path(exported["deployment"]).parent / "encoder_inputs.jsonl").read_text().splitlines()]
            self.assertEqual(len(admitted), 70)
            self.assertFalse(any("mask" in row or row["split"] == "train" for row in admitted))


if __name__ == "__main__":
    unittest.main()
