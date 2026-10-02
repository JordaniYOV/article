"""Offline coverage; synthetic fixtures do not constitute model results."""

from dataclasses import asdict
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planetary_vlm.contracts import ModelRequest, ModelResponse
from planetary_vlm.inference.config import RunConfig
from planetary_vlm.inference.full_systems import load_pipeline, prepare_inputs, execute_pipeline
from planetary_vlm.inference.runner import file_hash, run
from planetary_vlm.io import write_jsonl, read_jsonl
from planetary_vlm.models.ibm_imp import restore_imp_png, save_maps, validate_head_keys, verify_saved_maps

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = Image = None


@unittest.skipIf(np is None or Image is None, "Optional image dependencies are missing")
class FullSystemsTests(unittest.TestCase):
    def fixture(self, root):
        image = root / "data_orbital" / "image.png"
        image.parent.mkdir()
        Image.fromarray(np.full((256, 256), 64, dtype=np.uint8)).save(image)
        manifest = image.parent / "manifest.jsonl"
        row = {"sample_id": "moon_seg_010", "planet": "moon", "mask_semantics": "imp",
            "source_dataset": "nasa-ibm-ai4science/Sombench-IMP-Segmentation",
            "source_split": "test", "evaluation_eligible": True,
            "image": "data_orbital/image.png", "image_sha256": file_hash(image),
            "mask": "data_orbital/DO_NOT_OPEN_MASK.tif", "mask_values_present": [0, 1],
            "scene_group_id": "scene", "source_revision": "fixture", "source_image_id": "tile",
            "license_claim": "fixture-only"}
        write_jsonl(manifest, [row, {**row, "sample_id": "train", "source_split": "train"},
                                {**row, "sample_id": "mars", "planet": "mars"}])
        config = {"dataset": {"manifest": manifest, "prepared_dir": image.parent / "prepared",
            "source_dataset": row["source_dataset"], "splits": ["val", "test"], "conditions": ["clean"]},
            "generation": {"seed": 42, "prompt": "Describe visible IMP evidence.", "max_new_tokens": 128},
            "distortion": {"max_shift_px": 4.0, "control_spacing_px": 32, "scan_axis": "rows",
                "num_hits": 80, "max_streak_length": 10, "flare_strength": .55, "shadow_strength": .65}}
        return config, image

    def test_prepare_does_not_read_targets_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, image = self.fixture(root)
            folder, metadata = prepare_inputs(config, root)
            requests = read_jsonl(folder / "requests.jsonl")
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]["modality_paths"], [])
            self.assertNotIn("mask", requests[0])
            self.assertNotIn("mask_values_present", requests[0])
            self.assertEqual(metadata["source_images"], 1)
            self.assertEqual(prepare_inputs(config, root)[0], folder)
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare_inputs(config, root)

    def test_all_conditions_and_image_only_geometry(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is optional")
        from planetary_vlm.distortion import apply_synchronous_shear, spoil_image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, image = self.fixture(root)
            config["dataset"]["conditions"] = ["clean", "shear", "radiation", "lens_flare", "hard_shadows", "combined"]
            folder, metadata = prepare_inputs(config, root)
            self.assertEqual(metadata["requests"], 6)
            pixels = np.array(Image.open(image))
            for effect in (apply_synchronous_shear, spoil_image):
                output, dem, seg = effect(pixels, image_only=True)
                paired = effect(pixels, seg_mask=np.zeros_like(pixels))[0]
                np.testing.assert_array_equal(output, paired)
                self.assertEqual(output.shape, pixels.shape)
                self.assertIsNone(dem)
                self.assertIsNone(seg)
                with self.assertRaises(ValueError):
                    effect(pixels, seg_mask=np.zeros_like(pixels), image_only=True)

    def test_reflectance_and_saved_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, image = self.fixture(root)
            values = restore_imp_png(image)
            self.assertAlmostEqual(float(values[0, 0]), 64 * .12 / 255, places=7)
            logits = np.zeros((2, 256, 256), dtype=np.float32)
            logits[1, 5:9, 5:9] = 2
            payload = save_maps(root / "maps", "../../bad:id", logits)
            raw = json.dumps(payload)
            verify_saved_maps(raw)
            with np.load(root / "maps" / payload["arrays_npz"]) as arrays:
                np.testing.assert_allclose(arrays["logits"], logits)
                np.testing.assert_allclose(arrays["probabilities"].sum(axis=0), 1, atol=1e-6)
            with Image.open(root / "maps" / payload["label_png"]) as png:
                labels = np.array(png)
            self.assertEqual(int(labels.sum()), 16)
            (root / "maps" / payload["label_png"]).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                verify_saved_maps(raw)
            with self.assertRaises(ValueError):
                save_maps(root / "maps", "bad", np.full_like(logits, np.nan))

    def test_head_keys_fail_closed(self):
        validate_head_keys(["model.encoder.block.weight", "criterion.weight"], [])
        for missing, unexpected in ((["model.decoder.weight"], []), (["model.neck.weight"], []),
                                    (["model.head.weight"], []), ([], ["foreign.weight"])):
            with self.assertRaises(ValueError):
                validate_head_keys(missing, unexpected)

    def test_load_once_close_on_failure_and_durable_resume(self):
        events = []
        class FixtureAdapter:
            def initialize(self):
                events.append("load")
                return {"checkpoint_sha256": "fixture"}
            def predict(self, request):
                events.append(request.request_id)
                if request.request_id == "bad":
                    raise ValueError("fixture failure")
                return ModelResponse(request.request_id, "fixture text", "ok", .1)
            def close(self):
                events.append("close")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, image = self.fixture(root)
            manifest = root / "requests.jsonl"
            write_jsonl(manifest, [asdict(ModelRequest(key, (str(image),), "Q", ())) for key in ("good", "bad")])
            config = RunConfig("fixture", manifest, root / "outputs", "fixture", "spacellava", {}, 42, "fixture", root / "config.toml")
            with patch("planetary_vlm.models.registry.create_adapter", return_value=FixtureAdapter()):
                destination = run(config)
                run(config, resume=True)
            self.assertEqual(events, ["load", "good", "bad", "close"])
            self.assertEqual([row["status"] for row in read_jsonl(destination / "predictions.jsonl")], ["ok", "error"])
            self.assertTrue((destination / "loading.json").is_file())
            events.clear()
            with patch.object(FixtureAdapter, "initialize", side_effect=RuntimeError("load fails")):
                with patch("planetary_vlm.models.registry.create_adapter", return_value=FixtureAdapter()):
                    with self.assertRaises(RuntimeError):
                        run(RunConfig("failed", manifest, root / "outputs", "fixture", "spacellava", {}, 42, "fixture", root / "config.toml"))
            self.assertEqual(events, ["close"])

    def test_spacellava_native_generation_decodes_only_returned_ids(self):
        from planetary_vlm.models.spacellava import SpaceLLaVAAdapter
        messages, calls = [], []
        class Tensor:
            def to(self, *args, **kwargs): return self
            def unsqueeze(self, *args): return self
        class Conversation:
            roles = ("user", "assistant")
            def copy(self): return self
            def append_message(self, role, text): messages.append((role, text))
            def get_prompt(self): return "fixture native conversation"
        returned_ids = object()
        def generate(ids, **kwargs):
            calls.append(kwargs)
            return returned_ids
        def decode(ids, **kwargs):
            self.assertIs(ids, returned_ids)
            return ["native generated answer"]
        modules = {
            "llava.constants": SimpleNamespace(DEFAULT_IMAGE_TOKEN="<image>", IMAGE_TOKEN_INDEX=-200),
            "llava.mm_utils": SimpleNamespace(process_images=lambda *args: Tensor(),
                tokenizer_image_token=lambda *args, **kwargs: Tensor()),
            "llava.conversation": SimpleNamespace(conv_templates={"llava_v1": Conversation()})}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, image = self.fixture(root)
            adapter = SpaceLLaVAAdapter({"runtime_repo": str(root)})
            adapter.torch = SimpleNamespace(inference_mode=nullcontext)
            adapter.processor = object()
            adapter.tokenizer = SimpleNamespace(batch_decode=decode)
            adapter.model = SimpleNamespace(config=SimpleNamespace(mm_use_im_start_end=False),
                get_vision_tower=lambda: SimpleNamespace(device="fixture", dtype="fixture"),
                get_input_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(device="fixture")), generate=generate)
            with patch("planetary_vlm.models.spacellava.importlib.import_module", side_effect=lambda name: modules[name]):
                response = adapter.predict(ModelRequest("fixture", (str(image),), "Visible IMP evidence?", (), 128))
            self.assertEqual(response.raw_response, "native generated answer")
            self.assertEqual(messages[0], ("user", "<image>\nVisible IMP evidence?"))
            self.assertEqual(calls[0]["image_sizes"], [(256, 256)])
            self.assertEqual(calls[0]["max_new_tokens"], 128)
            self.assertIs(calls[0]["do_sample"], False)

    def test_sequential_systems_and_independent_asset_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, image = self.fixture(root)
            models = {name: RunConfig("fixture", root / "unused", root / "outputs" / name, name,
                backend, {}, 42, "fixture", root / "config.toml") for name, backend in
                (("SpaceLLaVA", "spacellava"), ("NASA-IBM-Lunar-Foundation-Model", "ibm_imp"))}
            events = []
            class Adapter:
                def preflight(self): pass
            def run_fixture(item, **kwargs):
                events.append(item.backend)
                self.assertEqual(read_jsonl(item.requests)[0]["image_paths"], [str(image)])
                destination = item.output_dir / item.run_id
                write_jsonl(destination / "predictions.jsonl", [{"status": "ok"}])
                return destination
            with patch("planetary_vlm.inference.full_systems.load_pipeline", return_value=(config, root, models)), \
                 patch("planetary_vlm.models.registry.create_adapter", return_value=Adapter()), \
                 patch("planetary_vlm.inference.full_systems.run", side_effect=run_fixture):
                result = execute_pipeline(root / "config.toml")
            self.assertEqual(events, ["spacellava", "ibm_imp"])
            self.assertTrue(all(item["status"] == "finished" for item in result["models"].values()))
            with patch("planetary_vlm.inference.full_systems.load_pipeline", return_value=(config, root, models)), \
                 patch("planetary_vlm.models.registry.create_adapter", side_effect=[ValueError("Space weights absent"), Adapter()]):
                result = execute_pipeline(root / "config.toml", check_only=True)
            self.assertEqual(result["models"]["SpaceLLaVA"]["status"], "blocked")
            self.assertEqual(result["models"]["NASA-IBM-Lunar-Foundation-Model"]["status"], "assets_available")


class PipelineConfigTests(unittest.TestCase):
    def test_real_configs_resolve_to_two_model_folders(self):
        config, root, models = load_pipeline(ROOT / "configs/full_systems.toml")
        self.assertEqual(root, ROOT)
        self.assertEqual(len(models), 2)
        for name, item in models.items():
            self.assertEqual(item.output_dir, ROOT / "outputs" / name)
            self.assertTrue(item.options["local_files_only"])
            self.assertEqual(len(item.options["code_revision"]), 40)


if __name__ == "__main__":
    unittest.main()
