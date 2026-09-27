"""Offline adapter tests; no weights, network or ML imports needed."""
import sys
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from planetary_vlm.contracts import ModelRequest
from planetary_vlm.models import create_adapter
from planetary_vlm.models import register_adapter
from planetary_vlm.models.base import local_model_options


def request(images=(), modality_paths=()):
    return ModelRequest("r1", images, "Is a rock visible?", ("YES", "NO"), modality_paths=modality_paths)


class ModelTests(unittest.TestCase):
    def test_mock_fixture(self):
        adapter = create_adapter("mock", {"responses": {"r1": "YES"}})
        self.assertEqual(adapter.predict(request()).raw_response, "YES")
        adapter.close()

    def test_mock_never_chooses_allowed_answer_implicitly(self):
        self.assertEqual(create_adapter("mock", {}).predict(request()).raw_response, "UNKNOWN")

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            create_adapter("unregistered", {})

    def test_text_rejects_image_before_imports(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            result = create_adapter("hf_text", {}).predict(request(("missing.png",)))
        self.assertEqual(result.status, "unsupported")

    def test_visual_rejects_native_input(self):
        self.assertEqual(create_adapter("hf_vlm", {}).predict(request(modality_paths=(("native_tile", "tile"),))).status, "unsupported")

    def test_missing_local_path_returns_error(self):
        result = create_adapter("hf_text", {"model_path": "missing/local/checkpoint"}).predict(request())
        self.assertEqual(result.status, "error")
        self.assertIn("local model directory", result.error_message)

    def test_local_only_enforced(self):
        with self.assertRaises(ValueError):
            local_model_options({"model_path": "Qwen/model", "revision": "a" * 40, "local_files_only": False})

    def test_local_directory_options(self):
        with tempfile.TemporaryDirectory() as directory:
            options = local_model_options({"model_path": directory})
        self.assertTrue(options["local_files_only"])
        self.assertFalse(options["trust_remote_code"])

    def test_native_rejects_rover(self):
        self.assertEqual(create_adapter("seleno_native", {}).predict(request(("rover.png",))).status, "unsupported")

    def test_native_requires_explicit_protocol(self):
        result = create_adapter("seleno_native", {}).predict(request(modality_paths=(("native_tile", "tile"),)))
        self.assertEqual(result.status, "error")
        self.assertIn("prompt_protocol", result.error_message)

    def test_native_missing_paths_fail_before_ml_import(self):
        with patch.dict(sys.modules, {"torch": None}):
            result = create_adapter("seleno_native", {"prompt_protocol": "native_prompt_plus_instruction"}).predict(request(modality_paths=(("native_tile", "missing"),)))
        self.assertEqual(result.status, "error")
        self.assertIn("runtime_repo", result.error_message)

    def test_custom_registry(self):
        register_adapter("test_custom_factory", lambda options: options["sentinel"])
        self.assertEqual(create_adapter("test_custom_factory", {"sentinel": 7}), 7)
        with self.assertRaises(ValueError):
            register_adapter("mock", lambda options: None)

    def test_native_uses_request_prompt_without_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            tile = Path(directory)
            (tile / "inputs.pt").touch()
            adapter = create_adapter("seleno_native", {})
            captured = {}
            def generate(batch, **kwargs):
                captured["prompt"] = adapter.model.cfg["extra_user_instruction"]
                captured["generation"] = kwargs
                return ["YES"]
            adapter.model = SimpleNamespace(cfg={}, generate_interpretation=generate)
            adapter.torch = SimpleNamespace(manual_seed=lambda seed: None, inference_mode=nullcontext)
            native_request = request(modality_paths=(("native_tile", str(tile)),))
            with patch.object(adapter, "_paths", return_value={"runtime_repo": tile}), patch.object(adapter, "_load"), patch.object(adapter, "_batch", return_value={}):
                result = adapter.predict(native_request)
        self.assertEqual(result.status, "ok")
        self.assertEqual(captured["prompt"], native_request.prompt)
        self.assertFalse(captured["generation"]["do_sample"])

    def test_optional_import_error_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": None}):
            result = create_adapter("hf_text", {"model_path": directory}).predict(request())
        self.assertEqual(result.status, "error")
        self.assertIn("ModuleNotFoundError", result.error_message)

    def test_native_rejects_missing_bridge_weights_but_allows_unused_age_head(self):
        from planetary_vlm.models.seleno import SelenoNativeAdapter
        paths = {name: Path("fixture") for name in (
            "runtime_repo", "model_path", "tokenizer_path", "data_dir",
            "mae_checkpoint_path", "checkpoint_path")}
        for missing, should_fail in ((["resampler.weight", "age_head.weight"], True),
                                     (["age_head.weight"], False)):
            with self.subTest(missing=missing):
                model = SimpleNamespace(
                    named_parameters=lambda: [(name, SimpleNamespace(requires_grad=True))
                                              for name in ("resampler.weight", "age_head.weight")],
                    llm_bridge=SimpleNamespace(llm=SimpleNamespace(config=SimpleNamespace(use_cache=False))))
                runtime = SimpleNamespace(LLM_NAME="base", MAE_HPARAMS={},
                    resolve_mae_checkpoint=lambda: "mae", inference_cfg=lambda **kwargs: {},
                    build_geovlm=lambda **kwargs: model,
                    load_geovlm_weights=lambda *args: (missing, []))
                adapter = SelenoNativeAdapter({})
                with patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
                     patch("planetary_vlm.models.seleno.importlib.import_module", return_value=runtime):
                    if should_fail:
                        with self.assertRaisesRegex(RuntimeError, "missing trainable"):
                            adapter._load(paths)
                        self.assertIsNone(adapter.model)
                    else:
                        adapter._load(paths)
                        self.assertIs(adapter.model, model)
                self.assertEqual(runtime.LLM_NAME, "base")
                self.assertEqual(runtime.MAE_HPARAMS, {})
