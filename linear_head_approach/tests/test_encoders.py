"""CLIP namespace regression checks: fake tensors only, no Torch or downloads."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from linear_head_approach import encoders


def vision_names():
    names = ["embeddings.class_embedding", "embeddings.patch_embedding.weight",
             "embeddings.position_embedding.weight", "pre_layrnorm.weight", "pre_layrnorm.bias",
             "post_layernorm.weight", "post_layernorm.bias"]
    for layer in range(24):
        for module in ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.out_proj",
                       "layer_norm1", "layer_norm2", "mlp.fc1", "mlp.fc2"):
            for parameter in ("weight", "bias"):
                names.append(f"encoder.layers.{layer}.{module}.{parameter}")
    return names


class ClipKeysTests(unittest.TestCase):
    def setUp(self):
        self.flat = vision_names()
        self.wrapped = ["vision_model." + name for name in self.flat]
        self.source = [encoders.SPACE_PREFIX + name for name in self.wrapped]
        self.assertEqual(len(self.source), 391)

    def test_flat_layout_maps_all_391_without_changing_tensor_identity(self):
        mapping, layout = encoders.clip_key_map(self.source, self.flat)
        self.assertEqual(layout, "flat")
        self.assertEqual(mapping, dict(zip(self.flat, self.source)))

    def test_wrapped_layout_preserves_legacy_keys(self):
        mapping, layout = encoders.clip_key_map(self.source, self.wrapped)
        self.assertEqual(layout, "wrapped")
        self.assertEqual(mapping, dict(zip(self.wrapped, self.source)))

    def test_extra_missing_or_mixed_model_keys_rejected(self):
        for keys in (self.flat[:-1], self.flat + ["visual_projection.weight"],
                     self.flat[:200] + self.wrapped[200:]):
            with self.subTest(keys=len(keys)), self.assertRaisesRegex(ValueError, "namespace mismatch"):
                encoders.clip_key_map(self.source, keys)

    def test_missing_checkpoint_tensor_rejected(self):
        with self.assertRaisesRegex(ValueError, "namespace mismatch"):
            encoders.clip_key_map(self.source[:-1], self.flat)

    def test_unknown_source_namespace_rejected(self):
        keys = self.source.copy()
        keys[0] = "model.language_model.weight"
        with self.assertRaisesRegex(ValueError, "only pinned"):
            encoders.clip_key_map(keys, self.flat)

    def test_duplicate_or_empty_keys_rejected(self):
        for source, model in (([], []), (self.source + self.source[:1], self.flat),
                              (self.source, self.flat + self.flat[:1])):
            with self.subTest(source=len(source), model=len(model)), self.assertRaisesRegex(ValueError, "duplicate"):
                encoders.clip_key_map(source, model)

    def exercise_loader(self, wrapped=False, malformed_model=False, tensor_error=None):
        names = self.wrapped if wrapped else self.flat
        if malformed_model:
            names = names[:-1]
        parameter = SimpleNamespace(requires_grad=True)
        model = Mock()
        model.state_dict.return_value = dict.fromkeys(names)
        model.float.return_value = model
        model.to.return_value = model
        model.eval.return_value = model
        model.parameters.return_value = [parameter]
        model.requires_grad_.side_effect = lambda flag: setattr(parameter, "requires_grad", flag)
        model.load_state_dict.side_effect = tensor_error
        tensors = {key: object() for key in self.source}
        get_tensor = Mock(side_effect=tensors.__getitem__)
        safe_open = Mock()
        safe_open.return_value.__enter__ = Mock(return_value=SimpleNamespace(get_tensor=get_tensor))
        safe_open.return_value.__exit__ = Mock(return_value=False)
        torch = ModuleType("torch")
        torch.use_deterministic_algorithms = Mock()
        torch.backends = SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, allow_tf32=True),
                                        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)))
        transformers = ModuleType("transformers")
        transformers.CLIPConfig = SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(vision_config=object())))
        transformers.CLIPVisionModel = Mock(return_value=model)
        transformers.CLIPImageProcessor = SimpleNamespace(from_pretrained=Mock(return_value=object()))
        safetensors = ModuleType("safetensors")
        safetensors.safe_open = safe_open
        configs = {
            "config.json": {"mm_vision_tower": "openai/clip-vit-large-patch14-336", "mm_vision_select_layer": -2},
            "model.safetensors.index.json": {"weight_map": dict.fromkeys(self.source, "model-00006-of-00006.safetensors")},
        }
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "safetensors": safetensors}), \
                patch.object(encoders, "assets", return_value={"fixture": True}) as assets, \
                patch.object(encoders, "read_json", side_effect=lambda path: configs[Path(path).name]):
            if malformed_model:
                with self.assertRaisesRegex(ValueError, "namespace mismatch"):
                    encoders.load(Path("fixture_weights"), "spacellava", device="cpu")
                safe_open.assert_not_called()
                model.load_state_dict.assert_not_called()
                return
            if tensor_error:
                with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                    encoders.load(Path("fixture_weights"), "spacellava", device="cpu")
                self.assertEqual(model.load_state_dict.call_count, 1)
                self.assertTrue(model.load_state_dict.call_args.kwargs["strict"])
                return
            loaded, forward, verified = encoders.load(Path("fixture_weights"), "spacellava", device="cpu")
            self.assertIs(loaded, model)
            self.assertTrue(callable(forward))
            self.assertEqual(verified, {"fixture": True})
            assets.assert_called_once_with(Path("fixture_weights"), "spacellava")
        model.load_state_dict.assert_called_once()
        self.assertTrue(model.load_state_dict.call_args.kwargs["strict"])
        self.assertEqual(get_tensor.call_count, 391)
        state = model.load_state_dict.call_args.args[0]
        self.assertEqual(set(state), set(names))
        for name, source in zip(names, self.source):
            self.assertIs(state[name], tensors[source])
        self.assertFalse(parameter.requires_grad)
        transformers.CLIPConfig.from_pretrained.assert_called_once_with(str(Path("fixture_weights/clip_config")), local_files_only=True)
        transformers.CLIPImageProcessor.from_pretrained.assert_called_once_with(str(Path("fixture_weights/spacellava")), local_files_only=True)

    def test_real_loader_branch_flat_and_wrapped_are_strict_and_offline(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                self.exercise_loader(wrapped=wrapped)

    def test_loader_rejects_partial_model_before_reading_tensors(self):
        self.exercise_loader(malformed_model=True)

    def test_tensor_load_error_is_not_silenced_or_retried_non_strictly(self):
        self.exercise_loader(tensor_error=RuntimeError("size mismatch for fixture tensor"))
