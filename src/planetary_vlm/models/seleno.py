"""Native SelenoVLM integration, requiring a separately provisioned upstream repo.

No rover RGB compatibility is implied. The user-provisioned upstream runtime is
imported locally. Native tile tensors use torch's restricted weights-only loader.
"""
from collections.abc import Mapping
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from planetary_vlm.contracts import ModelRequest, ModelResponse
from .base import modalities


@contextmanager
def offline_runtime(repo: Path):
    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous_env = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update({key: "1" for key in keys})
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        yield
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class SelenoNativeAdapter:
    REQUIRED = ("wac", "elevation", "pos", "geomap", "clementine")
    CAPTIONS = ("primary_units", "dominant_processes", "stratigraphy", "interpretation")

    def __init__(self, options: Mapping[str, Any]):
        self.options = dict(options)
        self.model = self.runtime = self.torch = None

    def _paths(self) -> dict[str, Path]:
        allowed = {"runtime_repo", "model_path", "checkpoint_path", "mae_checkpoint_path", "tokenizer_path", "data_dir", "code_revision", "local_files_only", "device", "prompt_protocol", "seed"}
        unknown = set(self.options) - allowed
        if unknown:
            raise ValueError(f"Unknown Seleno adapter options (native quantization is not implemented): {sorted(unknown)}")
        if self.options.get("local_files_only", True) is not True:
            raise ValueError("Only local_files_only=true is supported")
        if self.options.get("prompt_protocol") != "native_prompt_plus_instruction":
            raise ValueError("prompt_protocol must explicitly be native_prompt_plus_instruction")
        result = {}
        for key in ("runtime_repo", "model_path", "checkpoint_path", "mae_checkpoint_path", "tokenizer_path", "data_dir"):
            value = self.options.get(key)
            if not value:
                raise ValueError(f"Required local option: {key}")
            path = Path(str(value)).resolve()
            expected_file = key in {"checkpoint_path", "mae_checkpoint_path"}
            if not (path.is_file() if expected_file else path.is_dir()):
                raise ValueError(f"Missing local {key}: {path}")
            result[key] = path
        if not (result["runtime_repo"] / "model/vlm/runtime.py").is_file():
            raise ValueError("runtime_repo does not contain model/vlm/runtime.py")
        if not (result["data_dir"] / "UnifiedGeoMap/legend.pkl").is_file():
            raise ValueError("data_dir must contain UnifiedGeoMap/legend.pkl")
        return result

    def _load(self, paths: dict[str, Path]) -> None:
        if self.model is not None:
            return
        import torch
        self.torch = torch
        # Upstream absolute imports use generic module names; reject another repo's
        # cached package rather than silently importing the wrong implementation.
        for name in ("model", "model.vlm", "model.vlm.runtime", "data.modalityInfo"):
            loaded = sys.modules.get(name)
            filename = getattr(loaded, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(paths["runtime_repo"]):
                raise RuntimeError(f"Conflicting imported module {name}; use a fresh process")
        rt = importlib.import_module("model.vlm.runtime")
        original_llm = rt.LLM_NAME
        original_hparams = rt.MAE_HPARAMS
        original_resolver = rt.resolve_mae_checkpoint
        try:
            rt.LLM_NAME = str(paths["model_path"])
            rt.MAE_HPARAMS = dict(original_hparams, tokenizer_path=str(paths["tokenizer_path"]), data_dir=str(paths["data_dir"]))
            rt.resolve_mae_checkpoint = lambda *args, **kwargs: paths["mae_checkpoint_path"]
            cfg = rt.inference_cfg(closed_book=True)
            cfg.pop("prose_targets", None)
            model = rt.build_geovlm(device=str(self.options.get("device", "cuda:0")), cfg=cfg, retriever=None, load=False)
            missing, unexpected = rt.load_geovlm_weights(model, paths["checkpoint_path"])
            if unexpected:
                raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected}")
            missing_trainable = set(missing) & {name for name, parameter in model.named_parameters() if parameter.requires_grad and "age_head" not in name.split(".")}
            if missing_trainable:
                raise RuntimeError(f"Checkpoint missing trainable bridge parameters: {sorted(missing_trainable)}")
            model.llm_bridge.llm.config.use_cache = True
            self.model = model
            self.runtime = rt
        finally:
            rt.LLM_NAME = original_llm
            rt.MAE_HPARAMS = original_hparams
            rt.resolve_mae_checkpoint = original_resolver

    def _batch(self, tile: Path):
        # Never read meta.json, reference captions, or target parquet.
        sample = self.torch.load(tile / "inputs.pt", map_location="cpu", weights_only=True)
        if not isinstance(sample, dict) or not set(self.REQUIRED).issubset(sample):
            raise ValueError("Native inputs.pt must contain wac/elevation/geomap/clementine/pos")
        info = importlib.import_module("data.modalityInfo").MODALITY_INFO
        device = str(self.options.get("device", "cuda:0"))
        batch = {}
        for name in self.REQUIRED:
            entry = sample[name]
            if not isinstance(entry, dict) or not entry:
                raise ValueError(f"Invalid native modality: {name}")
            batch[name] = {key: (value.unsqueeze(0).to(device) if isinstance(value, self.torch.Tensor) else [value]) for key, value in entry.items()}
        # Discard any supplied caption targets; only upstream-defined blank pads.
        for name in self.CAPTIONS:
            batch[name] = {"text": [""], "tokens": self.torch.zeros(1, info[name]["max_tokens"], dtype=self.torch.long, device=device)}
        return batch

    def predict(self, request: ModelRequest) -> ModelResponse:
        started = perf_counter()
        native = modalities(request)
        if request.image_paths or set(native) != {"native_tile"}:
            return ModelResponse(request.request_id, "", "unsupported", 0.0, "Seleno native accepts only a native_tile directory, not rover RGB")
        try:
            paths = self._paths()
            tile = Path(native["native_tile"]).resolve()
            if not (tile / "inputs.pt").is_file():
                raise ValueError("native_tile must contain inputs.pt")
            with offline_runtime(paths["runtime_repo"]):
                self._load(paths)
                batch = self._batch(tile)
                # This preserves the upstream native geology prompt and APPENDS the
                # actual benchmark question. It is a separate prompt protocol.
                self.model.cfg["extra_user_instruction"] = request.prompt
                self.torch.manual_seed(int(self.options.get("seed", 42)))
                with self.torch.inference_mode():
                    output = self.model.generate_interpretation(batch, max_new_tokens=request.max_new_tokens, do_sample=request.do_sample)
                return ModelResponse(request.request_id, output[0].strip(), "ok", perf_counter() - started)
        except Exception as error:
            return ModelResponse(request.request_id, "", "error", perf_counter() - started, f"{type(error).__name__}: {error}")

    def close(self) -> None:
        self.model = self.runtime = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
