"""Local-only Qwen adapters; real inference still requires a hardware pilot."""
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any
from planetary_vlm.contracts import ModelRequest, ModelResponse
from .base import local_model_options, modalities


class HuggingFaceAdapter:
    def __init__(self, options: Mapping[str, Any], *, visual: bool):
        self.options = dict(options)
        self.visual = visual
        self.model = self.processor = self.torch = None

    def _load(self) -> None:
        if self.model is not None:
            return
        allowed = {"model_path", "model_name", "revision", "local_files_only", "dtype", "device", "seed", "min_pixels", "max_pixels", "quantization"}
        unknown = set(self.options) - allowed
        if unknown:
            raise ValueError(f"Unknown HF adapter options: {sorted(unknown)}")
        kwargs = local_model_options(self.options)
        import torch
        import transformers
        self.torch = torch
        source = str(self.options.get("model_path") or self.options["model_name"])
        dtype_name = str(self.options.get("dtype", "bfloat16"))
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16 or bfloat16")
        loader = transformers.Qwen2_5_VLForConditionalGeneration if self.visual else transformers.AutoModelForCausalLM
        processing = transformers.AutoProcessor if self.visual else transformers.AutoTokenizer
        processing_kwargs = dict(kwargs)
        for key in ("min_pixels", "max_pixels"):
            if key in self.options:
                if not self.visual or type(self.options[key]) is not int or self.options[key] < 1:
                    raise ValueError(f"{key} requires a positive integer and a visual backend")
                processing_kwargs[key] = self.options[key]
        if self.options.get("min_pixels", 0) > self.options.get("max_pixels", float("inf")):
            raise ValueError("min_pixels exceeds max_pixels")
        self.processor = processing.from_pretrained(source, **processing_kwargs)
        quantization = self.options.get("quantization", "none")
        device = str(self.options.get("device", "cuda:0"))
        if quantization == "bnb4":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=getattr(torch, dtype_name))
            kwargs["device_map"] = {"": device}
        elif quantization != "none":
            raise ValueError("quantization must be none or bnb4")
        self.model = loader.from_pretrained(source, torch_dtype=getattr(torch, dtype_name), **kwargs)
        if quantization == "none":
            self.model.to(device)
        self.model.eval()

    def predict(self, request: ModelRequest) -> ModelResponse:
        started = perf_counter()
        if modalities(request) or (not self.visual and request.image_paths):
            return ModelResponse(request.request_id, "", "unsupported", 0.0, "Backend does not support supplied visual/native modalities")
        try:
            self._load()
            images = []
            try:
                if self.visual:
                    from PIL import Image
                    for image_path in request.image_paths:
                        with Image.open(Path(image_path)) as source_image:
                            images.append(source_image.convert("RGB"))
                    content = [{"type": "image", "image": image} for image in images]
                    content.append({"type": "text", "text": request.prompt})
                    text = self.processor.apply_chat_template([{ "role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text=[text], images=images or None, return_tensors="pt", padding=True)
                else:
                    text = self.processor.apply_chat_template([{ "role": "user", "content": request.prompt}], tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                self.torch.manual_seed(int(self.options.get("seed", 42)))
                with self.torch.inference_mode():
                    outputs = self.model.generate(**inputs, max_new_tokens=request.max_new_tokens, do_sample=request.do_sample)
                generated = outputs[:, inputs["input_ids"].shape[1]:]
                decoded = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                return ModelResponse(request.request_id, decoded, "ok", perf_counter() - started)
            finally:
                for image in images:
                    image.close()
        except Exception as error:
            return ModelResponse(request.request_id, "", "error", perf_counter() - started, f"{type(error).__name__}: {error}")

    def close(self) -> None:
        self.model = self.processor = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
