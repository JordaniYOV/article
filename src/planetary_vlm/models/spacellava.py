"""Merged Remyx SpaceLLaVA checkpoint, using the publisher's LLaVA API.

Only local checkpoints are accepted. GPU/quantization support needs a smoke test.
"""

import importlib
import json
from pathlib import Path
from time import perf_counter

from planetary_vlm.contracts import ModelResponse
from .local_runtime import checked_repo, hashes, import_local, local_path, offline_imports, release


class SpaceLLaVAAdapter:
    def __init__(self, options):
        self.options = dict(options)
        self.model = self.torch = self.tokenizer = self.processor = None

    def preflight(self):
        if self.options.get("local_files_only", True) is not True:
            raise ValueError("Only offline inference is supported")
        repo = checked_repo(self.options, "llava")
        model = local_path(self.options, "model_dir", directory=True)
        vision = local_path(self.options, "vision_dir", directory=True)
        for folder, names in ((model, ("config.json", "model.safetensors.index.json", "preprocessor_config.json")),
                              (vision, ("config.json", "preprocessor_config.json"))):
            for name in names:
                if not (folder / name).is_file():
                    raise ValueError(f"Missing checkpoint file: {folder / name}")
        config = json.loads((model / "config.json").read_text())
        if config.get("model_type") != "llava" or config.get("mm_vision_tower") != "openai/clip-vit-large-patch14-336":
            raise ValueError("Expected the audited merged SpaceLLaVA / CLIP-336 checkpoint")
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        if not any(key.startswith("model.vision_tower.vision_tower.") for key in index):
            raise ValueError("Merged checkpoint must contain the actual SpaceLLaVA visual tower")
        shards = []
        for name in sorted(set(index.values())):
            path = (model / name).resolve()
            if not path.is_relative_to(model) or not path.is_file():
                raise ValueError(f"Missing/unsafe checkpoint shard: {name}")
            shards.append(path)
        if self.options.get("quantization", "4bit") not in {"4bit", "none"}:
            raise ValueError("quantization must be 4bit or none")
        return repo, model, vision, shards

    def initialize(self):
        repo, folder, vision, shards = self.preflight()
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
        self.torch = torch
        torch.manual_seed(self.options.get("seed", 42))
        with offline_imports(repo):
            import_local("llava", repo)
            api = importlib.import_module("llava.model.language_model.llava_llama")
            cfg = api.LlavaConfig.from_pretrained(str(folder), local_files_only=True)
            cfg.mm_vision_tower = str(vision)
            # Construct the tower BEFORE loading the merged checkpoint so its
            # trained tensors are restored, rather than overwritten later by CLIP.
            cfg.unfreeze_mm_vision_tower = True
            kwargs = dict(config=cfg, local_files_only=True, low_cpu_mem_usage=True,
                          torch_dtype=torch.float16, device_map=self.options.get("device_map", "auto"),
                          output_loading_info=True)
            if self.options.get("quantization", "4bit") == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    llm_int8_skip_modules=["vision_tower", "mm_projector", "lm_head"])
            self.tokenizer = AutoTokenizer.from_pretrained(str(folder), use_fast=False, local_files_only=True)
            self.model, info = api.LlavaLlamaForCausalLM.from_pretrained(str(folder), **kwargs)
            if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
                raise RuntimeError(f"Checkpoint does not fully restore this architecture: {info}")
            constants = importlib.import_module("llava.constants")
            tokens = []
            if getattr(cfg, "mm_use_im_patch_token", True):
                tokens.append(constants.DEFAULT_IMAGE_PATCH_TOKEN)
            if getattr(cfg, "mm_use_im_start_end", False):
                tokens.extend([constants.DEFAULT_IM_START_TOKEN, constants.DEFAULT_IM_END_TOKEN])
            self.tokenizer.add_tokens(tokens, special_tokens=True)
            if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
                raise ValueError("Tokenizer size differs from the merged checkpoint; refusing random embeddings")
            self.model.eval().requires_grad_(False)
            self.processor = CLIPImageProcessor.from_pretrained(str(folder), local_files_only=True)
        files = [folder / "config.json", folder / "preprocessor_config.json", folder / "model.safetensors.index.json", *shards]
        if (folder / "generation_config.json").is_file():
            files.append(folder / "generation_config.json")
        files.extend(path for path in folder.glob("*token*") if path.is_file())
        files.extend([vision / "config.json", vision / "preprocessor_config.json"])
        return {"checkpoint_sha256": hashes(files), "code_revision": self.options["code_revision"],
                "conversation": "llava_v1", "quantization": self.options.get("quantization", "4bit"),
                "device_map": str(self.model.hf_device_map) if hasattr(self.model, "hf_device_map") else str(self.model.device)}

    def predict(self, request):
        start = perf_counter()
        if len(request.image_paths) != 1 or request.modality_paths:
            return ModelResponse(request.request_id, "", "unsupported", 0, "Exactly one image is supported")
        if request.do_sample:
            raise ValueError("This protocol requires deterministic generation")
        if self.model is None:
            self.initialize()
        from PIL import Image
        repo = Path(self.options["runtime_repo"]).resolve()
        with offline_imports(repo):
            constants = importlib.import_module("llava.constants")
            utils = importlib.import_module("llava.mm_utils")
            conv = importlib.import_module("llava.conversation").conv_templates["llava_v1"].copy()
            image_token = constants.DEFAULT_IMAGE_TOKEN
            if getattr(self.model.config, "mm_use_im_start_end", False):
                image_token = constants.DEFAULT_IM_START_TOKEN + image_token + constants.DEFAULT_IM_END_TOKEN
            conv.append_message(conv.roles[0], image_token + "\n" + request.prompt)
            conv.append_message(conv.roles[1], None)
            with Image.open(request.image_paths[0]) as source:
                image = source.convert("RGB")
            tower = self.model.get_vision_tower()
            pixels = utils.process_images([image], self.processor, self.model.config)
            if isinstance(pixels, list):
                pixels = [value.to(tower.device, dtype=tower.dtype) for value in pixels]
            else:
                pixels = pixels.to(tower.device, dtype=tower.dtype)
            device = self.model.get_input_embeddings().weight.device
            ids = utils.tokenizer_image_token(conv.get_prompt(), self.tokenizer, constants.IMAGE_TOKEN_INDEX,
                                             return_tensors="pt").unsqueeze(0).to(device)
            with self.torch.inference_mode():
                output = self.model.generate(ids, images=pixels, image_sizes=[image.size],
                    do_sample=False, num_beams=1, max_new_tokens=request.max_new_tokens, use_cache=True)
            # Native LLaVA generates from inputs_embeds: decode the returned IDs,
            # without assuming a Qwen-style input-token prefix.
            answer = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        return ModelResponse(request.request_id, answer, "ok", perf_counter() - start)

    def close(self):
        self.tokenizer = self.processor = None
        release(self)
