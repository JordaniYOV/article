"""Standalone upstream-builder control. Deliberately no project runtime imports.

Uses pinned LLaVA APIs, not SpaceLLaVAAdapter, runner, prepare_inputs or the
project's generation/diagnostic helpers. Still shares Torch/Transformers/LLaVA
with the baseline; this is NOT a wholly independent engine or publisher oracle.
"""

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback
from types import SimpleNamespace

SCHEMA = "spacellava-upstream-reference-v1"
PRECISION_SCHEMA = "spacellava-fp16-nf4-pair-v1"
RUNTIME_COMMIT = "c121f0432da27facab705978f83c4ada465e46fd"
GENERATION = dict(do_sample=False, num_beams=1, max_new_tokens=96, use_cache=True,
    repetition_penalty=1.0, no_repeat_ngram_size=0, return_dict_in_generate=True, output_scores=True)
PROBE = "Describe the visible contents of this image."


def gpu_identity(torch):
    """Stable identity, not free-memory values which change during a run."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Precision control requires exactly one visible CUDA GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "0":
        raise ValueError("Precision control requires CUDA_VISIBLE_DEVICES=0 for unambiguous GPU identity")
    uuid = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not uuid.startswith("GPU-") or len(uuid.splitlines()) != 1:
        raise ValueError("Cannot establish one physical GPU UUID")
    props = torch.cuda.get_device_properties(0)
    return {"uuid": uuid, "name": props.name, "total_memory_bytes": props.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0))}


def precision_admission(torch, root):
    """Conservative free-memory gate, NOT a guarantee against inference OOM."""
    identity = gpu_identity(torch)
    checkpoint = guarded(Path(root) / "weights/SpaceLLaVA/model.safetensors.index.json")
    size = None
    if checkpoint.is_file():
        size = json.loads(checkpoint.read_text())["metadata"]["total_size"]
        if type(size) is not int or not 20_000_000_000 < size < 40_000_000_000:
            raise ValueError("Unexpected FP16 checkpoint size in index metadata")
    minimum = max(32 * 1024**3, (size or 0) + 6 * 1024**3)
    free, total = torch.cuda.mem_get_info(0)
    record = {"gpu_identity": identity, "free_memory_bytes": free, "total_memory_bytes": total,
        "checkpoint_bytes": size, "required_free_bytes": minimum,
        "admitted": free >= minimum, "limitation": "Conservative gate; actual load/forward/generation may still OOM. No offload or precision fallback."}
    if free < minimum:
        raise ValueError("FP16/NF4 pair needs at least " + str(round(minimum / 1024**3, 1))
            + " GiB FREE GPU memory; current GPU has " + str(round(free / 1024**3, 1))
            + " GiB. T4/16-GB GPUs are not admitted. Use a larger GPU and rerun BOTH arms there. " + json.dumps(record))
    return record


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def guarded(path):
    path = Path(path).resolve()
    if "data_rover" in {p.casefold() for p in path.parts}:
        raise ValueError("Protected archive is out of scope")
    return path


def write_json(path, value):
    path = guarded(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def array_summary(value):
    import numpy as np
    value = np.asarray(value, dtype=np.float32)
    if not bool(np.isfinite(value).all()):
        raise ValueError("Nonfinite reference tensor")
    return {"shape": list(value.shape), "sha256_float32": hashlib.sha256(value.tobytes()).hexdigest(),
        "min": float(value.min()), "max": float(value.max()), "std": float(value.std(dtype=np.float64))}


def verify_tensors(model, folder, torch, *, all_weights=False):
    """Exact visual weights plus five unquantized LM tensors; not all LM values."""
    from safetensors import safe_open
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    state = model.state_dict()
    visual = {k for k in index if k.startswith(("model.vision_tower.", "model.mm_projector."))}
    if len(visual) != 395 or {k for k in state if k.startswith(("model.vision_tower.", "model.mm_projector."))} != visual:
        raise ValueError("Reference visual checkpoint key coverage changed")
    floating_lm = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
        "model.layers.0.input_layernorm.weight", "model.layers.39.post_attention_layernorm.weight"}
    meta = [n for n, v in (*model.named_parameters(), *model.named_buffers()) if v.is_meta]
    if meta or set(index) - set(state):
        raise ValueError("Reference has meta or missing checkpoint tensors")
    selected = set(index) if all_weights else visual | floating_lm
    verified = []
    for shard in sorted({index[k] for k in selected}):
        path = guarded(folder / shard)
        if not path.is_relative_to(folder):
            raise ValueError("Unsafe checkpoint shard")
        with safe_open(str(path), framework="pt", device="cpu") as source:
            for name in sorted(k for k in selected if index[k] == shard):
                actual = state[name]
                if actual.dtype != torch.float16:
                    raise ValueError("Reference protected tensor is not FP16: " + name)
                expected = source.get_slice(name)
                if list(actual.shape) != expected.get_shape():
                    raise ValueError("Reference tensor shape mismatch: " + name)
                # Bound host memory even for the 32000 x 5120 embedding/head.
                for start in range(0, actual.shape[0], 512):
                    a = actual[start:start + 512].detach().cpu()
                    b = expected[start:start + 512].to(dtype=actual.dtype)
                    if not bool(torch.isfinite(a).all().item()) or not torch.equal(a, b):
                        raise ValueError("Reference tensor differs from checkpoint: " + name)
                verified.append(name)
    return {"visual_tensors_exact": len(visual), "floating_language_tensors_exact": sorted(floating_lm),
        "remaining_meta_tensors": 0, "all_language_weight_values_compared": all_weights,
        "checkpoint_tensors_exact": len(verified),
        "comparison": "exact after casting checkpoint to FP16; chunked host copies"}


def load_runtime(job, folder):
    """Native load_pretrained_model with explicit matched NF4 and merged vision."""
    root = guarded(job["workspace"])
    paired = job.get("schema") == PRECISION_SCHEMA
    precision = job.get("precision", "nf4")
    if precision not in {"fp16", "nf4"} or (not paired and precision != "nf4"):
        raise ValueError("Unsupported precision control")
    repo, model_dir, vision_dir = (guarded(root / p) for p in
        ("external/LLaVA", "weights/SpaceLLaVA", "weights/clip-vit-large-patch14-336"))
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != RUNTIME_COMMIT or dirty.strip():
        raise ValueError("Reference requires the clean pinned LLaVA runtime")
    actual_hashes = {}
    print(json.dumps({"event": "reference_stage", "stage": "verify_checkpoint_files"}), flush=True)
    for relative, expected in job["checkpoint_sha256"].items():
        parts = relative.split("/")
        if (len(parts) != 2 or parts[0] not in {"SpaceLLaVA", "clip-vit-large-patch14-336"}
                or parts[1] in {".", ".."} or Path(parts[1]).name != parts[1]):
            raise ValueError("Unsafe checkpoint identity")
        path = guarded((model_dir if parts[0] == "SpaceLLaVA" else vision_dir) / parts[1])
        actual_hashes[relative] = file_hash(path)
        if actual_hashes[relative] != expected:
            raise ValueError("Reference checkpoint SHA-256 differs: " + relative)
    config_data = json.loads((model_dir / "config.json").read_text())
    if (config_data.get("model_type") != "llava" or config_data.get("vocab_size") != 32000
            or config_data.get("mm_vision_tower") != "openai/clip-vit-large-patch14-336"):
        raise ValueError("Unexpected merged reference architecture")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    sys.path.insert(0, str(repo))
    import torch
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig, CLIPImageProcessor
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.llava_llama import LlavaConfig
    from llava.utils import disable_torch_init
    import llava
    if not guarded(llava.__file__).is_relative_to(repo):
        raise ValueError("Conflicting installed LLaVA package")
    if not torch.cuda.is_available():
        raise ValueError("Reference requires the GPU; no CPU fallback")
    admission = None
    if paired:
        admission = precision_admission(torch, root)
        if admission["gpu_identity"] != job["gpu_identity"]:
            raise ValueError("Precision worker is on a different GPU")
    torch.manual_seed(42)
    disable_torch_init()
    config = LlavaConfig.from_pretrained(str(model_dir), local_files_only=True)
    config.mm_vision_tower = str(vision_dir)
    config.unfreeze_mm_vision_tower = True
    precision_options = {}
    if precision == "nf4":
        precision_options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["vision_tower", "mm_projector", "lm_head"])
    # load_4bit=True would REPLACE this config and drop the protected modules.
    # False here is not FP16 loading: quantization_config explicitly enables NF4.
    print(json.dumps({"event": "reference_stage", "stage": "native_builder_load"}), flush=True)
    tokenizer, model, processor, context = load_pretrained_model(str(model_dir), None, "llava-v1.5-13b",
        load_4bit=False, load_8bit=False, device_map="cuda:0" if paired else "auto", device="cuda", config=config,
        local_files_only=True, **precision_options)
    model.eval().requires_grad_(False)
    if bool(getattr(model, "is_loaded_in_4bit", False)) != (precision == "nf4") or len(tokenizer) != 32000 or tokenizer.is_fast:
        raise ValueError("Reference precision/tokenizer changed")
    if getattr(model, "is_loaded_in_8bit", False):
        raise ValueError("Unexpected 8-bit model in precision control")
    device_map = str(getattr(model, "hf_device_map", model.device))
    if "cpu" in device_map.casefold() or "disk" in device_map.casefold():
        raise ValueError("Reference must be GPU-only, without offload")
    quantized = {n: m for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)}
    if len(quantized) != (280 if precision == "nf4" else 0) or any(any(p in n for p in ("vision_tower", "mm_projector", "lm_head")) for n in quantized):
        raise ValueError("Reference quantized module coverage changed")
    if any(m.weight.quant_state.quant_type != "nf4" for m in quantized.values()):
        raise ValueError("Reference is not NF4")
    # Upstream returns the tower processor; reject semantic preprocessing drift.
    baseline_processor = CLIPImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    keys = ("do_resize", "size", "resample", "do_center_crop", "crop_size", "do_rescale",
        "rescale_factor", "do_normalize", "image_mean", "image_std", "do_convert_rgb")
    semantics = {k: getattr(processor, k, None) for k in keys}
    if semantics != {k: getattr(baseline_processor, k, None) for k in keys}:
        raise ValueError("Upstream and baseline processors differ; not a matched control")
    print(json.dumps({"event": "reference_stage", "stage": "verify_loaded_tensors"}), flush=True)
    if precision == "fp16" and any(p.is_floating_point() and p.dtype != torch.float16 for p in model.parameters()):
        raise ValueError("FP16 arm contains non-FP16 floating parameters")
    checks = verify_tensors(model, model_dir, torch, **({"all_weights": True} if precision == "fp16" else {}))
    provenance = {"checkpoint_sha256": actual_hashes, "code_revision": revision, "quantization": "nf4" if precision == "nf4" else "none",
        "precision": precision, "memory_admission": admission,
        "quantized_linear_modules": len(quantized), "protected_modules": ["vision_tower", "mm_projector", "lm_head"],
        "device_map": device_map, "processor_semantics": semantics, "tensor_checks": checks,
        "source_sha256": {n: file_hash(repo / n) for n in
            ("llava/model/builder.py", "llava/eval/run_llava.py", "llava/model/language_model/llava_llama.py")},
        "compatibility_overrides": {"unfreeze_mm_vision_tower": True, "mm_vision_tower": str(vision_dir),
            "reason": "Restore merged visual weights before native builder can reload base CLIP; match NF4 exclusions"}}
    return SimpleNamespace(torch=torch, model=model, tokenizer=tokenizer, processor=processor,
        constants=importlib.import_module("llava.constants"),
        conversation=importlib.import_module("llava.conversation"),
        mm_utils=importlib.import_module("llava.mm_utils"), provenance=provenance)


def prepare(runtime, image_path, question):
    from PIL import Image
    c = runtime.constants
    image_token = c.DEFAULT_IMAGE_TOKEN
    if getattr(runtime.model.config, "mm_use_im_start_end", False):
        image_token = c.DEFAULT_IM_START_TOKEN + image_token + c.DEFAULT_IM_END_TOKEN
    conversation = runtime.conversation.conv_templates["llava_v1"].copy()
    conversation.append_message(conversation.roles[0], image_token + "\n" + question)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    pixels = runtime.mm_utils.process_images([image], runtime.processor, runtime.model.config)
    if not runtime.torch.is_tensor(pixels):
        raise ValueError("Reference expects one square image tensor")
    pixels = pixels.to(runtime.model.device, dtype=runtime.torch.float16)
    ids = runtime.mm_utils.tokenizer_image_token(prompt, runtime.tokenizer, c.IMAGE_TOKEN_INDEX,
        return_tensors="pt").unsqueeze(0).to(runtime.model.device)
    if int((ids == c.IMAGE_TOKEN_INDEX).sum().item()) != 1:
        raise ValueError("Reference requires one image token")
    metadata = {"rendered_prompt": prompt, "input_token_ids": ids[0].detach().cpu().tolist(),
        "pixels": array_summary(pixels.detach().float().cpu().numpy())}
    return ids, pixels, image.size, metadata


def neutral_probes(runtime, sources):
    rows = []
    for source in sources:
        ids, pixels, size, row = prepare(runtime, source["image_path"], PROBE)
        captures = {}; handles = []; counts = {}
        def capture(name):
            def hook(module, args, output):
                counts[name] = counts.get(name, 0) + 1
                captures[name] = array_summary(output.detach().float().cpu().numpy())
            return hook
        try:
            handles.append(runtime.model.get_vision_tower().register_forward_hook(capture("vision")))
            handles.append(runtime.model.get_model().mm_projector.register_forward_hook(capture("projector")))
            with runtime.torch.inference_mode():
                result = runtime.model(input_ids=ids, images=pixels, image_sizes=[size], use_cache=False, return_dict=True)
                row["next_token_logits"] = array_summary(result.logits[0, -1, :].detach().float().cpu().numpy())
            if counts != {"vision": 1, "projector": 1}:
                raise ValueError("Reference neutral forward must use both visual modules exactly once")
            row.update(captures, sample_id=source["sample_id"])
            rows.append(row)
        finally:
            for handle in handles: handle.remove()
    return rows


def predict(runtime, request, expected):
    ids, pixels, size, trace = prepare(runtime, request["image_paths"][0], request["prompt"])
    if (trace["input_token_ids"] != expected["input_token_ids"] or trace["rendered_prompt"] != expected["rendered_prompt"]
            or trace["pixels"]["sha256_float32"] != expected["pixels_sha256_float32"]):
        raise ValueError("Reference tokenized/pixel inputs differ from baseline; stopping unmatched control")
    cache = {"calls": 0, "initial_past_key_values": None, "effective_modes": []}
    def observe(module, args, kwargs):
        if cache["calls"] == 0:
            cache["initial_past_key_values"] = "not_passed" if "past_key_values" not in kwargs else (
                "none" if kwargs["past_key_values"] is None else "provided")
        cache["calls"] += 1
        if kwargs.get("use_cache") not in cache["effective_modes"]:
            cache["effective_modes"].append(kwargs.get("use_cache"))
    handle = runtime.model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        with runtime.torch.inference_mode():
            generated = runtime.model.generate(ids, images=pixels, image_sizes=[size], **GENERATION)
    finally:
        handle.remove()
    if not cache["calls"] or cache["initial_past_key_values"] not in {"none", "not_passed"}:
        raise ValueError("Reference generation did not start with fresh cache")
    sequence = generated.sequences[0].detach().cpu().tolist()
    steps = len(generated.scores)
    tokens = sequence[len(sequence) - steps:] if steps else []
    text = runtime.tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
    trace.update(raw_response=text, returned_ids=sequence, generated_token_ids=tokens, generated_steps=steps,
        ended_with_eos=bool(tokens and tokens[-1] == runtime.tokenizer.eos_token_id),
        hit_token_budget=steps >= 96 and not bool(tokens and tokens[-1] == runtime.tokenizer.eos_token_id),
        generation_arguments=GENERATION, generation_config=runtime.model.generation_config.to_dict(), cache_trace=cache,
        decoded_with_special_tokens=runtime.tokenizer.batch_decode(generated.sequences, skip_special_tokens=False)[0])
    if generated.scores:
        import numpy as np
        first = generated.scores[0][0].detach().float().cpu().numpy()
        trace["first_token_top5"] = [{"token_id": int(i), "token": runtime.tokenizer.decode([int(i)]),
            "score": float(first[i])} for i in np.argsort(first)[-5:][::-1]]
    return text, trace


def execute(job_path):
    job = json.loads(guarded(job_path).read_text(encoding="utf-8"))
    paired = job.get("schema") == PRECISION_SCHEMA
    fields = {"schema", "workspace", "output_dir", "requests", "input_expectations", "sources", "checkpoint_sha256", "worker_sha256"}
    if paired: fields |= {"precision", "gpu_identity"}
    if set(job) != fields or (paired and job["precision"] not in {"fp16", "nf4"}):
        raise ValueError("Unexpected reference job fields, including possible outcomes")
    if job["schema"] not in {SCHEMA, PRECISION_SCHEMA} or job["worker_sha256"] != file_hash(Path(__file__)):
        raise ValueError("Reference job/schema/code changed")
    root, folder = guarded(job["workspace"]), guarded(job["output_dir"])
    if not folder.is_relative_to(root / "outputs"):
        raise ValueError("Reference output is outside workspace outputs")
    requests = job["requests"]
    required = {"request_id", "image_paths", "prompt", "allowed_answers", "max_new_tokens", "do_sample", "modality_paths"}
    if len(requests) != 16 or len({r["request_id"] for r in requests}) != 16 or set(job["input_expectations"]) != {r["request_id"] for r in requests}:
        raise ValueError("Reference requires sixteen distinct paired requests")
    for request in requests:
        if (set(request) != required or request["do_sample"] is not False or request["max_new_tokens"] != 96
                or request["allowed_answers"] or request["modality_paths"] or len(request["image_paths"]) != 1):
            raise ValueError("Reference input contract changed")
        if not guarded(request["image_paths"][0]).is_relative_to(root / "data_orbital"):
            raise ValueError("Reference image is outside data_orbital")
        if set(job["input_expectations"][request["request_id"]]) != {"input_token_ids", "rendered_prompt", "pixels_sha256_float32"}:
            raise ValueError("Reference input sidecar has unexpected/outcome fields")
    if (len(job["sources"]) != 2 or len({s["sample_id"] for s in job["sources"]}) != 2
            or any(set(s) != {"sample_id", "image_path"} for s in job["sources"])
            or {str(guarded(s["image_path"])) for s in job["sources"]} != {str(guarded(r["image_paths"][0])) for r in requests}):
        raise ValueError("Reference sources differ from the model-visible images")
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder / n).exists() for n in ("predictions.jsonl", "generation_diagnostics.json", "loading.json")):
        raise ValueError("Reference output exists; never resume or overwrite")
    with (folder / "worker_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "parent_pid": os.getppid(), "worker_sha256": job["worker_sha256"],
            "schema": job["schema"], "precision": job.get("precision", "nf4"),
            "adapter_used": False, "project_runner_used": False}, stream)
    runtime = None; traces = {}
    try:
        start = perf_counter()
        runtime = load_runtime(job, folder)
        write_json(folder / "loading.json", {"elapsed_seconds": perf_counter() - start, "provenance": runtime.provenance})
        write_json(folder / "neutral_probes.json", neutral_probes(runtime, job["sources"]))
        for index, request in enumerate(requests, 1):
            start = perf_counter()
            try:
                text, trace = predict(runtime, request, job["input_expectations"][request["request_id"]])
            except Exception as error:
                with (folder / "predictions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"request_id": request["request_id"], "raw_response": "", "status": "error",
                        "error_message": f"{type(error).__name__}: {error}"}) + "\n")
                    stream.flush(); os.fsync(stream.fileno())
                raise
            traces[request["request_id"]] = trace
            write_json(folder / "generation_diagnostics.json", traces)
            with (folder / "predictions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"request_id": request["request_id"], "raw_response": text, "status": "ok",
                    "elapsed_seconds": perf_counter() - start, "mock": False}) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            print(json.dumps({"event": "reference_response_saved", "completed": index, "total": len(requests)}), flush=True)
    except BaseException as error:
        write_json(folder / "reference_error.json", {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        raise
    finally:
        if runtime is not None:
            runtime.model = None
            gc.collect()
            if runtime.torch.cuda.is_available(): runtime.torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    execute(parser.parse_args().job)
