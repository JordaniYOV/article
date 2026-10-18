"""Read-only checks on one loaded adapter; no downloads, labels or training.

These checks diagnose visual conditioning, not crater accuracy. Torch and
safetensors are imported only in the isolated model process.
"""

from hashlib import sha256
import importlib
import json
from pathlib import Path

from planetary_vlm.contracts import ModelRequest
from .adapter import SpaceLLaVAAdapter
from ..common.local_runtime import offline_imports

PROTOCOL = "spacellava-visual-diagnostic-v1"
WEIGHT_PREFIXES = ("model.vision_tower.", "model.mm_projector.")
PROBE_PROMPT = "Describe the visible contents of this image."


def array_summary(value):
    import numpy as np
    value = np.asarray(value, dtype=np.float32)
    finite = bool(np.isfinite(value).all())
    return {"shape": list(value.shape), "finite": finite,
        "sha256_float32": sha256(value.tobytes(order="C")).hexdigest(),
        "min": float(value.min()) if finite and value.size else None,
        "max": float(value.max()) if finite and value.size else None,
        "std": float(value.std(dtype=np.float64)) if finite and value.size else None}


def compare_arrays(left, right):
    import numpy as np
    left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    same_shape = left.shape == right.shape
    if not finite or not same_shape or not left.size:
        return {"finite": finite, "same_shape": same_shape, "different": False,
                "max_absolute_difference": None, "rms_difference": None}
    difference = left.astype(np.float64) - right.astype(np.float64)
    return {"finite": True, "same_shape": True, "different": bool(np.any(difference != 0)),
        "max_absolute_difference": float(np.abs(difference).max()),
        "rms_difference": float(np.sqrt(np.mean(difference * difference)))}


def inspect_weights(adapter):
    """Compare ALL visual/projector tensors against the merged local checkpoint."""
    from safetensors import safe_open
    torch, model = adapter.torch, adapter.model
    meta = [name for name, value in (*model.named_parameters(), *model.named_buffers()) if value.is_meta]
    state = {key: value for key, value in model.state_dict().items() if key.startswith(WEIGHT_PREFIXES)}
    folder = Path(adapter.options["model_dir"]).resolve()
    index = json.loads((folder / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    checkpoint_keys = {key for key in index if key.startswith(WEIGHT_PREFIXES)}
    missing = sorted(set(state) - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - set(state))
    problems, checked = [], 0
    shards = sorted({index[key] for key in set(state) & checkpoint_keys})
    for shard in shards:
        path = (folder / shard).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Unsafe visual checkpoint shard path")
        with safe_open(str(path), framework="pt", device="cpu") as archive:
            for key in sorted(key for key in state if index.get(key) == shard):
                actual = state[key]
                if actual.is_meta:
                    problems.append({"key": key, "reason": "meta_parameter"})
                    continue
                if actual.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                    problems.append({"key": key, "reason": "unexpected_quantized_visual_parameter"})
                    continue
                expected = archive.get_tensor(key).to(dtype=actual.dtype)
                actual_cpu = actual.detach().cpu()
                finite = bool(torch.isfinite(actual_cpu).all().item())
                same = actual_cpu.shape == expected.shape and torch.equal(actual_cpu, expected)
                checked += 1
                if not finite or not same:
                    problems.append({"key": key, "reason": "nonfinite" if not finite else "checkpoint_value_mismatch"})
    return {"success": bool(state) and checked == len(state) and not (meta or missing or unexpected or problems),
        "scope": "Every visual tower/projector state tensor; language-model weight values not compared",
        "comparison": "exact equality after casting checkpoint to the loaded floating dtype",
        "visual_tensors": len(state), "compared_tensors": checked, "meta_parameters_or_buffers": meta,
        "missing_checkpoint_keys": missing, "extra_checkpoint_keys": unexpected, "problems": problems}


def trace_image(adapter, image_path):
    """Trace actual LM forward hooks so merely calling CLIP separately cannot pass."""
    request = ModelRequest("visual-diagnostic", (str(image_path),), PROBE_PROMPT, (), max_new_tokens=16)
    torch = adapter.torch
    captures, hooks = {}, []
    def capture(name):
        def hook(module, args, output):
            captures.setdefault(name, []).append(output.detach().float().cpu().numpy())
        return hook
    with offline_imports(Path(adapter.options["runtime_repo"]).resolve()), torch.inference_mode():
        constants = importlib.import_module("llava.constants")
        ids, pixels, size = adapter.prepare_inputs(request)
        token_count = int((ids == constants.IMAGE_TOKEN_INDEX).sum().item())
        if token_count != 1:
            raise ValueError(f"Expected exactly one image token, found {token_count}")
        if isinstance(pixels, list):
            raise ValueError("This fixed two-image diagnostic requires one square pixel tensor per image")
        pixels_cpu = pixels.detach().float().cpu().numpy()
        ids_cpu = ids.detach().cpu().numpy()
        try:
            hooks.append(adapter.model.get_vision_tower().register_forward_hook(capture("vision")))
            hooks.append(adapter.model.get_model().mm_projector.register_forward_hook(capture("projector")))
            result = adapter.model(input_ids=ids, images=pixels, image_sizes=[size],
                                   use_cache=False, return_dict=True)
            logits = result.logits[0, -1, :].detach().float().cpu().numpy()
            del result
        finally:
            for handle in hooks:
                handle.remove()
    if len(captures.get("vision", [])) != 1 or len(captures.get("projector", [])) != 1:
        raise ValueError("LM forward did not execute exactly one visual encoder and projector")
    return {"pixels": pixels_cpu, "input_ids": ids_cpu, "vision": captures["vision"][0],
            "projector": captures["projector"][0], "next_token_logits": logits, "image_tokens": token_count}


def diagnose(adapter, image_paths):
    import numpy as np
    report = {"protocol": PROTOCOL, "success": False, "scientific_result": False,
        "probe_prompt": PROBE_PROMPT, "images": list(image_paths),
        "limitation": "Different logits/features establish a live visual path, not correct crater reasoning. "
                      "Identical decoded answers alone do not prove identical logits."}
    try:
        if not isinstance(adapter, SpaceLLaVAAdapter) or adapter.model is None or len(image_paths) != 2:
            raise ValueError("Diagnostics require the loaded real SpaceLLaVA adapter and two images")
        report["weights"] = inspect_weights(adapter)
        if not report["weights"]["success"]:
            return report
        traces = [trace_image(adapter, path) for path in image_paths]
        report["traces"] = [{"image_tokens": row["image_tokens"],
            **{key: array_summary(row[key]) for key in ("pixels", "vision", "projector", "next_token_logits")}}
            for row in traces]
        comparisons = {key: compare_arrays(traces[0][key], traces[1][key])
                       for key in ("pixels", "vision", "projector", "next_token_logits")}
        report["same_text_input_ids"] = bool(np.array_equal(traces[0]["input_ids"], traces[1]["input_ids"]))
        report["comparisons"] = comparisons
        report["success"] = report["same_text_input_ids"] and all(
            row["finite"] and row["same_shape"] and row["different"] for row in comparisons.values())
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report
