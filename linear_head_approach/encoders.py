"""Load the actual frozen vision towers, without any language model or task head."""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .common import ROOT, atomic_json, code_hash, environment, fingerprint, identity_guard, logger, read_json, rows, safe_asset, sha, write_rows, closes_stage_logs
from .data import DATA, INPUT_KEYS, validate_inputs

IBM_REV = "b657bdfe6fc4c0445d918f77a2d7b919805f3bce"
SPACE_REV = "5a69f71bda9dea3e4d8113c1047c07fd3da483f8"
CLIP_REV = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
RUNTIME_REV = "d54c67aad513cb9daca444afa425cfb278e4fbf8"
SPACE_PREFIX = "model.vision_tower.vision_tower."
ASSETS = {
    "ibm": ("nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model", IBM_REV,
            ["backbone/checkpoint.pt", "backbone/config.yaml"]),
    "spacellava": ("remyxai/SpaceLLaVA", SPACE_REV,
                   ["config.json", "preprocessor_config.json", "model.safetensors.index.json", "model-00006-of-00006.safetensors"]),
    "clip_config": ("openai/clip-vit-large-patch14-336", CLIP_REV, ["config.json"]),
}
IBM_HASHES = {"backbone/checkpoint.pt": "5843824dfa56f89552b37b5295aa865ed28777c5f2b33286db2ef81b4cf08e5b",
              "backbone/config.yaml": "fd94d20eb2b1cd4379516da3d2c1912a56a8684e1952ee9331abd63e7c59e619"}


def clip_key_map(checkpoint_keys, model_keys):
    """Map the pinned tower to either HF layout, with an exact key bijection.

    Transformers 4 wraps CLIPVisionTransformer in ``vision_model``; 5 exposes
    the same vision parameters directly. No tensor is skipped, added or guessed.
    Tensor shapes remain the responsibility of strict load_state_dict.
    """
    checkpoint_keys, model_keys = list(checkpoint_keys), list(model_keys)
    if (not checkpoint_keys or len(set(checkpoint_keys)) != len(checkpoint_keys)
            or len(set(model_keys)) != len(model_keys)):
        raise ValueError("Empty or duplicate CLIP state-dict keys")
    prefix = SPACE_PREFIX + "vision_model."
    if any(not key.startswith(prefix) for key in checkpoint_keys):
        raise ValueError("Expected only pinned SpaceLLaVA vision_model tensor keys")
    expected = set(model_keys)
    for layout, remove in (("wrapped", SPACE_PREFIX), ("flat", prefix)):
        mapped = {key.removeprefix(remove): key for key in checkpoint_keys}
        if len(mapped) == len(checkpoint_keys) and set(mapped) == expected:
            return mapped, layout
    raise ValueError(
        "CLIP state-dict namespace mismatch: neither wrapped nor flat layout "
        f"exactly matches all model keys (checkpoint={len(checkpoint_keys)}, model={len(model_keys)}). "
        "Partial/randomly initialized vision weights are forbidden."
    )


def assets(directory, encoder, allow_download=False):
    """Explicit pinned download on the GPU host only. No full SpaceLLaVA weights."""
    directory = Path(directory)
    if "data_rover" in {p.casefold() for p in directory.resolve().parts}:
        raise ValueError("Protected archive cannot contain weights")
    names = [encoder] + (["clip_config"] if encoder == "spacellava" else [])
    result = {}
    for name in names:
        repo, revision, files = ASSETS[name]
        folder = directory / name
        stamp = folder / "assets.json"
        spec = {"repo": repo, "revision": revision, "files": files}
        if stamp.exists():
            saved = read_json(stamp)
            if saved["source"] != spec or any(not (folder / f).is_file() or sha(folder / f) != saved["hashes"][f] for f in files):
                raise ValueError(f"Asset provenance/hash mismatch: {folder}")
            if name == "ibm" and saved["hashes"] != IBM_HASHES:
                raise ValueError("IBM asset stamp differs from the audited checkpoint hashes")
        else:
            if not allow_download:
                raise FileNotFoundError(f"Pinned assets absent: {folder}. Use assets --allow-download explicitly, or restore a verified assets folder.")
            import shutil
            from huggingface_hub import HfApi
            from huggingface_hub import hf_hub_download
            folder.mkdir(parents=True, exist_ok=True)
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            sizes = {item.rfilename: item.size for item in info.siblings}
            if any(sizes.get(file) is None for file in files):
                raise ValueError("Remote file sizes unavailable; download preflight cannot safely estimate disk usage")
            needed = sum(sizes[file] for file in files if not (folder / file).exists())
            free = shutil.disk_usage(folder).free
            if free < needed + 2 * 2**30:
                raise OSError(f"Insufficient disk space: need download {needed/2**30:.2f} GiB plus 2 GiB reserve; free {free/2**30:.2f} GiB")
            print(f"Pinned {name}: download <= {needed/2**30:.2f} GiB; disk free {free/2**30:.2f} GiB", flush=True)
            for file in files:
                hf_hub_download(repo, filename=file, revision=revision, local_dir=str(folder))
            saved = {"source": spec, "hashes": {f: sha(folder / f) for f in files}}
            if name == "ibm" and saved["hashes"] != IBM_HASHES:
                raise ValueError("IBM checkpoint/config differ from audited SHA-256")
            atomic_json(stamp, saved)
        result[name] = saved
    return result


def load(directory, encoder, root=ROOT, device="cuda"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory, root = Path(directory), Path(root)
    verified = assets(directory, encoder)
    if encoder == "spacellava":
        from safetensors import safe_open
        from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModel
        folder = directory / encoder
        cfg = read_json(folder / "config.json")
        if cfg["mm_vision_tower"] != "openai/clip-vit-large-patch14-336" or cfg["mm_vision_select_layer"] != -2:
            raise ValueError("Unrecognized SpaceLLaVA vision configuration")
        index = read_json(folder / "model.safetensors.index.json")["weight_map"]
        keys = [key for key in index if key.startswith(SPACE_PREFIX)]
        if len(keys) != 391 or {index[k] for k in keys} != {"model-00006-of-00006.safetensors"}:
            raise ValueError("Actual SpaceLLaVA tower shard/key count changed")
        cfg = CLIPConfig.from_pretrained(str(directory / "clip_config"), local_files_only=True)
        model = CLIPVisionModel(cfg.vision_config)
        key_map, layout = clip_key_map(keys, model.state_dict().keys())
        with safe_open(str(folder / "model-00006-of-00006.safetensors"), framework="pt", device="cpu") as handle:
            model.load_state_dict({target: handle.get_tensor(source) for target, source in key_map.items()}, strict=True)
        print(f"CLIP vision weights loaded: tensors={len(key_map)} namespace={layout} strict=True", flush=True)
        processor = CLIPImageProcessor.from_pretrained(str(folder), local_files_only=True)

        def forward(row):
            with Image.open(safe_asset(root, row["rgb_image"], row["rgb_image_sha256"])) as image:
                if image.size != (512, 512):
                    raise ValueError("Non-square CLIP input would require mask crop alignment")
                pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
            hidden = model(pixels, output_hidden_states=True).hidden_states[-2]
            if tuple(hidden.shape) != (1, 577, 1024):
                raise ValueError(f"Unexpected CLIP feature shape: {hidden.shape}")
            return hidden[:, 1:].reshape(1, 24, 24, 1024).permute(0, 3, 1, 2)
    else:
        import rasterio
        source = root / "external/ibm-lfm"
        bundle = root / "bundle.json"
        if bundle.exists():
            metadata = read_json(bundle)
            if metadata["ibm_runtime_revision"] != RUNTIME_REV:
                raise ValueError("Wrong bundled IBM runtime")
            expected = {name: digest for name, digest in metadata["files"].items()
                        if name.startswith("external/ibm-lfm/") and name.endswith(".py")}
            actual = {p.relative_to(root).as_posix(): sha(p) for folder in ("ni_lfm", "terratorch_integration")
                      for p in (source / folder).rglob("*.py")}
            if actual != expected:
                raise ValueError("Bundled IBM runtime source changed")
        else:
            from .common import git_revision
            if git_revision(source) != RUNTIME_REV:
                raise ValueError("Wrong IBM source revision")
            import subprocess
            if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--", "ni_lfm", "terratorch_integration"], text=True).strip():
                raise ValueError("IBM source has uncommitted edits")
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location("head_lunar_backbone", source / "terratorch_integration/lunar_backbone.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class CheckedBackbone(module.LunarBackbone):
            def load_checkpoint(self, checkpoint_path):
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                state = {k.removeprefix("module."): v for k, v in state.items()}
                incompatible = self.model.load_state_dict(state, strict=False)
                # Other pretrained modalities are expected extras. Missing active weights are NOT.
                active = set(dict(self.model.named_parameters()))
                missing = active.intersection(incompatible.missing_keys)
                if missing:
                    raise ValueError(f"IBM parameters would be randomly initialized: {sorted(missing)[:8]}")

        model = CheckedBackbone(variant="base", modalities=["vis"], merge_method="dict",
                                cfg=str(directory / "ibm/backbone/config.yaml"),
                                checkpoint_path=str(directory / "ibm/backbone/checkpoint.pt"), patch_size=8)
        means = np.array([.018106, .025262, .027113, .028616, .030757], dtype=np.float32)[:, None, None]
        stds = np.array([.010051, .013646, .014491, .015267, .016555], dtype=np.float32)[:, None, None]

        def forward(row):
            path = safe_asset(root, row["native_vis"], row["native_vis_sha256"])
            with rasterio.open(path) as source_image:
                image = source_image.read(out_dtype="float32")
            if image.shape != (5, 512, 512) or np.isinf(image).any():
                raise ValueError("Expected 5x512x512 native WAC, no infinite pixels")
            image = np.nan_to_num(image, nan=0.0)
            pixels = torch.from_numpy(((image - means) / stds)[None]).to(device)
            hidden = model({"vis": pixels})[11]["vis"]
            if tuple(hidden.shape) != (1, 4096, 768):
                raise ValueError(f"Unexpected IBM feature shape: {hidden.shape}")
            return hidden.reshape(1, 64, 64, 768).permute(0, 3, 1, 2)
    model = model.float().to(device).eval()
    model.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Encoder must be frozen")
    return model, forward, verified


@closes_stage_logs
def extract(encoder, directory, output, session, smoke=False, root=ROOT, device="cuda"):
    import torch
    import torch.nn.functional as F
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Real feature extraction requires CUDA; run CPU fixture tests separately")
    root, output = Path(root), Path(output)
    log = logger(output)
    inputs = rows(root / DATA / "inputs.jsonl")
    validate_inputs(inputs)
    if any(set(r) != INPUT_KEYS for r in inputs):
        raise ValueError("Only clean encoder inputs accepted (no targets)")
    verified = assets(directory, encoder)
    identity = {"encoder": encoder, "assets": verified, "inputs_sha256": sha(root / DATA / "inputs.jsonl"),
                "environment": environment(), "code": code_hash(), "grid": 24, "precision": "float32",
                "math_policy": "deterministic_fp32_no_tf32",
                "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda}
    identity_guard(output / "identity.json", identity)
    marker = output / "smoke.json"
    if not smoke and (not marker.exists() or read_json(marker).get("session") != session):
        raise ValueError("Run extract --smoke with this --session-id first")
    model, forward, _ = load(directory, encoder, root, device)
    versions = {n: p._version for n, p in model.named_parameters()}
    manifest = output / "features.jsonl"
    records = rows(manifest) if manifest.exists() else []
    known = {r["sample_id"]: r for r in records}
    selected = inputs[:2] if smoke else inputs
    elapsed = []
    for index, row in enumerate(selected):
        sid = row["sample_id"]
        if sid in known:
            if known[sid]["file"] != f"{sid}.npy":
                raise ValueError("Unsafe cached feature path")
            path = output / known[sid]["file"]
            if not path.is_file() or sha(path) != known[sid]["sha256"]:
                raise ValueError(f"Corrupted feature cache: {sid}")
            if not smoke:
                continue
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            feature = F.interpolate(forward(row), size=(24, 24), mode="bilinear", align_corners=False)[0].cpu().numpy()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if not np.isfinite(feature).all() or float(feature.std()) <= 1e-8:
            raise ValueError("Nonfinite/constant encoder features")
        if sid in known:
            old = np.load(output / known[sid]["file"], allow_pickle=False)
            if not np.allclose(feature, old, rtol=1e-4, atol=1e-5):
                raise ValueError("Repeated feature extraction changed; do not reuse cache")
        else:
            path = output / f"{sid}.npy"
            with path.with_suffix(".tmp").open("wb") as stream:
                np.save(stream, feature.astype(np.float32), allow_pickle=False)
            os.replace(path.with_suffix(".tmp"), path)
            record = {"sample_id": sid, "split": row["split"], "scene_group_id": row["scene_group_id"],
                      "file": path.name, "sha256": sha(path), "shape": list(feature.shape)}
            records.append(record)
            known[sid] = record
            write_rows(manifest, records)
        elapsed.append(seconds)
        log.info("feature encoder=%s sample=%s done=%s/%s seconds=%.2f peak_vram_gb=%.2f measured_remaining_minutes=%.1f",
                 encoder, sid, index + 1, len(selected), seconds, torch.cuda.max_memory_allocated() / 2**30,
                 np.mean(elapsed) * (len(selected) - index - 1) / 60)
    if versions != {n: p._version for n, p in model.named_parameters()} or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Encoder weights changed / acquired gradients")
    if smoke:
        a, b = [np.load(output / known[r["sample_id"]]["file"], allow_pickle=False) for r in selected]
        if np.array_equal(a, b):
            raise ValueError("Two different images produced identical features")
        atomic_json(marker, {"session": session, "identity": fingerprint(identity), "two_images_differ": True,
                             "mean_seconds_per_image": float(np.mean(elapsed)), "scientific_result": False})
    return {"images_cached": len(records), "smoke": smoke, "encoder": encoder}
