"""Extract frozen spatial tokens for the selected WAC crater probe.

Requires local model files and an explicit model revision. Never downloads
weights or reads target manifests. Run one encoder at a time, first with
--limit 1 on the GPU host. This adapter remains unverified until that smoke test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPACE_PREFIX = "model.vision_tower.vision_tower."
IBM_MEANS = np.array([0.018106, 0.025262, 0.027113, 0.028616, 0.030757], dtype=np.float32)
IBM_STDS = np.array([0.010051, 0.013646, 0.014491, 0.015267, 0.016555], dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_input(relative: str, expected: str) -> Path:
    path = (ROOT / relative).resolve()
    if (ROOT / "data_orbital").resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Unsafe or missing orbital input: {relative}")
    if sha256(path) != expected:
        raise ValueError(f"Input SHA-256 mismatch: {relative}")
    return path


def load_spacellava(model_dir: Path, base_clip_dir: Path, device: str):
    """Load only the actual SpaceLLaVA visual-tower tensors from one shard."""
    import torch
    from safetensors import safe_open
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModel

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if (config.get("mm_vision_tower") != "openai/clip-vit-large-patch14-336"
            or config.get("mm_vision_select_layer") != -2):
        raise ValueError("SpaceLLaVA tower or selected layer differs from audited config")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mapping = {key: shard for key, shard in index["weight_map"].items()
               if key.startswith(SPACE_PREFIX)}
    if not mapping or len(set(mapping.values())) != 1:
        raise ValueError("SpaceLLaVA visual tower must occupy one indexed shard")
    shard = model_dir / next(iter(mapping.values()))
    if not shard.is_file():
        raise FileNotFoundError(shard)
    clip_config = CLIPConfig.from_pretrained(str(base_clip_dir), local_files_only=True)
    processor = CLIPImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = CLIPVisionModel(clip_config.vision_config)
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        state = {key.removeprefix(SPACE_PREFIX): handle.get_tensor(key) for key in mapping}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    preprocess_hash = hashlib.sha256("".join((
        sha256(model_dir / "config.json"),
        sha256(model_dir / "preprocessor_config.json"),
        sha256(base_clip_dir / "config.json"),
    )).encode("ascii")).hexdigest()
    checkpoint_hash = sha256(shard)

    def extract(row: dict) -> np.ndarray:
        path = checked_input(row["rgb_image"], row["rgb_image_sha256"])
        with Image.open(path) as source:
            image = source.convert("RGB")
            pixels = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            hidden = model(pixels, output_hidden_states=True).hidden_states[-2]
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != 577:
            raise ValueError(f"Unexpected SpaceLLaVA visual tokens: {tuple(hidden.shape)}")
        return hidden[0, 1:, :].reshape(24, 24, -1).float().cpu().numpy()

    return extract, checkpoint_hash, f"spacellava-clip336-processor-{preprocess_hash}"


def load_ibm(model_dir: Path, source_dir: Path, device: str):
    """Use the publisher's LunarBackbone and WAC data-adapter normalization."""
    import rasterio
    import torch

    source_file = source_dir / "terratorch_integration" / "lunar_backbone.py"
    if not source_file.is_file():
        raise FileNotFoundError(source_file)
    sys.path.insert(0, str(source_dir))
    spec = importlib.util.spec_from_file_location("probe_lunar_backbone", source_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load publisher LunarBackbone")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = model_dir / "backbone" / "checkpoint.pt"
    configuration = model_dir / "backbone" / "config.yaml"
    model = module.LunarBackbone(
        variant="base", modalities=["vis"], merge_method="dict",
        cfg=str(configuration), checkpoint_path=str(checkpoint), patch_size=8,
    ).to(device).eval()
    checkpoint_hash = sha256(checkpoint)
    preprocess_hash = sha256(configuration)

    def extract(row: dict) -> np.ndarray:
        relative = row.get("native_vis_5band_path")
        if not relative:
            raise ValueError(f"Native five-band WAC input missing: {row['sample_id']}")
        path = checked_input(relative, row["native_vis_5band_source_sha256"])
        with rasterio.open(path) as dataset:
            image = dataset.read(out_dtype="float32")
        if image.ndim != 3 or image.shape[0] != 5:
            raise ValueError(f"Expected five WAC bands: {path}")
        if np.isinf(image).any():
            raise ValueError(f"Infinite values in WAC raster: {path}")
        image = np.nan_to_num(image, nan=0.0)
        resized = np.empty((5, 512, 512), dtype=np.float32)
        for band in range(5):
            resized[band] = np.asarray(Image.fromarray(image[band], mode="F").resize(
                (512, 512), Image.Resampling.BILINEAR), dtype=np.float32)
        tensor = torch.from_numpy(((resized - IBM_MEANS[:, None, None]) /
                                   IBM_STDS[:, None, None])[None]).to(device)
        with torch.inference_mode():
            hidden = model({"vis": tensor})[11]["vis"]
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != 4096:
            raise ValueError(f"Unexpected IBM WAC visual tokens: {tuple(hidden.shape)}")
        return hidden[0].reshape(64, 64, -1).float().cpu().numpy()

    return extract, checkpoint_hash, f"ibm-wac-512-ps8-official-normalization-{preprocess_hash}"


def run(args: argparse.Namespace) -> dict:
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    inputs = [json.loads(line) for line in args.input_manifest.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if len(inputs) != 120 or len({row["sample_id"] for row in inputs}) != 120:
        raise ValueError("Expected 120 unique WAC probe inputs")
    if args.limit is not None and not 1 <= args.limit <= 120:
        raise ValueError("--limit must be in 1..120")
    args.output_dir = args.output_dir.resolve()
    if (ROOT / "linear_head_approach" / "outputs").resolve() not in args.output_dir.parents:
        raise ValueError("Features must be written beneath linear_head_approach/outputs/")
    if args.encoder == "spacellava":
        if args.base_clip_dir is None:
            raise ValueError("--base-clip-dir is required for the visual architecture config")
        extract, checkpoint_hash, preprocessing_id = load_spacellava(
            args.model_dir, args.base_clip_dir, args.device)
    else:
        if args.ibm_source_dir is None:
            raise ValueError("--ibm-source-dir is required")
        extract, checkpoint_hash, preprocessing_id = load_ibm(
            args.model_dir, args.ibm_source_dir, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in (inputs if args.limit is None else inputs[:args.limit]):
        features = extract(row)
        if features.ndim != 3 or not np.isfinite(features).all():
            raise ValueError(f"Invalid features for {row['sample_id']}")
        path = args.output_dir / f"{row['sample_id']}.npy"
        np.save(path, features.astype(np.float32))
        records.append({
            "sample_id": row["sample_id"], "split": row["split"],
            "scene_group_id": row["scene_group_id"],
            "rgb_image_sha256": row["rgb_image_sha256"],
            "feature": path.relative_to(ROOT).as_posix(), "feature_sha256": sha256(path),
            "feature_shape": list(features.shape), "extractor_id": args.encoder,
            "model_revision": args.model_revision,
            "checkpoint_sha256": checkpoint_hash, "preprocessing_id": preprocessing_id,
        })
        if len(records) % 10 == 0 or args.limit == 1:
            print(f"Extracted {len(records)}/{args.limit or len(inputs)} {args.encoder} images", flush=True)
    (args.output_dir / "features.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return {"encoder": args.encoder, "images": len(records),
            "checkpoint_sha256": checkpoint_hash, "preprocessing_id": preprocessing_id,
            "feature_manifest": str(args.output_dir / "features.jsonl")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("spacellava", "ibm"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-clip-dir", type=Path)
    parser.add_argument("--ibm-source-dir", type=Path)
    parser.add_argument("--input-manifest", type=Path, default=ROOT / "linear_head_approach" / "outputs" / "encoder_probe_wac_v1" / "encoder_inputs.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
