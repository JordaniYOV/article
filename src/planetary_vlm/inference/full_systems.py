"""Orbital image preparation and sequential complete-system inference.

Heavy packages are imported only when images are transformed or a model runs.
No acquisition, training or ground-truth access takes place here.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tomllib

from planetary_vlm.contracts import ModelRequest
from planetary_vlm.inference.config import load_config
from planetary_vlm.inference.runner import file_hash, run
from planetary_vlm.io import read_jsonl, write_json, write_jsonl

CONDITIONS = {"clean", "shear", "radiation", "lens_flare", "hard_shadows", "combined"}
MODEL_BACKENDS = {"SpaceLLaVA": "spacellava", "NASA-IBM-Lunar-Foundation-Model": "ibm_imp"}


def load_pipeline(path):
    path = Path(path).resolve()
    config = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    if set(config) != {"dataset", "generation", "distortion", "models"}:
        raise ValueError("Expected dataset/generation/distortion/models pipeline tables")
    root = (path.parent / config["dataset"]["workspace_root"]).resolve()
    def resolve(value):
        result = (root / value).resolve()
        if not result.is_relative_to(root) or "data_rover" in {part.lower() for part in result.parts}:
            raise ValueError(f"Unsafe pipeline path: {value}")
        return result
    dataset = config["dataset"]
    for key in ("manifest", "prepared_dir"):
        dataset[key] = resolve(dataset[key])
        if not dataset[key].is_relative_to(root / "data_orbital"):
            raise ValueError("Dataset preparation is restricted to data_orbital")
    models = {}
    if set(config["models"]) != set(MODEL_BACKENDS):
        raise ValueError("Pipeline must declare the two selected complete systems")
    for name, value in config["models"].items():
        item = load_config(resolve(value))
        if item.backend != MODEL_BACKENDS[name] or item.output_dir != root / "outputs" / name:
            raise ValueError("Model backend and outputs/<model name> must match")
        models[name] = item
    conditions = dataset["conditions"]
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - CONDITIONS:
        raise ValueError("Unknown, duplicate or empty conditions")
    if not dataset["splits"] or set(dataset["splits"]) - {"val", "test"}:
        raise ValueError("Only publisher val/test inputs are admitted")
    if type(config["generation"]["seed"]) is not int or config["generation"]["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    return config, root, models


def transform(image, condition, seed, parameters):
    from planetary_vlm.distortion import (apply_synchronous_shear, add_radiation_artifacts,
        apply_lens_flare, apply_hard_shadows, spoil_image)
    geometry = {}
    if condition in {"shear", "combined"}:
        geometry = dict(max_shift_px=parameters["max_shift_px"],
                        control_spacing_px=parameters["control_spacing_px"], scan_axis=parameters["scan_axis"])
    if condition == "shear":
        return apply_synchronous_shear(image, seed=seed, image_only=True, **geometry)[0]
    if condition == "radiation":
        return add_radiation_artifacts(image, seed=seed, num_hits=parameters["num_hits"],
                                      max_streak_length=parameters["max_streak_length"])
    if condition == "lens_flare":
        return apply_lens_flare(image, seed=seed, strength=parameters["flare_strength"])
    if condition == "hard_shadows":
        return apply_hard_shadows(image, seed=seed, strength=parameters["shadow_strength"])
    if condition == "combined":
        return spoil_image(image, seed=seed, image_only=True, **geometry, num_hits=parameters["num_hits"],
            max_streak_length=parameters["max_streak_length"], flare_strength=parameters["flare_strength"],
            shadow_strength=parameters["shadow_strength"])[0]
    raise ValueError(f"Unknown distortion: {condition}")


def prepare_inputs(config, root, *, limit=None):
    dataset, generation = config["dataset"], config["generation"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive number of source images")
    selected = [row for row in read_jsonl(dataset["manifest"])
                if row.get("planet") == "moon" and row.get("source_dataset") == dataset["source_dataset"]
                and row.get("mask_semantics") == "imp" and row.get("source_split") in dataset["splits"]
                and row.get("evaluation_eligible") is True]
    selected.sort(key=lambda row: row["sample_id"])
    if limit is not None:
        selected = selected[:limit]
    if not selected or len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("Expected unique held-out lunar IMP inputs")
    sources = []
    for row in selected:
        image = (root / row["image"]).resolve()
        if not image.is_relative_to(root / "data_orbital") or not image.is_file():
            raise ValueError(f"Unsafe/missing orbital image: {image}")
        if file_hash(image) != row["image_sha256"]:
            raise ValueError(f"Source image checksum changed: {image}")
        sources.append((row, image))
    folder = dataset["prepared_dir"]
    if limit is not None:
        folder = folder.with_name(folder.name + f"_limit{limit}")
    protocol = {"version": "full-systems-imp-png-v1", "limit": limit,
                "source_manifest_sha256": file_hash(dataset["manifest"]),
                "generation": generation, "distortion": config["distortion"],
                "conditions": dataset["conditions"], "splits": dataset["splits"],
                "source_ids": [row["sample_id"] for row, _ in sources]}
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    metadata_path = folder / "preparation.json"
    if folder.exists():
        if not metadata_path.is_file():
            raise ValueError(f"Incomplete preparation; choose a new prepared_dir: {folder}")
        previous = json.loads(metadata_path.read_text())
        if previous["protocol_sha256"] != digest:
            raise ValueError("Preparation changed; choose a new prepared_dir/run_id")
        for relative, expected in previous["files_sha256"].items():
            path = (folder / relative).resolve()
            if not path.is_relative_to(folder) or not path.is_file() or file_hash(path) != expected:
                raise ValueError("Prepared inputs are missing or changed")
        return folder, previous
    # Validation before writing catches missing dependencies and incompatible PNGs.
    from planetary_vlm.models.ibm_imp import restore_imp_png
    for _, image in sources:
        restore_imp_png(image)
    requests, provenance = [], []
    for row, image in sources:
        seed = int.from_bytes(hashlib.sha256(f"{generation['seed']}:{row['sample_id']}".encode()).digest()[:4], "big")
        for condition in dataset["conditions"]:
            variant = image
            if condition != "clean":
                import numpy as np
                from PIL import Image
                with Image.open(image) as source:
                    pixels = np.array(source)
                changed = transform(pixels, condition, seed, config["distortion"])
                variant = folder / "images" / condition / (row["sample_id"] + ".png")
                variant.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(changed).save(variant)
            request_id = f"{row['sample_id']}__{condition}"
            requests.append(asdict(ModelRequest(request_id, (str(variant),), generation["prompt"], (),
                max_new_tokens=generation["max_new_tokens"], do_sample=False)))
            provenance.append({"request_id": request_id, "source_sample_id": row["sample_id"],
                "condition": condition, "seed": seed, "scene_group_id": row["scene_group_id"],
                "source_split": row["source_split"], "source_dataset": row["source_dataset"],
                "source_revision": row["source_revision"], "source_image_id": row["source_image_id"],
                "license_claim": row["license_claim"], "image_sha256": file_hash(variant),
                "input_protocol": "curated_imp_png_reflectance_0_0.12_v1"})
    write_jsonl(folder / "requests.jsonl", requests)
    write_jsonl(folder / "provenance.jsonl", provenance)
    files = {str(path.relative_to(folder)): file_hash(path) for path in folder.rglob("*") if path.is_file()}
    metadata = {"protocol": protocol, "protocol_sha256": digest, "files_sha256": files,
                "source_images": len(sources), "requests": len(requests),
                "scientific_result": False,
                "limitation": "Curated 8-bit PNGs: irreversible reflectance clipping, quantization and nodata loss"}
    write_json(metadata_path, metadata)
    return folder, metadata


def execute_pipeline(path, *, prepare_only=False, check_only=False, resume=False, model=None, limit=None):
    if prepare_only and check_only:
        raise ValueError("Choose prepare-only or check-only")
    config, root, models = load_pipeline(path)
    if model is not None and model not in models:
        raise ValueError(f"Unknown model: {model}")
    folder, metadata = prepare_inputs(config, root, limit=limit)
    result = {"prepared_dir": str(folder), "source_images": metadata["source_images"],
              "requests": metadata["requests"], "scientific_result": False, "models": {}}
    from planetary_vlm.models.registry import create_adapter
    for name, run_config in models.items():
        run_config.output_dir.mkdir(parents=True, exist_ok=True)
        if model is not None and name != model:
            continue
        if prepare_only:
            result["models"][name] = {"status": "prepared", "output_dir": str(run_config.output_dir)}
            continue
        actual = replace(run_config, requests=folder / "requests.jsonl",
                         run_id=run_config.run_id + (f"_limit{limit}" if limit is not None else ""))
        try:
            # All asset gates are local and run before model imports or GPU load.
            adapter = create_adapter(actual.backend, actual.options)
            adapter.preflight()
            if check_only:
                result["models"][name] = {"status": "assets_available", "gpu_verified": False}
                continue
            destination = run(actual, resume=resume and (actual.output_dir / actual.run_id).exists())
            rows = read_jsonl(destination / "predictions.jsonl")
            counts = {status: sum(row["status"] == status for row in rows) for status in ("ok", "error", "unsupported")}
            result["models"][name] = {"status": "finished" if counts["ok"] == len(rows) else "incomplete",
                                     "output_dir": str(destination), "counts": counts}
        except Exception as error:
            # One unavailable system does not prevent the other system's run.
            result["models"][name] = {"status": "blocked", "error": f"{type(error).__name__}: {error}"}
    return result
