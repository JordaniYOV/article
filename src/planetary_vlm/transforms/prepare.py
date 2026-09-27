"""Prepare new requests without reading targets or annotation masks."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tomllib

from planetary_vlm.datasets.manifest import load_requests, check_assets
from planetary_vlm.io import read_jsonl, write_json, write_jsonl
from planetary_vlm.transforms.images import apply_image_transform


def prepare(request_path: str | Path, config_path: str | Path,
            destination: str | Path) -> Path:
    requests = load_requests(request_path)
    check_assets(requests)
    configuration = tomllib.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    if set(configuration) - {"seed", "variants"}:
        raise ValueError("Transform config accepts seed and variants only")
    seed = configuration.get("seed", 42)
    if type(seed) is not int or seed < 0:
        raise ValueError("Transform seed must be a nonnegative integer")
    variants = configuration.get("variants", [])
    if not variants:
        raise ValueError("No variants configured")
    ids = []
    for variant in variants:
        if set(variant) - {"id", "transform", "parameters"}:
            raise ValueError("Unknown variant fields")
        if not isinstance(variant.get("id"), str) or not variant["id"]:
            raise ValueError("Each variant needs an ID")
        if variant.get("transform") not in {"clean", "blur", "noise", "low_light", "contrast", "occlusion"}:
            raise ValueError("Unknown transform")
        if not isinstance(variant.get("parameters", {}), dict):
            raise ValueError("Variant parameters must be a table")
        ids.append(variant["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate condition IDs")
    for request in requests:
        if request.modality_paths or not request.image_paths:
            raise ValueError("Image corruption pipeline requires image-only requests; native modalities need their own processor")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    records, mapping, cache = [], [], {}
    source_hashes = {source: hashlib.sha256(Path(source).read_bytes()).hexdigest()
                     for request in requests for source in request.image_paths}
    for request in requests:
        for variant in variants:
            condition_fingerprint = hashlib.sha256(json.dumps(
                {"variant": variant, "seed": seed}, sort_keys=True).encode()).hexdigest()
            paths, asset_records = [], []
            for source in request.image_paths:
                # Same scene image receives identical noise for all its questions.
                source_digest = source_hashes[source]
                key = (source_digest, condition_fingerprint)
                if key not in cache:
                    image_seed = int.from_bytes(hashlib.sha256(
                        f"{seed}:{source_digest}:{condition_fingerprint}".encode()).digest()[:8], "big")
                    output_name = hashlib.sha256(f"{source_digest}:{condition_fingerprint}".encode()).hexdigest() + ".png"
                    output = destination / "images" / output_name
                    apply_image_transform(source, output, variant["transform"],
                                          variant.get("parameters", {}), image_seed)
                    cache[key] = (str(output), image_seed)
                path, image_seed = cache[key]
                paths.append(path)
                asset_records.append({"source": source, "source_sha256": source_digest,
                                      "output": path, "seed": image_seed})
            identity = {"parent_request_id": request.request_id, "condition": condition_fingerprint,
                        "sources": [source_hashes[source] for source in request.image_paths],
                        "prompt": request.prompt, "allowed_answers": request.allowed_answers,
                        "max_new_tokens": request.max_new_tokens, "do_sample": request.do_sample}
            request_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            records.append(asdict(replace(request, request_id=request_id, image_paths=tuple(paths))))
            mapping.append({"request_id": request_id, "parent_request_id": request.request_id,
                            "condition_id": variant["id"], "transform": variant["transform"],
                            "parameters": variant.get("parameters", {}), "assets": asset_records,
                            "target_visibility": "unknown" if variant["transform"] == "occlusion" else "not_assessed"})
    write_jsonl(destination / "requests.jsonl", records)
    write_jsonl(destination / "variants.jsonl", mapping)
    write_json(destination / "processing.json", {"schema_version": "1.0", "seed": seed,
                                                "config": configuration, "scientific_parameters_frozen": False})
    return destination


def expand_targets(target_path: str | Path, mapping_path: str | Path,
                   output: str | Path) -> None:
    """Separate evaluation step; it does not expose labels to the processor/model."""
    targets = read_jsonl(target_path)
    ids = [item.get("request_id") for item in targets]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Invalid or duplicate target IDs")
    by_id = {item["request_id"]: item for item in targets}
    expanded, used = [], set()
    for variant in read_jsonl(mapping_path):
        if variant.get("parent_request_id") not in by_id:
            raise ValueError("Variant has no source target")
        if variant.get("request_id") in used:
            raise ValueError("Duplicate variant IDs")
        used.add(variant["request_id"])
        expanded.append({**by_id[variant["parent_request_id"]],
                         "request_id": variant["request_id"], "condition_id": variant["condition_id"]})
    write_jsonl(output, expanded)
