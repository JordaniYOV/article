"""Render once in the API environment; both model services consume identical bytes."""

from dataclasses import asdict
from importlib import metadata
import json
import os
from pathlib import Path
import platform

from planetary_vlm.contracts import ModelRequest
from planetary_vlm.datasets.manifest import load_requests
from .files import atomic_json, digest, file_hash, safe_path
from .transforms import apply_chain, variants
from . import settings as paths


class PreparationCancelled(RuntimeError):
    pass


def renderer_provenance():
    versions = {"python": platform.python_version()}
    for name in ("Pillow", "numpy", "opencv-python-headless"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"version": "shared-renderer-v1", "dependencies": versions,
        "code_sha256": {name: file_hash(Path(__file__).with_name(name))
            for name in ("preparation.py", "transforms.py")}}


def bundle_directory(run):
    return safe_path(run.protocol_sha256, paths.WORKSPACE_ROOT / "data_orbital/inference/api")


def expected_requests(run):
    folder = bundle_directory(run)
    requests, records = [], {}
    for source in run.samples:
        image_path = paths.orbital_path(source["image"])
        if file_hash(image_path) != source["image_sha256"]:
            raise ValueError("Source image changed; resume refused")
        for condition, chain in variants(run.protocol):
            request_id = digest([source["sample_id"], condition])
            modalities = ()
            from planetary_vlm.heads_contract import TASK, TRANSFER_PROTOCOL
            if run.protocol["task_id"] == TASK and (source.get("native_vis") or
                    run.protocol.get("preprocessing", {}).get("input_protocol") != TRANSFER_PROTOCOL):
                native = paths.orbital_path(source["native_vis"])
                if file_hash(native) != source["native_vis_sha256"]:
                    raise ValueError("Paired native VIS input changed")
                modalities = (("vis", str(native)),)
            requests.append(ModelRequest(request_id, (str(folder / "images" / (request_id + ".png")),),
                run.protocol["prompt"], tuple(run.protocol["allowed_answers"]),
                max_new_tokens=run.protocol["max_new_tokens"], modality_paths=modalities))
            records[request_id] = {"source_sample_id": source["sample_id"], "scene_group_id": source["scene_group_id"],
                "source_split": source["source_split"], "condition_id": condition, "chain": chain}
            if modalities:
                records[request_id]["vis_sha256"] = source["native_vis_sha256"]
    return requests, records


def load_bundle(run):
    folder = bundle_directory(run)
    meta = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
    if meta["protocol_sha256"] != run.protocol_sha256 or meta["selection_sha256"] != digest(run.samples):
        raise ValueError("Prepared bundle belongs to a different protocol/selection")
    if run.protocol.get("renderer") is not None and meta["renderer"] != run.protocol["renderer"]:
        raise ValueError("Prepared bundle renderer differs from the declared protocol")
    for name in ("requests.jsonl", "records.json"):
        if file_hash(folder / name) != meta["files_sha256"][name]:
            raise ValueError("Prepared bundle manifest changed")
    requests = load_requests(folder / "requests.jsonl", resolve_paths=False)
    expected, expected_records = expected_requests(run)
    if requests != expected:
        raise ValueError("Prepared requests differ from the queued image-only contract")
    records = json.loads((folder / "records.json").read_text(encoding="utf-8"))
    if set(records) != set(expected_records):
        raise ValueError("Prepared record identities differ")
    for request in requests:
        record = records[request.request_id]
        if {key: value for key, value in record.items() if key != "input_sha256"} != expected_records[request.request_id]:
            raise ValueError("Prepared source/condition provenance differs")
        if file_hash(request.image_paths[0]) != record["input_sha256"]:
            raise ValueError("Prepared image changed; resume refused")
    return requests, records, file_hash(folder / "prepared.json")


def prepare_bundle(run, should_stop=lambda: False):
    folder = bundle_directory(run)
    if (folder / "prepared.json").exists():
        return load_bundle(run)
    renderer = renderer_provenance()
    if run.protocol.get("renderer") is not None and run.protocol["renderer"] != renderer:
        raise ValueError("API renderer environment changed; create a new run or restore the saved environment")
    import numpy as np
    from PIL import Image
    requests, records = expected_requests(run)
    index = {request.request_id: request for request in requests}
    for source in run.samples:
        if should_stop():
            raise PreparationCancelled("Preparation stopped")
        with Image.open(paths.orbital_path(source["image"])) as image:
            pixels = np.array(image)
        for condition, chain in variants(run.protocol):
            if should_stop():
                raise PreparationCancelled("Preparation stopped")
            request_id = digest([source["sample_id"], condition])
            destination = Path(index[request_id].image_paths[0])
            transformed, _ = apply_chain(pixels.copy(), chain, run.protocol["parameters"],
                run.protocol["seed"], source["sample_id"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            Image.fromarray(transformed).save(temporary, format="PNG")
            if destination.exists() and file_hash(destination) != file_hash(temporary):
                temporary.unlink()
                raise ValueError("Prepared image changed or transform implementation differs")
            os.replace(temporary, destination)
            records[request_id]["input_sha256"] = file_hash(destination)
    atomic_json(folder / "requests.jsonl", [asdict(request) for request in requests], lines=True)
    atomic_json(folder / "records.json", records)
    atomic_json(folder / "prepared.json", {"protocol_sha256": run.protocol_sha256,
        "selection_sha256": digest(run.samples), "renderer": renderer,
        "files_sha256": {name: file_hash(folder / name) for name in ("requests.jsonl", "records.json")}})
    return load_bundle(run)
