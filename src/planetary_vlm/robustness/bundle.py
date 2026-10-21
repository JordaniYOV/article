"""Materialize shared PNGs once; keep model inputs and transformed targets separate."""
import io
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

import numpy as np
from PIL import Image

from planetary_vlm.adaptation.common.io import atomic, fingerprint, gpu_lock, guard, orbital, read, rows, sha, within


def implementation():
    package = Path(__file__).resolve().parents[1]
    shared = package.parent / "benchmark_services/common/transforms.py"
    paths = sorted((package / "distortion").glob("*.py"))
    paths += sorted(Path(__file__).parent.glob("*.py"))
    paths += [Path(__file__).with_name("protocol.toml"), shared]
    return {p.relative_to(package.parent).as_posix(): sha(p) for p in paths}


def renderer():
    result = {"python": platform.python_version()}
    for name in ("numpy", "Pillow", "opencv-python-headless", "opencv-python"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def definition(index, protocol, selected, split):
    controls = index.controls()
    source = selected + controls
    return {"schema": "mars-corruption-bundle-v1", "source_dataset": index.identity,
            "split": split, "protocol": protocol, "implementation": implementation(),
            "selected_ids": [r["sample_id"] for r in selected],
            "control_ids": [r["sample_id"] for r in controls],
            "source_assets": [{"sample_id": r["sample_id"], "image_sha256": r["image_sha256"],
                               "mask_sha256": index.targets[r["sample_id"]]["mask_sha256"]} for r in source],
            "target_policy": "photometric_unchanged; shear_nearest_same_seed; 255_border_ignore; full_catalog_under_overlays"}


def save_png(path, pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    payload = stream.getvalue()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Orphan PNG differs; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.partial")
    temporary.write_bytes(payload)
    temporary.replace(path)


def read_target(root, row):
    with Image.open(orbital(root, row["mask"], row["mask_sha256"])) as image:
        target = np.asarray(image).copy()
    if target.shape != (512, 512) or target.dtype != np.uint8 or not set(np.unique(target)) <= {0, 1, 255}:
        raise ValueError("Invalid prepared target grid/classes")
    if not (target != 255).any():
        raise ValueError("No evaluable pixels")
    return target


def check_input(root, row):
    path = orbital(root, row["image"], row["image_sha256"])
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (512, 512):
            raise ValueError("Prepared image must remain native 512x512 grayscale")


def descriptors(root, folder, source, condition, role):
    sid, cid = source["sample_id"], condition["id"]
    prefix = folder.relative_to(Path(root).resolve()).as_posix()
    return {"sample_id": sid, "source_split": source["split"], "role": role,
            "scene_group_id": source["scene_group_id"], "condition": cid,
            "source_image_sha256": source["image_sha256"],
            "image": f"{prefix}/images/{cid}/{sid}.png"}


def prepare(root, folder, index, protocol, selected, split):
    folder = Path(folder)
    # Reject a symlink back into original data, or another pre-existing collection.
    source_parent = index.folder.resolve()
    if folder == source_parent or folder in source_parent.parents or folder.is_relative_to(source_parent):
        raise ValueError("Corruption bundle must be separate from the original dataset")
    if folder.exists() and any(folder.iterdir()) and not (folder / "identity.json").is_file():
        raise ValueError("Existing non-bundle directory is protected")
    expected = definition(index, protocol, selected, split)
    folder.mkdir(parents=True, exist_ok=True)
    with gpu_lock(folder):  # Separate bundle writer lock, not the workspace GPU lock.
        guard(folder / "identity.json", expected)
        if (folder / "complete.json").is_file():
            verify(root, folder, expected)
            return {"bundle": str(folder), "reused_complete_bundle": True}
        from benchmark_services.common.transforms import apply_chain, effect_seed, validate_parameters
        for condition in protocol["conditions"]:
            validate_parameters(condition["parameters"])
        guard(folder / "renderer.json", renderer())
        inputs, targets = [], []
        controls = index.controls()
        sources = [(r, "evaluation") for r in selected] + [(r, "control") for r in controls]
        total = len(sources) * len(protocol["conditions"])
        for condition in protocol["conditions"]:
            cid = condition["id"]
            for source, role in sources:
                sid = source["sample_id"]
                record_path = within(folder, f"records/{cid}/{sid}.json")
                descriptor = descriptors(root, folder, source, condition, role)
                target_descriptor = {"sample_id": sid, "condition": cid,
                    "source_mask_sha256": index.targets[sid]["mask_sha256"],
                    "mask": (folder / f"targets/{cid}/{sid}.png").relative_to(Path(root).resolve()).as_posix()}
                # Source bytes are checked even when an interrupted build is resumed.
                orbital(root, source["image"], source["image_sha256"])
                orbital(root, index.targets[sid]["mask"], target_descriptor["source_mask_sha256"])
                if record_path.is_file():
                    record = read(record_path)
                    if (any(record["input"].get(k) != v for k, v in descriptor.items())
                            or any(record["target"].get(k) != v for k, v in target_descriptor.items())):
                        raise ValueError("Prepared record source/condition differs")
                    check_input(root, record["input"])
                    read_target(root, record["target"])
                else:
                    with Image.open(orbital(root, source["image"], source["image_sha256"])) as image:
                        pixels = np.asarray(image).copy()
                        if image.mode != "L" or pixels.shape != (512, 512):
                            raise ValueError("Source image grid/format differs")
                    target = index.mask(sid)
                    pixels, target = apply_chain(pixels, condition["chain"], condition["parameters"],
                                                 protocol["seed"], sid, mask=target)
                    image_path = Path(root) / descriptor["image"]
                    if cid == "clean":
                        # Preserve clean PNG bytes, not just decoded pixel equality.
                        original = orbital(root, source["image"], source["image_sha256"]).read_bytes()
                        if image_path.exists() and image_path.read_bytes() != original:
                            raise ValueError("Orphan clean image differs")
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        if not image_path.exists():
                            temporary = image_path.with_suffix(".png.partial")
                            temporary.write_bytes(original)
                            temporary.replace(image_path)
                    else:
                        save_png(image_path, pixels)
                    target_path = Path(root) / target_descriptor["mask"]
                    save_png(target_path, target)
                    record = {"input": {**descriptor, "image_sha256": sha(image_path)},
                              "target": {**target_descriptor, "mask_sha256": sha(target_path)},
                              "effect_seeds": {e: effect_seed(protocol["seed"], sid, e) for e in condition["chain"]}}
                    atomic(record_path, record)
                inputs.append(record["input"])
                targets.append(record["target"])
                if len(inputs) % 100 == 0 or len(inputs) == total:
                    print(f"prepare_completed={len(inputs)}/{total}", flush=True)
                    atomic(folder / "progress.json", {"completed": len(inputs), "total": total})
        atomic(folder / "inputs.jsonl", inputs, lines=True)
        atomic(folder / "targets.jsonl", targets, lines=True)
        atomic(folder / "complete.json", {"identity_sha256": fingerprint(expected), "count": len(inputs),
               "files": {name: sha(folder / name) for name in ("inputs.jsonl", "targets.jsonl", "renderer.json")}})
        verify(root, folder, expected)
    return {"bundle": str(folder), "images": len(inputs), "reused_complete_bundle": False}


def verify(root, folder, expected):
    """No transformation/model call: all bytes must already exist before inference."""
    folder = Path(folder)
    if read(folder / "identity.json") != expected or expected["implementation"] != implementation():
        raise ValueError("Bundle definition/code changed; use a new versioned directory")
    marker = read(folder / "complete.json")
    if marker["identity_sha256"] != fingerprint(expected):
        raise ValueError("Incomplete/changed bundle identity")
    for name in ("inputs.jsonl", "targets.jsonl", "renderer.json"):
        if marker["files"].get(name) != sha(folder / name):
            raise ValueError("Prepared manifest/provenance hash changed")
    inputs, targets = rows(folder / "inputs.jsonl"), rows(folder / "targets.jsonl")
    target_map = {(r["condition"], r["sample_id"]): r for r in targets}
    expected_keys = {(c["id"], sid) for c in expected["protocol"]["conditions"]
                     for sid in expected["selected_ids"] + expected["control_ids"]}
    keys = [(r["condition"], r["sample_id"]) for r in inputs]
    if (len(keys) != len(set(keys)) or set(keys) != expected_keys or set(target_map) != expected_keys
            or len(targets) != len(keys) or marker["count"] != len(keys)):
        raise ValueError("Missing/duplicate/foreign prepared records")
    sources = {r["sample_id"]: r for r in expected["source_assets"]}
    prefix = folder.relative_to(Path(root).resolve()).as_posix()
    for row in inputs:
        if set(row) != {"sample_id", "source_split", "role", "scene_group_id", "condition",
                        "source_image_sha256", "image", "image_sha256"}:
            raise ValueError("Unknown/target fields in input manifest")
        is_control = row["sample_id"] in expected["control_ids"]
        if row["role"] != ("control" if is_control else "evaluation") or row["source_split"] != ("train" if is_control else expected["split"]):
            raise ValueError("Prepared roles/splits differ")
        source = sources[row["sample_id"]]
        target = target_map[(row["condition"], row["sample_id"])]
        if (row["source_image_sha256"] != source["image_sha256"]
                or row["image"] != f"{prefix}/images/{row['condition']}/{row['sample_id']}.png"
                or target["source_mask_sha256"] != source["mask_sha256"]
                or target["mask"] != f"{prefix}/targets/{row['condition']}/{row['sample_id']}.png"):
            raise ValueError("Prepared path/source provenance differs")
        check_input(root, row)
        read_target(root, target)
    return inputs, target_map
