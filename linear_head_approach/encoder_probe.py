"""Leakage-safe manifest preparation for the lunar WAC crater encoder probe.

This module does not load a model or expose target paths in encoder inputs.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path


SOURCE = "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection"
PROBE_VERSION = "wac-crater-encoder-probe-v1"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _asset(root: Path, relative: str, expected_hash: str) -> None:
    data_root = (root / "data_orbital").resolve()
    path = (root / relative).resolve()
    if data_root not in path.parents or not path.is_file():
        raise ValueError(f"Missing or unsafe orbital asset: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"SHA-256 mismatch: {relative}")


def _height_crater_targets(root: Path, rows: list[dict]) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Rasterize only publisher COCO polygons for selected WAC height tiles."""
    from PIL import Image, ImageDraw

    native = root / "data_orbital" / "encoder_probe_wac_v1" / "native_vis"
    masks = root / "data_orbital" / "encoder_probe_wac_v1" / "derived_masks"
    masks.mkdir(parents=True, exist_ok=True)
    result: dict[str, tuple[str, str]] = {}
    coco_hashes: dict[str, str] = {}
    for split in ("val", "test"):
        coco_path = native / f"{split}_coco.json"
        if not coco_path.is_file():
            raise ValueError(f"Pinned {split} COCO index is required: {coco_path}")
        coco_hashes[split] = hashlib.sha256(coco_path.read_bytes()).hexdigest()
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        images = {Path(image["file_name"]).stem: image for image in coco["images"]}
        annotations: dict[int, list[dict]] = {}
        for annotation in coco["annotations"]:
            annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
        for row in (item for item in rows if item["source_split"] == split):
            stem = str(row["source_image_id"]).removeprefix(f"moon_wac_{split}_")
            image_info = images.get(stem)
            if image_info is None:
                raise ValueError(f"Selected WAC height tile absent from {split} COCO: {stem}")
            with Image.open(root / row["image"]) as displayed:
                size = displayed.size
            if size != (int(image_info["width"]), int(image_info["height"])):
                raise ValueError(f"COCO/image shape mismatch: {stem}")
            mask = Image.new("L", size, 0)
            draw = ImageDraw.Draw(mask)
            for annotation in annotations.get(int(image_info["id"]), []):
                for polygon in annotation.get("segmentation", []):
                    if len(polygon) >= 6:
                        draw.polygon([(polygon[i], polygon[i + 1])
                                      for i in range(0, len(polygon), 2)], fill=1)
            buffer = io.BytesIO()
            mask.save(buffer, format="PNG")
            payload = buffer.getvalue()
            path = masks / f"{row['sample_id']}.png"
            if path.exists() and path.read_bytes() != payload:
                raise ValueError(f"Derived mask differs from existing file: {path}")
            if not path.exists():
                path.write_bytes(payload)
            result[row["sample_id"]] = (path.relative_to(root).as_posix(), hashlib.sha256(payload).hexdigest())
    if len(result) != len(rows):
        raise ValueError("Not every height tile received a publisher polygon mask")
    return result, coco_hashes


def build_wac_crater_probe(root: Path) -> tuple[list[dict], list[dict], dict]:
    """Return separated encoder inputs, targets, and split/provenance audit."""
    root = root.resolve()
    base = root / "data_orbital" / "orbital_300_v3" / "moon"
    train = _rows(base / "train_manifest.jsonl")
    evaluation = [row for row in _rows(base / "segmentation_manifest.jsonl")
                  if row.get("source_dataset") == SOURCE]
    height = [row for row in _rows(base / "height_manifest.jsonl")
              if row.get("source_dataset") == SOURCE]
    if len(train) != 50 or len(evaluation) != 20 or len(height) != 50:
        raise ValueError("Expected 50 WAC train, 20 segmentation and 50 height val/test samples")
    derived_masks, coco_hashes = _height_crater_targets(root, height)
    native_dir = root / "data_orbital" / "encoder_probe_wac_v1" / "native_vis"
    stems_by_id: dict[str, dict[int, str]] = {}
    for split in ("val", "test"):
        coco = json.loads((native_dir / f"{split}_coco.json").read_text(encoding="utf-8"))
        stems_by_id[split] = {int(item["id"]): Path(item["file_name"]).stem for item in coco["images"]}
    inputs: list[dict] = []
    targets: list[dict] = []
    split_by_group: dict[str, str] = {}
    image_hashes: set[str] = set()
    sample_ids: set[str] = set()
    revisions: set[str] = set()
    native_verified = 0
    for row, source_track in ([(item, "train") for item in train]
                              + [(item, "segmentation") for item in evaluation]
                              + [(item, "height") for item in height]):
        split = row.get("source_split")
        is_train = source_track == "train"
        if row.get("source_dataset") != SOURCE or split not in ({"train"} if is_train else {"val", "test"}):
            raise ValueError(f"Incompatible source or split: {row.get('sample_id')}")
        if is_train and (row.get("evaluation_eligible") or row.get("track") != "train"):
            raise ValueError("Train row marked for evaluation")
        if source_track == "segmentation" and (not row.get("evaluation_eligible") or row.get("mask_semantics") != "catalog_crater_polygon"):
            raise ValueError("Evaluation row has incompatible target semantics")
        if source_track == "height" and (not row.get("evaluation_eligible") or row.get("track") != "height"
                                          or row.get("segmentation_mask") is not None):
            raise ValueError("Height row has incompatible source semantics")
        sample_id = str(row["sample_id"])
        group = str(row["scene_group_id"])
        if sample_id in sample_ids or row["image_sha256"] in image_hashes:
            raise ValueError("Duplicate sample or image across probe splits")
        if group in split_by_group and split_by_group[group] != split:
            raise ValueError(f"Scene group crosses splits: {group}")
        sample_ids.add(sample_id)
        image_hashes.add(row["image_sha256"])
        split_by_group[group] = split
        revisions.add(str(row["source_revision"]))
        if source_track == "height":
            mask_path, mask_hash = derived_masks[sample_id]
        else:
            mask_key = "segmentation_mask" if is_train else "mask"
            mask_hash_key = "segmentation_mask_sha256" if is_train else "mask_sha256"
            mask_path, mask_hash = row[mask_key], row[mask_hash_key]
        _asset(root, row["image"], row["image_sha256"])
        _asset(root, mask_path, mask_hash)
        transform = row["display_transform"]
        if transform.get("method") != "per_tile_visible_rgb_percentile_v1" or transform.get("bands_zero_based") != [0, 1, 2]:
            raise ValueError("Unexpected RGB rendering; input protocol requires review")
        source_id = row["source_image_id"]
        if split == "train":
            stem = str(source_id).removeprefix("moon_wac_train_")
        elif isinstance(source_id, int):
            stem = stems_by_id[split][source_id]
        else:
            stem = str(source_id).removeprefix(f"moon_wac_{split}_")
        if not stem or "/" in stem or "\\" in stem or ".." in stem:
            raise ValueError(f"Unsafe native WAC stem: {stem}")
        native = native_dir / f"{stem}.tif"
        native_relative = None
        if native.exists():
            native_relative = native.relative_to(root).as_posix()
            _asset(root, native_relative, transform["source_raster_sha256"])
            native_verified += 1
        inputs.append({
            "probe_version": PROBE_VERSION, "sample_id": sample_id, "split": split,
            "scene_group_id": group, "source_dataset": SOURCE,
            "source_revision": row["source_revision"], "source_image_id": row["source_image_id"],
            "rgb_image": row["image"], "rgb_image_sha256": row["image_sha256"],
            "rgb_display_transform": transform,
            "native_vis_5band_source_sha256": transform["source_raster_sha256"],
            "native_vis_5band_path": native_relative,
        })
        targets.append({
            "probe_version": PROBE_VERSION, "sample_id": sample_id, "split": split,
            "scene_group_id": group, "mask": mask_path,
            "mask_sha256": mask_hash, "target": "catalog_crater_polygon",
            "target_source_track": source_track,
            "foreground_value": 1,
        })
    if len(revisions) != 1:
        raise ValueError("Mixed source revisions in probe")
    counts = dict(Counter(item["split"] for item in inputs))
    if counts != {"train": 50, "val": 39, "test": 31}:
        raise ValueError(f"Unexpected WAC split counts: {counts}")
    audit = {
        "probe_version": PROBE_VERSION, "source_dataset": SOURCE,
        "source_revision": next(iter(revisions)), "split_counts": counts,
        "unique_scene_groups": len(split_by_group), "image_hashes_unique": len(image_hashes),
        "task": "catalog_crater_polygon_segmentation",
        "height_track_publisher_masks_derived": len(derived_masks),
        "publisher_coco_sha256": coco_hashes,
        "native_vis_verified_count": native_verified,
        "ibm_native_wac_vis_status": "verified_source_rasters" if native_verified == len(inputs) else "blocked_missing_selected_5band_rasters",
        "space_llava_tower_status": "identity_vs_base_clip_unverified",
        "scientific_result": False,
    }
    return inputs, targets, audit


def write_probe(root: Path, destination: Path) -> dict:
    inputs, targets, audit = build_wac_crater_probe(root)
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in (("encoder_inputs.jsonl", inputs), ("targets.jsonl", targets)):
        (destination / name).write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    (destination / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit
