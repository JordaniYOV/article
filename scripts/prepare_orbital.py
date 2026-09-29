"""Extract orbital images and labels while preserving official splits.

Run after acquire_orbital.py with numpy, Pillow and pyarrow installed.
Masks and COCO annotations remain separate from image-only model inputs.
"""

from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "data_orbital"
RAW = ROOT / "raw"
PROCESSED = ROOT / "processed"
MANIFESTS = ROOT / "manifests"


def digest(data):
    return sha256(data).hexdigest()


def save_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path.read_bytes()) != digest(data):
            raise ValueError(f"Existing file differs: {path}")
    else:
        path.write_bytes(data)


def save_rows(name, rows):
    # This is a provenance index, never a model-visible inference request.
    destination = ROOT / "source_index" / f"{name}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                   for row in rows), encoding="utf-8")
    print(f"{name}: {len(rows)} rows, splits={dict(Counter(r['split'] for r in rows))}")
    return rows


def checked_image(data, expected_size):
    with Image.open(BytesIO(data)) as opened:
        if opened.size != expected_size:
            raise ValueError(f"Image size {opened.size} != {expected_size}")
        opened.verify()


def mars(name, group_prefix):
    source = RAW / name
    acquisition = json.loads((source / "acquisition.json").read_text(encoding="utf-8"))
    rows = []
    for split in ("val", "test"):
        shard = source / "data" / f"{split}-00000-of-00001.parquet"
        parquet = pq.ParquetFile(shard)
        seen = set()
        for batch in parquet.iter_batches(batch_size=32):
            for record in batch.to_pylist():
                image = record["image"]
                mask = record["mask"]
                source_id = Path(image["path"]).name
                if not source_id or source_id in seen:
                    raise ValueError(f"Missing or duplicate image name: {name}/{split}/{source_id}")
                seen.add(source_id)
                extension = Path(source_id).suffix.lower()
                if extension not in {".png", ".tif", ".tiff"}:
                    raise ValueError(f"Unexpected image type: {source_id}")
                image_bytes, mask_bytes = image["bytes"], mask["bytes"]
                if not image_bytes or not mask_bytes:
                    raise ValueError(f"Missing image/mask: {name}/{split}/{source_id}")
                size = (record["width"], record["height"])
                checked_image(image_bytes, size)
                checked_image(mask_bytes, size)
                image_path = PROCESSED / name / split / "images" / source_id
                mask_path = PROCESSED / name / split / "masks" / source_id
                save_bytes(image_path, image_bytes)
                save_bytes(mask_path, mask_bytes)
                scene_id = group_prefix(source_id)
                rows.append({
                    "dataset": name, "planet": "mars", "view": "orbital", "split": split,
                    "source_image_id": source_id, "scene_group_id": scene_id,
                    "image": image_path.relative_to(ROOT).as_posix(),
                    "mask": mask_path.relative_to(ROOT).as_posix(),
                    "image_sha256": digest(image_bytes), "mask_sha256": digest(mask_bytes),
                    "width": size[0], "height": size[1],
                    "classes": record["class_labels"],
                    "source_repo": acquisition["repo"], "source_revision": acquisition["revision"],
                    "license_claim": acquisition["license_claim"],
                    "raw_parquet": shard.relative_to(ROOT).as_posix(),
                })
    return save_rows(name, rows)


def moon_imp():
    name = "moon_imp_seg"
    source = RAW / name
    acquisition = json.loads((source / "acquisition.json").read_text(encoding="utf-8"))
    rows = []
    seen = set()
    for split in ("train", "val", "test"):
        stems = [line.strip() for line in (source / f"{split}.txt").read_text().splitlines()
                 if line.strip()]
        for stem in stems:
            if stem in seen:
                raise ValueError(f"IMP tile appears in multiple splits: {stem}")
            seen.add(stem)
            image_path = source / "all" / f"{stem}_img.tif"
            mask_path = source / "all" / f"{stem}_mask.tif"
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing IMP pair: {stem}")
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError(f"IMP dimensions differ: {stem}")
                width, height = image.size
                float_image = np.asarray(image, dtype=np.float32)
                values = np.unique(np.asarray(mask)).tolist()
                if not set(values) <= {0, 1, 255}:
                    raise ValueError(f"Unexpected IMP mask values {values}: {stem}")
            # Fixed reflectance display interval for a standard image-only VLM input.
            # The float32 source and its nodata values remain unchanged in raw/.
            valid = np.isfinite(float_image) & (float_image > -1e20)
            rendered = np.zeros(float_image.shape, dtype=np.uint8)
            rendered[valid] = np.rint(np.clip(float_image[valid] / 0.12, 0, 1) * 255).astype(np.uint8)
            png_path = PROCESSED / name / split / "images" / f"{stem}.png"
            with BytesIO() as buffer:
                Image.fromarray(rendered).save(buffer, format="PNG")
                save_bytes(png_path, buffer.getvalue())
            rows.append({
                "dataset": name, "planet": "moon", "view": "orbital", "split": split,
                "source_image_id": stem, "scene_group_id": stem.split(".ech.cog")[0],
                "image": png_path.relative_to(ROOT).as_posix(),
                "source_reflectance": image_path.relative_to(ROOT).as_posix(),
                "mask": mask_path.relative_to(ROOT).as_posix(),
                "width": width, "height": height, "mask_values": values,
                "display_transform": {"version": "moon-imp-reflectance-png-v1",
                                      "minimum": 0.0, "maximum": 0.12,
                                      "nodata": "nonfinite_or_below_minus_1e20_to_zero"},
                "source_repo": acquisition["repo"], "source_revision": acquisition["revision"],
                "license_claim": acquisition["license_claim"],
            })
    if len(rows) != 130:
        raise ValueError(f"Expected all 130 IMP source pairs, found {len(rows)}")
    return save_rows(name, rows)


def moon_nac():
    name = "moon_nac_crater_det"
    source = RAW / name
    acquisition = json.loads((source / "acquisition.json").read_text(encoding="utf-8"))
    rows = []
    seen = set()
    for split in ("train", "val", "test"):
        annotation_file = source / f"annotations_{split}.json"
        coco = json.loads(annotation_file.read_text(encoding="utf-8"))
        counts = Counter(annotation["image_id"] for annotation in coco["annotations"])
        for entry in coco["images"]:
            if entry["split"] != split:
                raise ValueError(f"NAC split mismatch: {entry['file_name']}")
            name_in_repo = entry["file_name"]
            if name_in_repo in seen:
                raise ValueError(f"NAC tile appears in multiple splits: {name_in_repo}")
            seen.add(name_in_repo)
            raw_image = source / name_in_repo
            image = np.load(raw_image, allow_pickle=False)
            if image.dtype != np.uint8 or image.shape != (entry["height"], entry["width"]):
                raise ValueError(f"NAC array schema mismatch: {name_in_repo}")
            png_path = PROCESSED / name / split / "images" / (Path(name_in_repo).stem + ".png")
            with BytesIO() as buffer:
                Image.fromarray(image).save(buffer, format="PNG")
                save_bytes(png_path, buffer.getvalue())
            rows.append({
                "dataset": name, "planet": "moon", "view": "orbital", "split": split,
                "source_image_id": name_in_repo,
                "scene_group_id": f"{entry['site']}|{entry['label_set']}|{entry['box_no']}",
                "image": png_path.relative_to(ROOT).as_posix(),
                "source_array": raw_image.relative_to(ROOT).as_posix(),
                "annotations": annotation_file.relative_to(ROOT).as_posix(),
                "annotation_image_id": entry["id"], "annotation_count": counts[entry["id"]],
                "annotation_type": "COCO bounding boxes; no pixel mask",
                "label_set": entry["label_set"], "site": entry["site"],
                "width": entry["width"], "height": entry["height"],
                "source_repo": acquisition["repo"], "source_revision": acquisition["revision"],
                "license_claim": acquisition["license_claim"],
            })
    if len(rows) != 408:
        raise ValueError(f"Expected all 408 NAC tiles, found {len(rows)}")
    return save_rows(name, rows)


def audit_groups(all_rows):
    report = {}
    for name, rows in all_rows.items():
        groups = defaultdict(set)
        for row in rows:
            groups[row["split"]].add(row["scene_group_id"])
        report[name] = {
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "group_counts": {split: len(ids) for split, ids in groups.items()},
            "val_test_shared_groups": sorted(groups["val"] & groups["test"]),
            "train_test_shared_groups": sorted(groups["train"] & groups["test"]),
            "grouping_note": "Filename/site grouping is an audit heuristic, not proof of geographic independence",
        }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    destination = MANIFESTS / "split_group_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    all_rows = {
        "mars_crater_binary_seg": mars("mars_crater_binary_seg", lambda s: "_".join(Path(s).stem.split("_")[:4])),
        "mars_conequest_seg": mars("mars_conequest_seg", lambda s: "_".join(Path(s).stem.split("_")[:2])),
        "moon_imp_seg": moon_imp(),
        "moon_nac_crater_det": moon_nac(),
    }
    audit_groups(all_rows)
