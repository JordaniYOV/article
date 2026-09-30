"""Fetch only non-train LROC NAC image/COCO mask pairs from a small dataset."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_orbital"
REPO = "F1nnSBK/lunar-debris-and-voids"
API = f"https://huggingface.co/api/datasets/{REPO}"
OUT = DATA / "raw" / "moon_debris_nontrain"
CLASS_ID = {"pit": 1, "stone": 2, "crater": 3}


def get_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "planetary-orbital-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def get_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "planetary-orbital-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_file(paths: list[str], wanted: str) -> str:
    matches = [p for p in paths if p == wanted or p.endswith("/" + wanted)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one repo file for {wanted}; found {matches[:5]}")
    return matches[0]


def resolve_image_file(paths: list[str], split: str, coco_name: str) -> str:
    direct = [p for p in paths if p.endswith("/" + coco_name) or p == coco_name]
    if len(direct) == 1:
        return direct[0]
    match = re.match(r"patch_(M\d+[LR]C)_", Path(coco_name).name)
    if not match:
        raise RuntimeError(f"Cannot map COCO patch name to LROC product: {coco_name}")
    product = match.group(1)
    candidates = [p for p in paths if p.startswith(f"images/{split}/pos_{product}_")]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {split} source image for {product}; found {candidates}")
    return candidates[0]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite source cache: {OUT}")
    info = get_json(API)
    revision = info.get("sha")
    if not revision:
        raise RuntimeError("Hub API did not return a pinned dataset revision")
    tree = get_json(f"{API}/tree/{revision}?recursive=true&expand=false")
    paths = [item["path"] for item in tree if item.get("type") == "file"]
    split_jsons = {}
    for split in ("val", "test"):
        name = f"instances_{split}.json"
        p = resolve_file(paths, name)
        payload = get_bytes(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{p}")
        split_jsons[split] = (p, payload, json.loads(payload))
    total = sum(len(data[2].get("images", [])) for data in split_jsons.values())
    if total != 9:
        raise RuntimeError(f"Expected nine non-train images in val/test; found {total}")
    all_paths = []
    manifest = []
    for split, (ann_path, ann_bytes, coco) in split_jsons.items():
        for image_info in coco.get("images", []):
            rel_name = image_info["file_name"]
            image_repo_path = resolve_image_file(paths, split, rel_name)
            image_bytes = get_bytes(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{image_repo_path}")
            image = Image.open(__import__("io").BytesIO(image_bytes)).convert("L")
            if image.size != (256, 256):
                raise ValueError(f"Unexpected image shape for {rel_name}: {image.size}")
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            instances = []
            for ann in coco.get("annotations", []):
                if ann.get("image_id") != image_info["id"]:
                    continue
                category_name = next((c["name"].lower() for c in coco.get("categories", []) if c["id"] == ann["category_id"]), None)
                class_id = CLASS_ID.get(category_name)
                segmentation = ann.get("segmentation")
                if class_id is None or not isinstance(segmentation, list):
                    raise ValueError(f"Unexpected category/segmentation in {rel_name}: {category_name}")
                for polygon in segmentation:
                    if len(polygon) >= 6 and len(polygon) % 2 == 0:
                        draw.polygon([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)], fill=class_id)
                instances.append({"annotation_id": ann["id"], "class": category_name, "class_id": class_id})
            if not instances:
                raise ValueError(f"No annotated instances for {rel_name}")
            image_id = str(image_info["id"])
            folder = OUT / split / image_id
            folder.mkdir(parents=True)
            image_path = folder / "image.png"
            mask_path = folder / "instance_class_mask.png"
            image.save(image_path)
            mask.save(mask_path)
            rel_image = image_path.relative_to(ROOT).as_posix()
            rel_mask = mask_path.relative_to(ROOT).as_posix()
            all_paths.extend((rel_image, rel_mask))
            manifest.append({
                "sample_id": f"moon_debris_{split}_{image_id}", "planet": "moon",
                "dataset": REPO, "source_repo": REPO, "source_revision": revision,
                "source_split": split, "view": "orbital", "source_image_id": rel_name,
                "scene_group_id": rel_name, "classes": ["Background", "Pit", "Stone", "Crater"],
                "instances": instances, "instance_count": len(instances),
                "annotation_method": "MobileSAM box-prompt to polygon with manual fallback (dataset card)",
                "annotation_format": "COCO polygons rasterized into class-ID mask by this converter",
                "license_claim": "CC-BY-4.0 (dataset card)",
                "image": rel_image, "mask": rel_mask,
                "image_sha256": sha(image_path.read_bytes()), "mask_sha256": sha(mask_path.read_bytes()),
                "height_map": None, "evaluation_eligible": True,
                "final_test_eligible": split == "test",
            })
    (OUT / "acquisition_audit.json").write_text(json.dumps({
        "source_repo": REPO, "source_revision": revision,
        "source_splits_downloaded": ["val", "test"], "rows_acquired": len(manifest),
        "split_counts": {s: len(x[2].get("images", [])) for s, x in split_jsons.items()},
        "annotation_json_sha256": {s: sha(x[1]) for s, x in split_jsons.items()},
        "files_downloaded": all_paths, "train_split_downloaded": False,
        "license_claim": "CC-BY-4.0 (dataset card)",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (DATA / "source_index" / "moon_debris_nontrain.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in manifest), encoding="utf-8")
    print(json.dumps({"revision": revision, "rows": len(manifest), "splits": {s: len(x[2].get("images", [])) for s, x in split_jsons.items()}}, indent=2))


if __name__ == "__main__":
    main()
