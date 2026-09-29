"""Check local orbital images, labels, source splits, and simple label balance."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "data_orbital"


def file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_segmentation(name):
    rows = [json.loads(line) for line in (ROOT / "source_index" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    counts = Counter()
    values = Counter()
    samples = []
    for row in rows:
        image_path = ROOT / row["image"]
        mask_path = ROOT / row["mask"]
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size or image.size != (row["width"], row["height"]):
                raise ValueError(f"Dimension mismatch: {row['source_image_id']}")
            array = np.asarray(mask)
            unique = np.unique(array)
            values.update(int(v) for v in unique)
            counts[(row["split"], "nonzero" if np.any(array != 0) else "zero")] += 1
            if len(samples) < 3 and row["split"] == "test":
                samples.append({"image": row["image"], "mask": row["mask"]})
        if "image_sha256" in row and file_hash(image_path) != row["image_sha256"]:
            raise ValueError(f"Image digest mismatch: {row['source_image_id']}")
        if "mask_sha256" in row and file_hash(mask_path) != row["mask_sha256"]:
            raise ValueError(f"Mask digest mismatch: {row['source_image_id']}")
    return {
        "total": len(rows),
        "label_values_seen_in_tiles": sorted(values),
        "tile_counts": {f"{split}:{state}": count for (split, state), count in sorted(counts.items())},
        "examples": samples,
    }


def audit_nac():
    rows = [json.loads(line) for line in (ROOT / "source_index" / "moon_nac_crater_det.jsonl").read_text(encoding="utf-8").splitlines()]
    counts = Counter()
    for row in rows:
        with Image.open(ROOT / row["image"]) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError(f"NAC dimensions differ: {row['source_image_id']}")
        counts[(row["split"], row["label_set"])] += 1
    return {
        "total": len(rows),
        "tile_counts": {f"{split}:{label_set}": count for (split, label_set), count in sorted(counts.items())},
        "annotation_type": "COCO boxes, not pixel masks",
    }


if __name__ == "__main__":
    report = {name: audit_segmentation(name) for name in
              ("mars_crater_binary_seg", "mars_conequest_seg", "moon_imp_seg")}
    report["moon_nac_crater_det"] = audit_nac()
    destination = ROOT / "manifests" / "asset_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
