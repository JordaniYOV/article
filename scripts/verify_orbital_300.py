"""Validate the final 300-image orbital dataset without reading source caches."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_orbital" / "orbital_300_v3"


def check_asset(row: dict, key: str, sha_key: str) -> Path:
    path = (ROOT / row[key]).resolve()
    if DATA.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Missing or escaped asset: {row[key]}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row[sha_key]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return path


def main() -> None:
    all_rows: list[dict] = []
    report: dict[str, object] = {"counts": {}, "segmentation_class_support": {}}
    for planet in ("mars", "moon"):
        for track in ("segmentation", "height", "train"):
            path = DATA / planet / f"{track}_manifest.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != 50:
                raise ValueError(f"Expected 50 rows in {path}, found {len(rows)}")
            report["counts"][f"{planet}_{track}"] = len(rows)
            support: dict[str, Counter[int]] = {}
            for row in rows:
                image_path = check_asset(row, "image", "image_sha256")
                with Image.open(image_path) as image:
                    if image.mode not in ("RGB", "L"):
                        raise ValueError(f"Unsupported image mode: {image_path}")
                    shape = image.size[::-1]
                if track == "segmentation":
                    if row["source_split"] not in ("val", "test") or not row["evaluation_eligible"]:
                        raise ValueError(f"Training row in segmentation evaluation: {row['sample_id']}")
                    mask_path = check_asset(row, "mask", "mask_sha256")
                    mask = np.asarray(Image.open(mask_path))
                    if mask.shape != shape:
                        raise ValueError(f"Mask shape mismatch: {row['sample_id']}")
                    values = {int(x) for x in np.unique(mask)}
                    if values != set(row["mask_values_present"]) or not any(values):
                        raise ValueError(f"Mask value mismatch: {row['sample_id']}")
                    support.setdefault(row["mask_semantics"], Counter()).update(values)
                else:
                    height = np.load(check_asset(row, "height_map", "height_map_sha256"), allow_pickle=False)
                    if height.shape != shape or not np.isfinite(height).any():
                        raise ValueError(f"Invalid height map: {row['sample_id']}")
                    if track == "height":
                        valid = np.load(check_asset(row, "height_valid_mask", "height_valid_mask_sha256"), allow_pickle=False)
                        if valid.shape != shape or not np.array_equal(valid.astype(bool), np.isfinite(height)):
                            raise ValueError(f"Invalid height validity mask: {row['sample_id']}")
                    elif row["source_split"] != "train" or row.get("evaluation_eligible", False):
                        raise ValueError(f"Invalid train split: {row['sample_id']}")
                    else:
                        mask = np.asarray(Image.open(check_asset(row, "segmentation_mask", "segmentation_mask_sha256")))
                        if mask.shape != shape:
                            raise ValueError(f"Train mask shape mismatch: {row['sample_id']}")
                all_rows.append(row)
            if track == "segmentation":
                report["segmentation_class_support"][planet] = {k: dict(v) for k, v in support.items()}
    if len(all_rows) != 300 or len({row["sample_id"] for row in all_rows}) != 300:
        raise ValueError("Expected 300 unique sample IDs")
    if len({row["image_sha256"] for row in all_rows}) != 300:
        raise ValueError("Duplicate image bytes across tracks")
    moon_train = {r["scene_group_id"] for r in all_rows if r["planet"] == "moon" and r["track"] == "train"}
    moon_wac_eval = {r["scene_group_id"] for r in all_rows if r["planet"] == "moon" and r["track"] != "train"
                     and r["source_dataset"] == "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection"}
    if moon_train & moon_wac_eval:
        raise ValueError("Moon WAC product overlap between train and eval")
    mars_height = [r for r in all_rows if r["planet"] == "mars" and r["track"] == "height"]
    if any(r["nearest_train_grid_manhattan_tiles"] < 3 for r in mars_height):
        raise ValueError("Mars height tile too close to train")
    report["total_images"] = len(all_rows)
    report["moon_wac_train_eval_product_overlap"] = 0
    report["mars_height_min_train_grid_distance_tiles"] = min(r["nearest_train_grid_manhattan_tiles"] for r in mars_height)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
