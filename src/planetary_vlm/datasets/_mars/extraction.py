"""Materialize the official Mars-Bench MSL test split and its native masks."""

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
from PIL import Image

from planetary_vlm.datasets.sources import save_json


SOURCE_URL = "https://zenodo.org/records/15494736"


def extract(root):
    source = root / "raw" / "mars_bench_msl" / "data.zip"
    destination = root / "interim" / "mars_bench_msl_v1" / "test"
    samples, details = [], []
    with zipfile.ZipFile(source) as archive:
        members = set(archive.namelist())
        images = sorted(name for name in members if
                        name.startswith("test/images/") and name.endswith(".jpg"))
        if not images:
            raise ValueError("Official test images absent")
        for name in images:
            image_name = PurePosixPath(name).name
            if name != "test/images/" + image_name:
                raise ValueError("Unexpected nested archive path")
            sid = PurePosixPath(image_name).stem
            mask_name = f"test/masks/{sid}.png"
            if mask_name not in members:
                raise ValueError(f"No exact mask for {name}")
            local = {}
            hashes = {}
            for folder, member in (("images", name), ("masks", mask_name)):
                content = archive.read(member)  # ZipFile verifies CRC during read.
                path = destination / folder / PurePosixPath(member).name
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if path.read_bytes() != content:
                        raise ValueError(f"Existing extraction differs: {path}")
                else:
                    path.write_bytes(content)
                local[folder] = path
                hashes[folder] = hashlib.sha256(content).hexdigest()
            with Image.open(local["images"]) as image, Image.open(local["masks"]) as mask:
                if image.size != mask.size or mask.mode != "L":
                    raise ValueError(f"Image/mask shape or label mode mismatch: {sid}")
                counts = Counter(np.asarray(mask).ravel().tolist())
                if set(counts) - set(range(7)):
                    raise ValueError(f"Unexpected native mask IDs: {sid}")
                size = list(image.size)
            match = re.fullmatch(r"(?:cr_)?(\d+)_(.+)", sid)
            group = f"MSL_sol_{int(match[1])}" if match else "MSL_unknown_scene_group"
            samples.append({"sample_id": "mb_msl_" + sid,
                            "image_path": "images/" + image_name,
                            "mask_path": "masks/" + sid + ".png",
                            "group_id": group, "split": "test",
                            "license": "CC-BY-4.0 (Mars-Bench paper release claim)",
                            "source_url": SOURCE_URL})
            details.append({"sample_id": "mb_msl_" + sid, "mission_image_id": match[2] if match else None,
                            "original_archive_member": name, "size": size, "sha256": hashes,
                            "native_mask_pixel_counts": dict(counts), "depth_path": None,
                            "physical_traversability_gt": False, "instance_mask_path": None})
    (destination / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
    (destination / "provenance.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details), encoding="utf-8")
    report = {"source_url": SOURCE_URL, "test_image_mask_pairs": len(samples),
              "group_count_conservative_sol_grouping": len({row["group_id"] for row in samples}),
              "mask_label_zero_policy": "unlabeled_background_semantics_requires_audit_do_not_use_as_negative",
              "depth_maps": 0, "instance_masks": 0, "traversal_telemetry": 0,
              "benchmark_questions_generated": False, "training_overlap_audit": "pending"}
    save_json(destination / "extraction.json", report)
    print(json.dumps({"destination": str(destination), **report}, ensure_ascii=False, indent=2))
    return destination / "samples.jsonl"
