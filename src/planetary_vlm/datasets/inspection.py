"""Inspect downloaded arrays and archive schemas without inventing missing labels."""

from collections import Counter
import csv
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image

from planetary_vlm.datasets.sources import save_json


def inspect(root, domains=("moon", "mars")):
    result = {"lunar_pilots": [], "mars_bench": {}, "change4_annotations": {}}
    lunar = root / "raw" / "lumon_change3"
    for path in (sorted((lunar / "images").glob("*.png")) if "moon" in domains else []):
        sid = path.stem
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        depth = np.load(lunar / "depth" / f"{sid}.npy", allow_pickle=False)
        mask = np.load(lunar / "masks" / f"{sid}.npy", allow_pickle=False)
        aligned = depth.shape == mask.shape == (height, width)
        # The supplied binary mask means valid depth, not terrain category IDs.
        valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
        values = depth[valid]
        result["lunar_pilots"].append({
            "sample_id": sid, "image_size": [width, height], "image_mode": mode,
            "depth_shape": list(depth.shape), "depth_dtype": str(depth.dtype),
            "validity_mask_dtype": str(mask.dtype), "validity_mask_values": np.unique(mask).tolist(),
            "pixel_alignment": aligned, "positive_finite_valid_fraction": float(valid.mean()),
            "valid_depth_min": float(values.min()) if values.size else None,
            "valid_depth_max": float(values.max()) if values.size else None,
            "semantic_segmentation_available": False,
            "admitted_to_requested_benchmark": False,
        })
    annotation_file = root / "raw" / "change4_tcam_annotations" / "train.jsonl"
    if "moon" in domains and annotation_file.exists():
        rows = [json.loads(line) for line in annotation_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        result["change4_annotations"] = {
            "rows": len(rows), "nonempty_rows": sum(bool(row["shapes"]) for row in rows),
            "shape_types": dict(Counter(s["shape_type"] for row in rows for s in row["shapes"])),
            "labels": dict(Counter(s["label"] for row in rows for s in row["shapes"])),
            "source_file_examples": [row.get("source_file") for row in rows[:3]],
            "embedded_images": sum(bool(row.get("imageData")) for row in rows),
            "depth_available": False,
        }
    mars = root / "raw" / "mars_bench_msl"
    if "mars" in domains and (mars / "mapping.json").exists():
        result["mars_bench"]["native_mapping"] = json.loads((mars / "mapping.json").read_text())
    if "mars" in domains and (mars / "partitions.zip").exists():
        with zipfile.ZipFile(mars / "partitions.zip") as archive:
            result["mars_bench"]["partition_members"] = archive.namelist()
            result["mars_bench"]["partition_split_counts"] = {
                name: dict(Counter(row["split"] for row in csv.DictReader(
                    io.StringIO(archive.read(name).decode("utf-8-sig")))))
                for name in archive.namelist() if name.endswith(".csv")}
            result["mars_bench"]["partition_examples"] = {
                name: archive.read(name).decode("utf-8-sig")[:1500]
                for name in archive.namelist() if name.endswith((".txt", ".csv", ".json"))}
    if "mars" in domains and (mars / "data.zip").exists():
        with zipfile.ZipFile(mars / "data.zip") as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f"Archive CRC failure: {bad_member}")
            members = [name for name in archive.namelist() if not name.endswith("/")]
            result["mars_bench"]["archive_extensions"] = dict(Counter(PurePosixPath(name).suffix for name in members))
            result["mars_bench"]["archive_member_examples"] = members[:20]
            result["mars_bench"]["root_directories"] = dict(Counter(PurePosixPath(name).parts[0] for name in members))
            examples = {}
            for name in members:
                if name.endswith((".png", ".jpg", ".jpeg")):
                    group = str(PurePosixPath(name).parent)
                    if group not in examples:
                        with archive.open(name) as stream, Image.open(stream) as image:
                            values = np.asarray(image)
                            examples[group] = {"path": name, "mode": image.mode, "size": list(image.size),
                                               "unique_values": np.unique(values).tolist()[:256] if values.ndim == 2 else None}
            result["mars_bench"]["image_examples"] = examples
    name = "file_validation.json" if len(domains) == 2 else f"{domains[0]}_file_validation.json"
    save_json(root / "source_audit_v1" / name, result)
    summary = {**result, "mars_bench": {key: value for key, value in result["mars_bench"].items()
                                       if key != "partition_examples"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return result
