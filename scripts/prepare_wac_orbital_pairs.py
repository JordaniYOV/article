"""Prepare a small lunar WAC crater-mask + co-registered elevation subset.

Only selected rows from the pinned SomBench crater benchmark are materialized.
The official val and test rows are retained as distinct evaluation strata.
"""
from __future__ import annotations

import hashlib
import json
import concurrent.futures
import time
from urllib.error import URLError
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
import h5py
import hdf5plugin  # registers the Bitshuffle filter used by SomBench NetCDF tiles
import tifffile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_orbital" / "raw" / "moon_wac_crater_det"
DEST = ROOT / "data_orbital" / "working" / "moon" / "wac_crater_height"
REVISION = "20f800becb64a0c7ad314652a683f97f0281fe67"
REPO = "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection"
AWS = "https://nasa-lunar-fm-bench.s3.amazonaws.com/pretraining_dataset/1M_release/WAC_LowRes/dtm"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return
    last_error = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "planetary-vlm-benchmark/0.1"})
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
            if not data:
                raise IOError(f"empty download: {url}")
            path.write_bytes(data)
            return
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed after 5 attempts: {url}: {last_error}")


def diverse_train_rows(table_rows: list[dict], count: int = 50) -> list[dict]:
    """Select source-train tiles round-robin by LTM zone, avoiding eval products."""
    eval_products = set()
    for split in ("val", "test"):
        coco = json.loads((SOURCE / f"{split}.json").read_text(encoding="utf-8"))
        eval_products |= {Path(row["file_name"]).stem.split("_r", 1)[0] for row in coco["images"]}
    by_zone: dict[str, list[dict]] = {}
    for row in table_rows:
        if row["DATASET"] != "train" or row["PRODUCT_ID"] in eval_products:
            continue
        by_zone.setdefault(row["LTM_CODE"], []).append(row)
    for zone in by_zone:
        by_zone[zone].sort(key=lambda r: (r["PRODUCT_ID"], r["ROW"], r["COL"]))
    chosen: list[dict] = []
    zones = sorted(by_zone)
    while len(chosen) < count:
        moved = False
        for zone in zones:
            if by_zone[zone] and len(chosen) < count:
                chosen.append(by_zone[zone].pop(0))
                moved = True
        if not moved:
            raise RuntimeError(f"only {len(chosen)} eligible WAC train tiles")
    return chosen


def load_coco(split: str) -> dict:
    path = SOURCE / f"{split}.json"
    if not path.exists() and split == "train":
        fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/train.json?download=true", path)
    return json.loads(path.read_text(encoding="utf-8"))


def resample_dtm_to_wac_grid(path: Path, meta: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Bilinearly resample 60 m DTM centers onto the north-up 100 m WAC grid."""
    with h5py.File(path, "r") as ds:
        source = np.asarray(ds["band_data"][:], dtype=np.float32)
        x = np.asarray(ds["x"][:], dtype=np.float64)
        y = np.asarray(ds["y"][:], dtype=np.float64)
    if source.shape != (len(y), len(x)):
        raise ValueError(f"unexpected DTM shape/axis order in {path}")
    dx, dy = np.diff(x), np.diff(y)
    if np.all(dx < 0):
        x, source = x[::-1], source[:, ::-1]
    elif not np.all(dx > 0):
        raise ValueError(f"non-monotonic DTM x axis in {path}")
    if np.all(dy < 0):
        y, source = y[::-1], source[::-1, :]
    elif not np.all(dy > 0):
        raise ValueError(f"non-monotonic DTM y axis in {path}")
    xmin, xmax = float(meta["BOUNDS_XMIN"]), float(meta["BOUNDS_XMAX"])
    ymin, ymax = float(meta["BOUNDS_YMIN"]), float(meta["BOUNDS_YMAX"])
    # DTM coordinate centers should sit within the exact tile footprint.
    tolerance = 61.0  # half a native 60 m cell plus projection rounding
    if x[0] < xmin - tolerance or x[-1] > xmax + tolerance or y[0] < ymin - tolerance or y[-1] > ymax + tolerance:
        raise ValueError(f"DTM coordinates fall outside catalog tile bounds: {path}")
    height, width = shape
    target_x = xmin + (np.arange(width, dtype=np.float64) + 0.5) * ((xmax - xmin) / width)
    # A controlled source WAC NetCDF comparison established that TIFF row 0
    # corresponds to descending y (north-up); interpolation preserves that orientation.
    target_y = ymax - (np.arange(height, dtype=np.float64) + 0.5) * ((ymax - ymin) / height)
    col = np.interp(target_x, x, np.arange(len(x), dtype=np.float64))
    row = np.interp(target_y, y, np.arange(len(y), dtype=np.float64))
    c0 = np.floor(col).astype(int).clip(0, len(x) - 1)
    r0 = np.floor(row).astype(int).clip(0, len(y) - 1)
    c1 = np.minimum(c0 + 1, len(x) - 1)
    r1 = np.minimum(r0 + 1, len(y) - 1)
    wc = col - c0
    wr = row - r0
    weights = np.stack(((1 - wr[:, None]) * (1 - wc[None, :]),
                        (1 - wr[:, None]) * wc[None, :],
                        wr[:, None] * (1 - wc[None, :]),
                        wr[:, None] * wc[None, :]))
    vals = np.stack((source[r0[:, None], c0[None, :]], source[r0[:, None], c1[None, :]],
                     source[r1[:, None], c0[None, :]], source[r1[:, None], c1[None, :]]))
    valid = np.isfinite(vals)
    effective_weights = weights * valid
    denominator = effective_weights.sum(axis=0)
    output = np.divide((np.where(valid, vals, 0.0).astype(np.float64) * effective_weights).sum(axis=0),
                       denominator, out=np.full(shape, np.nan, dtype=np.float64), where=denominator > 0)
    return output.astype(np.float32), np.isfinite(output).astype(np.uint8), {
        "native_shape": list(source.shape), "native_resolution_m": 60,
        "target_resolution_m": 100, "source_x_extent_m": [float(x[0]), float(x[-1])],
        "source_y_extent_m": [float(y[0]), float(y[-1])],
        "catalog_bounds_m": [xmin, ymin, xmax, ymax],
    }


def main() -> None:
    metadata = pq.read_table(SOURCE / "metadata.parquet").to_pylist()
    metadata_by_tile = {Path(r["WAC_VIS_TILE"]).name: r for r in metadata}
    eval_records = []
    for split in ("val", "test"):
        coco = load_coco(split)
        annotations_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            annotations_by_image.setdefault(ann["image_id"], []).append(ann)
        for image in coco["images"]:
            stem = Path(image["file_name"]).stem
            source_image = SOURCE / "images_tiff" / f"{stem}.tif"
            if not source_image.is_file():
                fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/images_tiff/{stem}.tif?download=true", source_image)
            row = metadata_by_tile.get(f"{stem}.nc")
            if row is None:
                raise ValueError(f"No SomBench metadata row for {stem}")
            eval_records.append({"split": split, "id": image["id"], "image": image,
                                 "annotations": annotations_by_image.get(image["id"], []), "meta": row})

    train_json = load_coco("train")
    train_rows = diverse_train_rows(metadata, 50)
    train_tile_names = {Path(r["WAC_VIS_TILE"]).stem for r in train_rows}
    train_images = [im for im in train_json["images"] if Path(im["file_name"]).stem in train_tile_names]
    if len(train_images) != 50:
        raise ValueError(f"Only {len(train_images)} of 50 selected train images have COCO labels")
    train_ann: dict[int, list[dict]] = {}
    for ann in train_json["annotations"]:
        train_ann.setdefault(ann["image_id"], []).append(ann)
    for im in train_images:
        stem = Path(im["file_name"]).stem
        row = metadata_by_tile[f"{stem}.nc"]
        eval_records.append({"split": "train", "id": im["id"], "image": im,
                             "annotations": train_ann.get(im["id"], []), "meta": row})

    def materialize(record: dict) -> dict:
        split, image, anns, meta = record["split"], record["image"], record["annotations"], record["meta"]
        stem = Path(image["file_name"]).stem
        source_image = SOURCE / "images_tiff" / f"{stem}.tif"
        if not source_image.exists():
            fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/images_tiff/{stem}.tif?download=true", source_image)
        with tifffile.TiffFile(source_image) as source_tiff:
            if tuple(source_tiff.series[0].shape) != (image["height"], image["width"], 5):
                raise ValueError(f"unexpected WAC image array shape: {source_image}")
        sample_dir = DEST / split / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        target_image = sample_dir / "wac_image.tif"
        target_mask = sample_dir / "crater_polygon_mask.png"
        target_dtm = sample_dir / "paired_dtm_source.nc"
        target_height = sample_dir / "height_map_100m_m.npy"
        target_valid = sample_dir / "height_valid_mask.npy"
        if not target_image.exists():
            target_image.write_bytes(source_image.read_bytes())
        if not target_mask.exists():
            mask = Image.new("L", (image["width"], image["height"]), 0)
            draw = ImageDraw.Draw(mask)
            for ann in anns:
                for polygon in ann.get("segmentation", []):
                    if len(polygon) >= 6:
                        draw.polygon([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)], fill=1)
            mask.save(target_mask)
        if not target_dtm.exists():
            fetch(f"{AWS}/{stem}.nc", target_dtm)
        height_info = resample_dtm_to_wac_grid(target_dtm, meta, (image["height"], image["width"]))
        height_array, height_valid, grid_audit = height_info
        np.save(target_height, height_array, allow_pickle=False)
        np.save(target_valid, height_valid, allow_pickle=False)
        with Image.open(target_mask) as mask_image:
            mask = np.asarray(mask_image, dtype=np.uint8)
        return {
            "sample_id": f"moon_wac_{split}_{stem}", "planet": "moon", "sample_split": split,
            "source": REPO, "source_revision": REVISION, "source_split": split,
            "source_image_name": image["file_name"], "source_image_id": image["id"],
            "mask_semantics": "rasterized publisher COCO crater polygons; polygon outlines from crater catalog",
            "annotation_count": len(anns), "source_height_url": f"{AWS}/{stem}.nc",
            "product_id": meta["PRODUCT_ID"], "ltm_code": meta["LTM_CODE"],
            "row": meta["ROW"], "col": meta["COL"], "center_latitude": meta["CENTER_LATITUDE"],
            "center_longitude": meta["CENTER_LONGITUDE"], "incidence_angle": meta["INCIDENCE_ANGLE"],
            "pixel_scale_m": 100, "shape": [image["height"], image["width"]],
            "image": str(target_image.relative_to(ROOT)).replace("\\", "/"),
            "segmentation_mask": str(target_mask.relative_to(ROOT)).replace("\\", "/"),
            "height_map_file": str(target_dtm.relative_to(ROOT)).replace("\\", "/"),
            "height_map": str(target_height.relative_to(ROOT)).replace("\\", "/"),
            "height_valid_mask": str(target_valid.relative_to(ROOT)).replace("\\", "/"),
            "height_representation": "band_data bilinearly resampled from SomBench WAC LowRes 60 m DTM to 512x512 WAC 100 m grid; elevation in meters",
            "height_valid_pixels": int(height_valid.sum()), "height_invalid_pixels": int(height_valid.size - height_valid.sum()),
            "height_grid_audit": grid_audit,
            "foreground_pixels": int((mask > 0).sum()), "image_sha256": sha256(target_image),
            "mask_sha256": sha256(target_mask), "height_sha256": sha256(target_dtm),
            "height_map_sha256": sha256(target_height), "height_valid_mask_sha256": sha256(target_valid),
        }
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        output_rows = list(pool.map(materialize, eval_records))
    out = DEST / "manifest.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    summary = {"source": REPO, "revision": REVISION, "license_claim": "CC-BY-4.0 (HF card)",
               "eval_val": sum(r["sample_split"] == "val" for r in output_rows),
               "eval_test": sum(r["sample_split"] == "test" for r in output_rows),
               "train": sum(r["sample_split"] == "train" for r in output_rows),
               "unique_samples": len(output_rows), "netcdf_decoded": True,
               "height_processing": "h5py + hdf5plugin decode Bitshuffle-compressed band_data; source DTM 853x853 at 60 m resampled bilinearly onto matching 512x512 WAC 100 m pixel centers, north-up y orientation verified against source WAC VIS NetCDF; elevation units meters per SomBench card",
               "note": "COCO crater polygons rasterized as binary masks; raw DTM NetCDF and decoded/resampled height NPY plus finite-value masks retained."}
    (DEST / "acquisition_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
