"""Download and materialize the explicitly selected MMLSv2 orbital subset."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data_orbital" / "working" / "mars" / "mmlsv2_manifest.jsonl"
DEST = ROOT / "data_orbital" / "working" / "mars" / "mmlsv2"


def fetch(url: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    last_error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "planetary-vlm-benchmark/0.1"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if not payload:
                raise IOError("empty response")
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                tmp.write(payload)
                temp_path = Path(tmp.name)
            os.replace(temp_path, path)
            return hashlib.sha256(payload).hexdigest()
        except Exception as exc:  # noqa: BLE001 - retry public object-store errors
            last_error = exc
            time.sleep(1 + attempt * 2)
    raise RuntimeError(f"download failed: {url}: {last_error}")


def read_multiband(path: Path) -> np.ndarray:
    with Image.open(path) as tif:
        if tif.n_frames != 128:
            raise ValueError(f"expected 128 TIFF pages, got {tif.n_frames}: {path}")
        bands = [np.asarray(tif.seek(i) or tif, dtype=np.float32) for i in range(tif.n_frames)]
    array = np.stack(bands, axis=0)
    if array.shape != (128, 128, 7):
        raise ValueError(f"unexpected MMLSv2 raster shape {array.shape}: {path}")
    return array


def process(row: dict) -> dict:
    split = "eval" if row["source_split"] in {"test", "val"} else "train"
    sample_dir = DEST / split / row["sample_id"]
    image_path = sample_dir / "image_7band.tif"
    mask_path = sample_dir / "segmentation_mask.tif"
    image_hash = fetch(row["image_url"], image_path)
    mask_hash = fetch(row["mask_url"], mask_path)
    cube = read_multiband(image_path)
    height = cube[:, :, 3].astype(np.float32, copy=False)
    valid = np.isfinite(height)
    with Image.open(mask_path) as tif:
        segmentation = np.asarray(tif, dtype=np.uint8)
    if segmentation.shape != (128, 128) or not set(np.unique(segmentation)).issubset({0, 1}):
        raise ValueError(f"invalid binary segmentation mask: {mask_path}")
    height_path = sample_dir / "height_map_band4_source_scaled.npy"
    valid_path = sample_dir / "height_valid_mask_from_finite_values.npy"
    np.save(height_path, height, allow_pickle=False)
    np.save(valid_path, valid.astype(np.uint8), allow_pickle=False)
    return {
        **row,
        "sample_split": split,
        "evaluation_stratum": row["source_split"] if split == "eval" else None,
        "image": str(image_path.relative_to(ROOT)).replace("\\", "/"),
        "segmentation_mask": str(mask_path.relative_to(ROOT)).replace("\\", "/"),
        "height_map": str(height_path.relative_to(ROOT)).replace("\\", "/"),
        "height_valid_mask": str(valid_path.relative_to(ROOT)).replace("\\", "/"),
        "shape": [128, 128],
        "height_band_index_zero_based": 3,
        "height_units": "source-scaled [0,1] per dataset card; physical conversion not provided",
        "height_valid_pixels": int(valid.sum()),
        "height_invalid_pixels": int((~valid).sum()),
        "segmentation_values": sorted(int(x) for x in np.unique(segmentation)),
        "image_sha256": image_hash,
        "segmentation_sha256": mask_hash,
        "height_map_sha256": hashlib.sha256(height_path.read_bytes()).hexdigest(),
        "height_valid_mask_sha256": hashlib.sha256(valid_path.read_bytes()).hexdigest(),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        materialized = list(pool.map(process, rows))
    materialized.sort(key=lambda row: (row["sample_split"], row["sample_id"]))
    write_jsonl(DEST / "manifest.jsonl", materialized)
    audit = {
        "dataset": "MarsLS/MMLSv2",
        "revision": rows[0]["revision"],
        "license_claim": "CC-BY-4.0 (Hugging Face dataset card)",
        "unique_samples": len(materialized),
        "evaluation_samples": sum(row["sample_split"] == "eval" for row in materialized),
        "evaluation_source_strata": {split: sum(row["evaluation_stratum"] == split for row in materialized)
                                      for split in ("val", "test")},
        "training_samples": sum(row["sample_split"] == "train" for row in materialized),
        "height_channel": "band 4 (zero-based index 3), as documented by source",
        "height_representation": "source-scaled float32; do not interpret as meters",
        "height_validity_mask": "derived by np.isfinite(height); record per-tile valid counts",
        "selection_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "samples": materialized,
    }
    (DEST / "acquisition_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    counts = {split: sum(row["sample_split"] == split for row in materialized) for split in ("eval", "train")}
    print(f"materialized {len(materialized)} unique scenes: {counts}")
    print(f"manifest: {DEST / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
