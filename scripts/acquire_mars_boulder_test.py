"""Fetch only the four official Mars-Bench boulder test pairs from the Hub."""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_orbital"
REPO = "Mirali33/mb-boulder_seg"
API = f"https://huggingface.co/api/datasets/{REPO}"
OUT = DATA / "raw" / "mars_boulder_seg" / "test"


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "planetary-orbital-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "planetary-orbital-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite boulder source cache: {OUT}")
    info = get_json(API)
    revision = info.get("sha")
    if not revision:
        raise RuntimeError("Hub API did not return a pinned dataset revision")
    tree = get_json(f"{API}/tree/{revision}?recursive=true&expand=false")
    paths = [item["path"] for item in tree if item.get("type") == "file"]
    shards = [p for p in paths if p.rsplit("/", 1)[-1].startswith("test-") and p.endswith(".parquet")]
    if len(shards) != 1:
        raise RuntimeError(f"Expected one test parquet shard, found {shards}")
    shard = shards[0]
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{shard}"
    archive = get_bytes(url)
    table = pq.read_table(io.BytesIO(archive))
    records = table.to_pylist()
    # Dataset card lists exactly four official test images; fail closed if the
    # source changes instead of silently expanding the download.
    if len(records) != 4:
        raise RuntimeError(f"Expected 4 rows in the test split, found {len(records)}")
    manifest = []
    for index, row in enumerate(records, 1):
        class_labels = row.get("class_labels") or []
        if "Boulder" not in class_labels:
            raise ValueError(f"Test row {index} is missing the Boulder class label")
        folder = OUT / f"{index:02d}"
        folder.mkdir(parents=True)
        saved = {}
        for field, filename in (("image", "image.png"), ("mask", "mask.png")):
            value = row.get(field)
            payload = value.get("bytes") if isinstance(value, dict) else None
            if not payload:
                raise ValueError(f"Test row {index} has no embedded {field} bytes")
            im = Image.open(io.BytesIO(payload))
            if im.size != (500, 500):
                raise ValueError(f"Unexpected {field} size in row {index}: {im.size}")
            path = folder / filename
            im.save(path)
            saved[field] = path.relative_to(ROOT).as_posix()
            saved[field + "_sha256"] = digest(path.read_bytes())
            if field == "mask" and len(set(im.getdata())) > 2:
                raise ValueError(f"Unexpected mask labels in row {index}")
        manifest.append({
            "sample_id": f"mars_boulder_test_{index:02d}", "planet": "mars",
            "dataset": REPO, "source_repo": REPO, "source_revision": revision,
            "source_split": "test", "view": "orbital", "classes": class_labels,
            "scene_group_id": row.get("image", {}).get("path") or f"test-row-{index}",
            "source_image_id": row.get("image", {}).get("path") or f"test-row-{index}",
            "license_claim": "CC-BY-4.0 (dataset card)",
            "annotation_method": "planetary-scientist precise polygon outline; dataset card claim",
            "image": saved["image"], "mask": saved["mask"],
            "image_sha256": saved["image_sha256"], "mask_sha256": saved["mask_sha256"],
            "height_map": None, "evaluation_eligible": True, "final_test_eligible": True,
        })
    source_index = DATA / "source_index" / "mars_boulder_seg.jsonl"
    source_index.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in manifest), encoding="utf-8")
    audit = {
        "source_repo": REPO, "source_revision": revision, "source_split": "test",
        "source_file": shard, "source_file_sha256": digest(archive),
        "source_file_bytes": len(archive), "rows_acquired": len(manifest),
        "download_scope": "only the official four-row test parquet shard; no train/validation shards fetched",
        "license_claim": "CC-BY-4.0 (dataset card)",
    }
    (DATA / "raw" / "mars_boulder_seg" / "acquisition_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
