"""Recover only selected native five-band WAC rasters for the encoder probe.

Use the pinned SomBench revision and verify each TIFF against the source SHA-256
recorded when the curated RGB image was originally rendered. No full dataset is
downloaded, and the curated 300-image collection is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from linear_head_approach.encoder_probe import SOURCE


ROOT = Path(__file__).resolve().parents[1]
REVISION = "20f800becb64a0c7ad314652a683f97f0281fe67"
DESTINATION = ROOT / "data_orbital" / "encoder_probe_wac_v1" / "native_vis"
BASE_URL = f"https://huggingface.co/datasets/{SOURCE}/resolve/{REVISION}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, path: Path, expected_hash: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if expected_hash is None or sha256(path) == expected_hash:
            return
        raise ValueError(f"Existing file has the wrong SHA-256: {path}")
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "planetary-vlm-benchmark/0.2"})
    for attempt in range(4):
        try:
            if temporary.exists():
                temporary.unlink()
            with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if expected_hash is not None:
                actual_hash = sha256(temporary)
                if actual_hash != expected_hash:
                    raise ValueError(f"Downloaded file has the wrong SHA-256: {url}: {actual_hash}")
            os.replace(temporary, path)
            return
        except (OSError, TimeoutError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
        finally:
            if temporary.exists():
                temporary.unlink()


def run(limit: int | None = None) -> dict:
    manifest = ROOT / "linear_head_approach" / "outputs" / "encoder_probe_wac_v1" / "encoder_inputs.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 120 or any(row["source_dataset"] != SOURCE or row["source_revision"] != REVISION for row in records):
        raise ValueError("Input manifest does not match the pinned 120-image SomBench probe")
    names_by_split: dict[str, dict[int, str]] = {}
    for split in ("val", "test"):
        path = DESTINATION / f"{split}_coco.json"
        fetch(f"{BASE_URL}/{split}.json?download=true", path)
        coco = json.loads(path.read_text(encoding="utf-8"))
        names_by_split[split] = {int(image["id"]): Path(image["file_name"]).stem for image in coco["images"]}
    selected = records if limit is None else records[:limit]
    output = []
    for row in selected:
        if row["split"] == "train":
            stem = str(row["source_image_id"]).removeprefix("moon_wac_train_")
        elif isinstance(row["source_image_id"], int):
            stem = names_by_split[row["split"]][int(row["source_image_id"])]
        else:
            stem = str(row["source_image_id"]).removeprefix(f"moon_wac_{row['split']}_")
            if stem not in names_by_split[row["split"]].values():
                raise ValueError(f"Selected tile absent from {row['split']} COCO: {stem}")
        if not stem or "/" in stem or "\\" in stem or ".." in stem:
            raise ValueError(f"Unsafe WAC stem: {stem}")
        path = DESTINATION / f"{stem}.tif"
        expected = row["native_vis_5band_source_sha256"]
        fetch(f"{BASE_URL}/images_tiff/{stem}.tif?download=true", path, expected)
        output.append({"sample_id": row["sample_id"], "split": row["split"],
                       "scene_group_id": row["scene_group_id"],
                       "native_vis_5band": path.relative_to(ROOT).as_posix(),
                       "sha256": expected})
        if len(output) % 10 == 0:
            print(f"Verified {len(output)}/{len(selected)} selected WAC rasters", flush=True)
    report = {"source_dataset": SOURCE, "source_revision": REVISION,
              "selected": len(selected), "verified": len(output), "full_selection": limit is None,
              "val_coco_sha256": sha256(DESTINATION / "val_coco.json"),
              "test_coco_sha256": sha256(DESTINATION / "test_coco.json")}
    (DESTINATION / "acquisition_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (DESTINATION / "selected_rasters.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Download only the first N rows for a smoke test")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 120:
        parser.error("--limit must be in 1..120")
    print(json.dumps(run(args.limit), indent=2))


if __name__ == "__main__":
    main()
