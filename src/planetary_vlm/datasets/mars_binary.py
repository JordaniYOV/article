"""Pinned Mars-Bench binary crater collection; orbital data only, no models.

The full publisher train/val/test splits are preserved. Image and target
manifests are separate so an inference loader need not see reference masks.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.error import URLError
from urllib.request import Request

from planetary_vlm.network import urlopen


REPO = "Mirali33/mb-crater_binary_seg"
REVISION = "127ebfb30f2ec68781534a3071093a234ec90257"
SCHEMA = "mars-crater-binary-v1"
DEFAULT_OUTPUT = "data_orbital/mars_crater_binary_v1"
EXPECTED_COUNTS = {"train": 3600, "val": 900, "test": 900}
SOURCES = (
    ("train", "train-00000-of-00002.parquet", 355647868,
     "6d7a7c8fa28162aca782fdcabbf2eda570bfa4501bc574007c0edad80d851845"),
    ("train", "train-00001-of-00002.parquet", 355198952,
     "0f17df2248ae0f45dbbb8a652ed9127cb5d696ca91d93711a1e1d674e5d32875"),
    ("val", "val-00000-of-00001.parquet", 173308222,
     "f8e18e699cd6ae6da8bf3d01fdf2972af8e8bacdaa1a0838eedb092cce350cd5"),
    ("test", "test-00000-of-00001.parquet", 168512824,
     "c3e0d5f5407fac371d094b5f267875aad86f558e890974430a99216d14383172"),
)
CARD_SHA = "08262c9a8bf59803b865e9603d7bafc322c67cc0c7bafbdb25277384be2cb609"
CARD_SIZE = 1909
SOURCE_NAME = re.compile(r"(thm_dir_N-?\d+_\d+)_\d+\.png")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path, payload):
    """Create or verify; never replace an existing different dataset asset."""
    path = Path(path)
    if path.exists():
        if sha(path) != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"Immutable dataset file changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_json(path, value):
    _write(path, _json_bytes(value))


def _write_rows(path, records):
    _write(path, "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                         for row in records).encode("utf-8"))


def collection_path(root, output):
    root = Path(root).resolve()
    orbital = (root / "data_orbital").resolve()
    if not orbital.is_relative_to(root):
        raise ValueError("data_orbital must stay inside the workspace")
    output = (root / output).resolve()
    if not output.is_relative_to(orbital) or output.parent != orbital:
        raise ValueError("Use a new direct child of data_orbital, never an archive or workspace root")
    if output.exists() and any(output.iterdir()):
        marker = output / "collection.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")).get("schema") != SCHEMA:
            raise ValueError("Existing directory is not this collection; originals cannot be overwritten")
    return output


def _asset(root, relative, expected=None):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root / "data_orbital") or not path.is_file():
        raise ValueError(f"Missing or unsafe orbital asset: {relative}")
    if expected is not None and sha(path) != expected:
        raise ValueError(f"Asset SHA-256 differs: {relative}")
    return path


@contextmanager
def _lock(destination):
    """OS lock is released after a crash; the tiny marker may safely remain."""
    destination.mkdir(parents=True, exist_ok=True)
    with _child(destination, ".build.lock").open("a+b") as stream:
        if not stream.seek(0, 2):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _download(url, path, size, digest):
    """Bounded HTTPS, resumable Range requests, then pinned size/SHA validation."""
    if path.exists():
        if path.stat().st_size != size or sha(path) != digest:
            raise ValueError(f"Existing source file failed validation: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f"Oversized partial download: {partial.name}")
        if offset == size:
            break
        headers = {"User-Agent": "planetary-mars-binary/1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{size}"):
                        raise ValueError("Server returned an inconsistent Content-Range")
                    mode = "ab"
                elif response.status == 200:
                    # A server may ignore Range; restart only our own partial file.
                    offset, mode = 0, "wb"
                else:
                    raise ValueError(f"Unexpected HTTP status: {response.status}")
                with partial.open(mode) as stream:
                    count, last_log = offset, time.monotonic()
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > size:
                            raise ValueError("Download exceeds pinned source size")
                        stream.write(chunk)
                        if time.monotonic() - last_log > 20:
                            print(f"download {path.name}: {count}/{size} bytes ({100 * count / size:.1f}%)", flush=True)
                            last_log = time.monotonic()
            if partial.stat().st_size == size:
                break
        except (OSError, URLError) as error:
            if attempt == 2:
                raise
            print(f"Download interrupted; retrying verified-range transfer: {type(error).__name__}", flush=True)
    if not partial.exists() or partial.stat().st_size != size or sha(partial) != digest:
        raise ValueError(f"Source size/SHA-256 mismatch; partial retained: {partial.name}")
    partial.replace(path)


def acquire(root, output=DEFAULT_OUTPUT, allow_download=False):
    root = Path(root).resolve()
    destination = collection_path(root, output)
    # Establish ownership before opening the lock, so interrupted acquisition can resume.
    _write_json(destination / "collection.json", {"schema": SCHEMA, "repo": REPO, "revision": REVISION})
    with _lock(destination):
        _child(destination, "sources")
        specs = [(name, size, digest, f"resolve/{REVISION}/data/{name}") for _, name, size, digest in SOURCES]
        specs.append(("README.md", CARD_SIZE, CARD_SHA, f"raw/{REVISION}/README.md"))
        prior = root / "data_orbital/crater_seg_100_v1/sources"
        for name, size, digest, suffix in specs:
            path = _child(destination, "sources/" + name)
            cached = prior / {"test-00000-of-00001.parquet": "mars_test.parquet",
                              "README.md": "mars_README.md"}.get(name, "__unused__")
            if not path.exists() and cached.is_file() and cached.stat().st_size == size and sha(cached) == digest:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached, path)
                print(f"Reused verified local source: {name}", flush=True)
            if not path.exists() and not allow_download:
                raise ValueError("Missing source; pass --allow-download explicitly")
            print(f"Verify/acquire {name} ({size} bytes)", flush=True)
            _download(f"https://huggingface.co/datasets/{REPO}/{suffix}", path, size, digest)
        if "cc-by-4.0" not in _child(destination, "sources/README.md").read_text(encoding="utf-8").casefold():
            raise ValueError("Pinned source license claim missing")
        _write_json(_child(destination, "sources/inventory.json"), {
            "repo": REPO, "revision": REVISION,
            "files": [{"split": split, "file": name, "size": size, "sha256": digest}
                      for split, name, size, digest in SOURCES],
            "card_sha256": CARD_SHA, "license_claim": "CC-BY-4.0, pinned publisher dataset card",
            "publisher_doi": "10.5281/zenodo.15494686",
            "excluded": "Few-shot partition_* shards are subsets of train, not extra independent images",
        })
    return {"sources_verified": len(SOURCES), "dataset": str(destination)}


def _child(destination, relative):
    """Reject symlink/junction escapes before any output write or read."""
    destination = Path(destination).resolve()
    path = (destination / relative).resolve()
    if path == destination or not path.is_relative_to(destination):
        raise ValueError(f"Unsafe collection asset: {relative}")
    return path


def validate_record(record):
    """Validate source schema and native labels without changing either PNG."""
    import numpy as np
    from PIL import Image
    from planetary_vlm.datasets.crater_segmentation import image_signature

    name = record["image"]["path"]
    match = SOURCE_NAME.fullmatch(name)
    if not match or record["mask"]["path"] != name:
        raise ValueError("Unsafe source filename or image/mask pairing differs")
    with Image.open(BytesIO(record["image"]["bytes"])) as image, \
            Image.open(BytesIO(record["mask"]["bytes"])) as mask:
        if image.format != "PNG" or mask.format != "PNG" or image.mode != "L" or mask.mode != "L":
            raise ValueError("Expected unchanged single-channel PNG images and masks")
        if image.size != (512, 512) or mask.size != image.size:
            raise ValueError("Expected aligned native 512x512 image and mask")
        if (record["width"], record["height"]) != image.size:
            raise ValueError("Declared source dimensions differ from PNG")
        labels = np.asarray(mask)
        values = np.unique(labels).tolist()
        if not set(values) <= {0, 1}:
            raise ValueError("Binary masks must contain class IDs 0/1, not display values 255")
        if record["class_labels"] != [["Background", "Crater"][value] for value in values]:
            raise ValueError("Source class_labels differ from actual mask IDs")
        pixels, dhash = image_signature(image)
        foreground = int(np.count_nonzero(labels))
        intensity = np.asarray(image)
        qa = {"size": list(image.size), "image_mode": image.mode, "mask_mode": mask.mode,
              "mask_values": values, "foreground_pixels": foreground,
              "foreground_fraction": foreground / labels.size,
              "constant_image": image.getextrema()[0] == image.getextrema()[1],
              "constant_rows": int(np.count_nonzero(np.ptp(intensity, axis=1) == 0)),
              "constant_columns": int(np.count_nonzero(np.ptp(intensity, axis=0) == 0))}
    return name, match.group(1), pixels, dhash, qa


PRIOR_MANIFESTS = (
    "data_orbital/orbital_300_v4/mars/segmentation_manifest.jsonl",
    "data_orbital/orbital_300_v4/mars/height_manifest.jsonl",
    "data_orbital/orbital_300_v4/mars/train_manifest.jsonl",
    "data_orbital/crater_seg_100_v1/manifest.jsonl",
)


def _prior_cohorts(root):
    """Read only named orbital manifests; never discover/walk the rover archive."""
    from PIL import Image
    from planetary_vlm.datasets.crater_segmentation import image_signature, source_stem

    manifests, records, missing = [], [], []
    for relative in PRIOR_MANIFESTS:
        if not (root / relative).is_file():
            missing.append(relative)
            continue
        path = _asset(root, relative)
        manifests.append({"path": relative, "sha256": sha(path)})
        for row in _rows(path):
            if row.get("planet", "mars") != "mars":
                continue
            image = _asset(root, row["image"], row.get("image_sha256"))
            with Image.open(image) as decoded:
                pixels, dhash = image_signature(decoded)
            records.append({"prior_manifest": relative, "sample_id": row["sample_id"],
                            "source_stem": source_stem(row["source_image_id"]),
                            "scene_group_id": row["scene_group_id"],
                            "rgb_pixels_sha256": pixels, "dhash64": dhash})
    return {"scope": "Named local cohorts only, not a complete model-pretraining leakage audit",
            "manifests": manifests, "missing_manifests": missing, "records": records}


def _duplicate_groups(records, key):
    groups = defaultdict(list)
    for row in records:
        groups[row[key]].append(row)
    return [{"key": value, "sample_ids": [r["sample_id"] for r in rows],
             "splits": sorted({r["split"] for r in rows})}
            for value, rows in sorted(groups.items()) if len(rows) > 1]


def _near_pairs(records):
    """Exact Hamming<=2 search using three chunks (pigeonhole, no ANN recall loss)."""
    buckets, result = defaultdict(list), []
    numbers = [int(row["dhash64"], 16) for row in records]
    for i, value in enumerate(numbers):
        keys = [(0, value & ((1 << 22) - 1)), (1, (value >> 22) & ((1 << 21) - 1)),
                (2, value >> 43)]
        candidates = {j for key in keys for j in buckets[key]}
        for j in sorted(candidates):
            if records[i]["rgb_pixels_sha256"] == records[j]["rgb_pixels_sha256"]:
                continue
            distance = (value ^ numbers[j]).bit_count()
            if distance <= 2:
                result.append({"sample_ids": [records[j]["sample_id"], records[i]["sample_id"]],
                               "splits": [records[j]["split"], records[i]["split"]],
                               "hamming_distance": distance})
        for key in keys:
            buckets[key].append(i)
    return result


def audit_records(records, prior):
    """Inspection findings, not silent filtering or a claim of perfect geospatial separation."""
    from planetary_vlm.datasets.crater_segmentation import source_stem

    split_stats = {}
    for split in EXPECTED_COUNTS:
        selected = [row for row in records if row["split"] == split]
        foreground = sum(row["qa"]["foreground_pixels"] for row in selected)
        split_stats[split] = {
            "images": len(selected), "foreground_pixels": foreground,
            "background_pixels": len(selected) * 512 * 512 - foreground,
            "foreground_fraction": foreground / (len(selected) * 512 * 512) if selected else None,
            "empty_masks": sum(r["qa"]["foreground_pixels"] == 0 for r in selected),
            "constant_images": [r["sample_id"] for r in selected if r["qa"]["constant_image"]],
            "constant_row_or_column_candidates": [r["sample_id"] for r in selected
                if r["qa"].get("constant_rows", 0) or r["qa"].get("constant_columns", 0)],
            "scene_groups": len({r["scene_group_id"] for r in selected}),
        }
    shared_groups = _duplicate_groups(records, "scene_group_id")
    shared_groups = [g for g in shared_groups if len(g["splits"]) > 1]
    encoded = _duplicate_groups(records, "image_sha256")
    decoded = _duplicate_groups(records, "rgb_pixels_sha256")
    names = _duplicate_groups(records, "source_image_id")
    near = _near_pairs(records)
    prior_rows = prior["records"]
    overlaps = []
    for row in records:
        matches = []
        for old in prior_rows:
            reasons = []
            if source_stem(row["source_image_id"]) == old["source_stem"]:
                reasons.append("source_id_overlap")
            if row["rgb_pixels_sha256"] == old["rgb_pixels_sha256"]:
                reasons.append("decoded_pixels_overlap")
            elif (int(row["dhash64"], 16) ^ int(old["dhash64"], 16)).bit_count() <= 2:
                reasons.append("near_duplicate_review")
            if reasons:
                matches.append({"prior_manifest": old["prior_manifest"], "sample_id": old["sample_id"],
                                "reasons": reasons})
        if matches:
            overlaps.append({"sample_id": row["sample_id"], "split": row["split"], "matches": matches})
    prior_groups = {r["scene_group_id"] for r in prior_rows}
    cross_exact = [g for g in decoded + names if len(g["splits"]) > 1]
    return {
        "split_counts": dict(Counter(r["split"] for r in records)), "split_statistics": split_stats,
        "encoded_image_duplicates": encoded, "decoded_image_duplicates": decoded,
        "source_id_duplicates": names, "cross_split_scene_groups": shared_groups,
        "near_duplicate_pairs_dhash_le_2": near,
        "near_duplicate_note": "Review heuristic only; not a geographic-overlap proof or automatic exclusion",
        "prior_local_overlap": overlaps,
        "prior_overlap_counts": dict(Counter(r["split"] for r in overlaps)),
        "groups_shared_with_prior_cohorts": sorted({r["scene_group_id"] for r in records} & prior_groups),
        "spatial_audit_limit": "Source filename groups only; raw footprints and tile adjacency not independently reconstructed",
        "admission": "review_required" if cross_exact or shared_groups or any(
            len(set(p["splits"])) > 1 for p in near) or any(
            s["constant_images"] for s in split_stats.values()) else "publisher_splits_preserved",
        "test_policy": "Keep official test intact. Report previously exposed cases separately; do not call all test images a new blind holdout.",
    }


def _extract(root, destination):
    import pyarrow.parquet as pq

    records, ids = [], set()
    for split, name, _, digest in SOURCES:
        parquet = pq.ParquetFile(_child(destination, "sources/" + name))
        row_index = 0
        print(f"Extract/validate {name}: {parquet.metadata.num_rows} pairs", flush=True)
        for batch in parquet.iter_batches(batch_size=8):
            for record in batch.to_pylist():
                filename, group, pixels, dhash, qa = validate_record(record)
                sample_id = "mars_binary_" + filename[:-4]
                if sample_id in ids:
                    raise ValueError(f"Duplicate source sample ID: {sample_id}")
                ids.add(sample_id)
                row = {"sample_id": sample_id, "planet": "mars", "split": split,
                       "source_split": split, "scene_group_id": group, "source_image_id": filename,
                       "source_dataset": REPO, "source_revision": REVISION,
                       "source_container": "sources/" + name, "source_container_sha256": digest,
                       "source_parquet_row": row_index, "instrument": "THEMIS",
                       "observation": "orbital", "modality": "thermal_infrared_daytime",
                       "width": 512, "height": 512, "channels": 1,
                       "classes": ["Background", "Crater"], "qa": qa,
                       "mask_semantics": "publisher_elliptical_catalog_coverage_not_exhaustive_visible_census",
                       "rgb_pixels_sha256": pixels, "dhash64": dhash}
                for field, folder in (("image", "images"), ("mask", "masks")):
                    path = _child(destination, f"{split}/{folder}/{filename}")
                    payload = record[field]["bytes"]
                    _write(path, payload)
                    row[field] = path.relative_to(root).as_posix()
                    row[field + "_sha256"] = hashlib.sha256(payload).hexdigest()
                records.append(row)
                row_index += 1
        print(f"Validated {len(records)} image-mask pairs so far", flush=True)
    if dict(Counter(row["split"] for row in records)) != EXPECTED_COUNTS:
        raise ValueError("Publisher split counts differ from pinned collection contract")
    return records


INPUT_KEYS = ("sample_id", "planet", "split", "source_split", "scene_group_id", "source_image_id",
              "source_dataset", "source_revision", "instrument", "observation", "modality",
              "width", "height", "channels", "image", "image_sha256")
TARGET_KEYS = ("sample_id", "mask", "mask_sha256", "classes", "mask_semantics")


def _projections(records):
    return ([{key: row[key] for key in INPUT_KEYS} for row in records],
            [{key: row[key] for key in TARGET_KEYS} for row in records])


def _readme(audit):
    return f"""# Mars binary crater segmentation v1

5,400 original orbital image/mask pairs. Official splits: train 3,600; val 900; test 900.
Native images: 512x512 single-channel PNG, THEMIS daytime thermal infrared, Mars Odyssey.
Binary PNG masks contain **0 = Background, 1 = Crater** (NOT 0/255).
Image and mask encoded source bytes are preserved without resizing or conversion.

## Files

- `train/`, `val/`, `test/`: `images/` and `masks/`, paired by identical filename.
- `inputs.jsonl` and per-split `inputs.jsonl`: images only; safe input inventory, no answers/masks.
- `targets.jsonl` and per-split `targets.jsonl`: references; never send to inference.
- `manifest.jsonl`: joined provenance and QA, for preparation/evaluation, NOT model input.
- `audit.json`: counts, mask coverage, exact/near duplicate checks, prior-cohort overlap.
- `prior_cohorts.json`: hash-bound snapshot of the named local overlap-audit scope.
- `test/not_known_exposed_inputs.jsonl`: review candidates only, not certified blind holdout.
- `previews/`: three TRAIN examples, display-only colour overlays; never training targets.
- `metadata.json`: completion marker, dependencies, builder identity and asset manifest hashes.
- `sources/`: pinned original Parquet files and publisher dataset card (CC-BY-4.0 claim).

## Scientific limits

The targets rasterize publisher elliptical crater annotations. They are not exact geological
rim contours or an exhaustive census of all visible craters. Background means unlabeled by
this catalog, not verified physical absence. A blank mask is valid catalog ground truth.
No depth/DEM, new crater types, bounding-box targets or model predictions were fabricated.

Source: https://huggingface.co/datasets/{REPO}/tree/{REVISION}
Publisher: https://doi.org/10.5281/zenodo.15494686
The publisher describes geographically disjoint longitude ranges; our local audit checks
filename mosaic groups and pixel duplicates, not complete georeferenced footprints.
Few-shot partition shards were not added: they repeat subsets of the original training split.

Previously exposed local-image overlaps: {json.dumps(audit['prior_overlap_counts'], sort_keys=True)}.
Original splits are preserved, not silently repartitioned. Previous benchmark images must not
be described as a new blind test. Near-duplicate flags require review; dHash is only a heuristic.
This audit does not establish exclusion from foundation-model pretraining.

THEMIS thermal IR differs from lunar visible/NAC imagery in sensor, spectrum and resolution;
this is not an isolated planet-only domain shift. IBM input compatibility and GPU feasibility
still require a separate adapter and smoke test. No training/inference was performed here.

## Reproduce (CPU only)

Install the project with its `datasets` extra (Python >=3.11): `python -m pip install -e '.[datasets]'`.
Build/resume: `python -m planetary_vlm dataset mars-binary-build --workspace . --allow-download`
Offline verification: `python -m planetary_vlm dataset mars-binary-verify --workspace .`
Downloads resume verified ranges; existing changed assets are never silently overwritten.
"""


def _previews(root, destination, records):
    from PIL import Image
    from planetary_vlm.datasets.mask_preview import render

    selected = [r for r in records if r["split"] == "train"]
    # No new test images are opened for presentation/development.
    chosen = [selected[i] for i in (0, len(selected) // 2, len(selected) - 1)]
    paths = []
    for row in chosen:
        path = _child(destination, "previews/" + row["sample_id"] + ".png")
        if not path.exists():
            with Image.open(_asset(root, row["image"])) as image, Image.open(_asset(root, row["mask"])) as mask:
                panel = render(image, mask, row["classes"], row["sample_id"])
                buffer = BytesIO()
                panel.save(buffer, format="PNG")
                _write(path, buffer.getvalue())
        paths.append(path.relative_to(destination).as_posix())
    return paths


def build_collection(root, output=DEFAULT_OUTPUT, allow_download=False):
    """Resumable acquisition/extraction, then an immutable completed collection."""
    if any(find_spec(name) is None for name in ("PIL", "numpy", "pyarrow")):
        raise ImportError("Install the project's datasets extra before building: pip install -e .[datasets]")
    root = Path(root).resolve()
    destination = collection_path(root, output)
    if _child(destination, "metadata.json").is_file():
        return verify_collection(root, output)
    acquire(root, output, allow_download=allow_download)
    with _lock(destination):
        # Another builder may have finished after our initial check/acquisition.
        if _child(destination, "metadata.json").is_file():
            return verify_collection(root, output)
        records = _extract(root, destination)
        inputs, targets = _projections(records)
        prior_path = _child(destination, "prior_cohorts.json")
        prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else _prior_cohorts(root)
        _write_json(prior_path, prior)
        print("Audit duplicate pixels, scene groups and prior local cohorts", flush=True)
        audit = audit_records(records, prior)
        manifests = {"manifest.jsonl": records, "inputs.jsonl": inputs, "targets.jsonl": targets}
        for split in EXPECTED_COUNTS:
            ids = {r["sample_id"] for r in records if r["split"] == split}
            manifests[f"{split}/inputs.jsonl"] = [r for r in inputs if r["sample_id"] in ids]
            manifests[f"{split}/targets.jsonl"] = [r for r in targets if r["sample_id"] in ids]
        exposed = {r["sample_id"] for r in audit["prior_local_overlap"]}
        manifests["test/not_known_exposed_inputs.jsonl"] = [
            r for r in inputs if r["split"] == "test" and r["sample_id"] not in exposed]
        for name, rows in manifests.items():
            _write_rows(_child(destination, name), rows)
        _write_json(_child(destination, "audit.json"), audit)
        _write(_child(destination, "README.md"), _readme(audit).encode("utf-8"))
        extras = ["collection.json", "prior_cohorts.json", "audit.json", "README.md", "sources/inventory.json"]
        extras += _previews(root, destination, records)
        files = {name: sha(_child(destination, name)) for name in [*manifests, *extras]}
        _write_json(_child(destination, "metadata.json"), {
            "schema": SCHEMA, "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset": REPO, "source_revision": REVISION, "counts": EXPECTED_COUNTS,
            "builder_sha256": sha(Path(__file__)),
            "dependencies": {name: version(name) for name in ("Pillow", "numpy", "pyarrow")},
            "files_sha256": files, "admission": audit["admission"], "no_models_loaded": True,
        })
    return verify_collection(root, output)


def verify_collection(root, dataset=DEFAULT_OUTPUT):
    """Offline byte/pixel/label/source verification; never import Torch or access network."""
    root = Path(root).resolve()
    destination = collection_path(root, dataset)
    metadata = json.loads(_child(destination, "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("schema"), metadata.get("source_dataset"), metadata.get("source_revision"),
            metadata.get("status"), metadata.get("counts")) != (SCHEMA, REPO, REVISION, "complete", EXPECTED_COUNTS):
        raise ValueError("Collection completion/source identity differs")
    for name, digest in metadata["files_sha256"].items():
        if sha(_child(destination, name)) != digest:
            raise ValueError(f"Immutable collection file changed: {name}")
    for _, name, size, digest in SOURCES:
        path = _child(destination, "sources/" + name)
        if path.stat().st_size != size or sha(path) != digest:
            raise ValueError(f"Pinned source SHA-256 differs: {name}")
    card = _child(destination, "sources/README.md")
    if card.stat().st_size != CARD_SIZE or sha(card) != CARD_SHA:
        raise ValueError("Pinned publisher card differs")
    records = _rows(_child(destination, "manifest.jsonl"))
    ids = [r["sample_id"] for r in records]
    if len(set(ids)) != len(ids) or dict(Counter(r["split"] for r in records)) != EXPECTED_COUNTS:
        raise ValueError("Sample identities/counts differ")
    # Compare extracted bytes to pinned Parquet rows, not just self-recorded hashes.
    import pyarrow.parquet as pq
    by_locator = {(r["source_container"], r["source_parquet_row"]): r for r in records}
    count = 0
    for split, name, _, digest in SOURCES:
        index = 0
        for batch in pq.ParquetFile(_child(destination, "sources/" + name)).iter_batches(batch_size=8):
            for source in batch.to_pylist():
                filename, group, pixels, dhash, qa = validate_record(source)
                row = by_locator.get(("sources/" + name, index))
                if row is None or any(row[k] != value for k, value in {
                    "source_image_id": filename, "scene_group_id": group, "rgb_pixels_sha256": pixels,
                    "dhash64": dhash, "qa": qa, "source_revision": REVISION, "source_dataset": REPO,
                    "split": split, "source_split": split, "source_container_sha256": digest,
                }.items()):
                    raise ValueError("Manifest differs from pinned source row")
                for field, folder in (("image", "images"), ("mask", "masks")):
                    expected_path = _child(destination, f"{split}/{folder}/{filename}")
                    path = _asset(root, row[field], row[field + "_sha256"])
                    if path != expected_path or sha(path) != hashlib.sha256(source[field]["bytes"]).hexdigest():
                        raise ValueError(f"Extracted {field} bytes differ from publisher: {filename}")
                index += 1
                count += 1
        print(f"Verified against pinned source {name}: {index} pairs", flush=True)
    inputs, targets = _projections(records)
    if _rows(_child(destination, "inputs.jsonl")) != inputs or _rows(_child(destination, "targets.jsonl")) != targets:
        raise ValueError("Input/target separation differs from manifest")
    prior = json.loads(_child(destination, "prior_cohorts.json").read_text(encoding="utf-8"))
    expected_audit = audit_records(records, prior)
    if json.loads(_child(destination, "audit.json").read_text(encoding="utf-8")) != expected_audit:
        raise ValueError("Audit differs from assets/provenance")
    audit = expected_audit
    for split in EXPECTED_COUNTS:
        split_inputs = [r for r in inputs if r["split"] == split]
        split_ids = {r["sample_id"] for r in split_inputs}
        if (_rows(_child(destination, f"{split}/inputs.jsonl")) != split_inputs or
                _rows(_child(destination, f"{split}/targets.jsonl")) != [
                    r for r in targets if r["sample_id"] in split_ids]):
            raise ValueError("Per-split manifests differ from source split")
    exposed = {r["sample_id"] for r in audit["prior_local_overlap"]}
    if _rows(_child(destination, "test/not_known_exposed_inputs.jsonl")) != [
            r for r in inputs if r["split"] == "test" and r["sample_id"] not in exposed]:
        raise ValueError("Not-known-exposed candidates differ from documented audit")
    return {"valid": True, "pairs_verified": count, "split_counts": audit["split_counts"],
            "dataset": str(destination), "admission": audit["admission"],
            "prior_local_overlap": audit["prior_overlap_counts"],
            "cross_split_scene_groups": len(audit["cross_split_scene_groups"]),
            "cross_split_exact_duplicates": sum(len(g["splits"]) > 1 for g in audit["decoded_image_duplicates"]),
            "near_duplicate_review_pairs": len(audit["near_duplicate_pairs_dhash_le_2"]),
            "no_models_loaded": True}
