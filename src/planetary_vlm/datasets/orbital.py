"""Local-first curated orbital subsets and explicit restoration of missing assets.

Restoration uses pinned source IDs from the recovery specification, never a new
random selection. Only selected assets (or their containing Parquet shards) are
downloaded. Ground truth remains separate from model-visible images.
"""
from collections import Counter
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request
from uuid import uuid4

from planetary_vlm.network import urlopen

from .base import PROJECT_ROOT
from .sources import download

COLLECTION = "orbital_300_v3"
RECOVERY = PROJECT_ROOT / "configs/datasets/orbital_builtin_recovery.json"
LABELS = {"segmentation": "Снимки с масками", "height": "Снимки с картами высот", "imp": "IMP · готовая голова IBM"}
ASSETS = ("image", "mask", "height_map", "height_valid_mask")


def checked_path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root / "data_orbital") or "data_rover" in {p.lower() for p in path.parts}:
        raise ValueError("Orbital assets must stay within data_orbital/")
    return path


def validate_selection(planet, subset):
    if planet not in {"mars", "moon"} or subset not in LABELS or (subset == "imp" and planet != "moon"):
        raise ValueError("Unknown orbital planet/subset")


def recovery_rows(planet, subset):
    validate_selection(planet, subset)
    spec = json.loads(RECOVERY.read_text(encoding="utf-8"))
    return filter_rows(spec["tracks"][planet]["segmentation" if subset == "imp" else subset], planet, subset)


def filter_rows(rows, planet, subset):
    selected, seen = [], set()
    for row in rows:
        if row.get("planet") != planet:
            raise ValueError("Manifest mixes planets")
        if subset == "imp" and row.get("mask_semantics") != "imp":
            continue
        if row.get("evaluation_eligible") is False or row.get("source_split") not in {"val", "test"}:
            continue
        if not row.get("sample_id") or row["sample_id"] in seen:
            raise ValueError("Invalid or duplicate source sample ID")
        if row.get("mask") and row.get("height_map"):
            raise ValueError("Evaluation sample must have either a mask or a height map")
        if not row.get("image") or not row.get("mask" if subset != "height" else "height_map"):
            raise ValueError("Missing image/target reference")
        seen.add(row["sample_id"])
        selected.append(row)
    if not selected:
        raise ValueError("No held-out samples in orbital subset")
    return selected


def inspect_subset(root, planet, subset):
    validate_selection(planet, subset)
    track = "segmentation" if subset == "imp" else subset
    manifest = checked_path(root, f"data_orbital/{COLLECTION}/{planet}/{track}_manifest.jsonl")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()] if manifest.is_file() else recovery_rows(planet, subset)
    rows = filter_rows(rows, planet, subset)
    missing = [value for row in rows for key in ASSETS if (value := row.get(key)) and not checked_path(root, value).is_file()]
    return {"manifest": manifest, "rows": rows, "ready": manifest.is_file() and not missing,
        "missing": missing, "sample_count": len(rows), "split_counts": dict(Counter(r["source_split"] for r in rows))}


def verify_assets(root, rows):
    import hashlib
    for row in rows:
        for key in ASSETS:
            if not row.get(key):
                continue
            expected = row.get(key + "_sha256") or (row.get("height_sha256") if key == "height_map" else None)
            with checked_path(root, row[key]).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if not expected or actual != expected:
                raise ValueError(f"Changed orbital asset: {row['sample_id']}/{key}")


def script(name, root):
    """Reuse existing converters without invoking scripts that rebuild all datasets."""
    spec = spec_from_file_location(f"orbital_restore_{name}_{uuid4().hex}", PROJECT_ROOT / "scripts" / f"{name}.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = Path(root).resolve()
    if hasattr(module, "DATA"):
        module.DATA = module.ROOT / "data_orbital"
    return module


class SourceCache:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = checked_path(root, f"data_orbital/.builtin-cache")
        self.listings = {}
        self.tables = {}

    def tree(self, repo, revision, folder=""):
        key = (repo, revision, folder)
        if key not in self.listings:
            url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{quote(folder, safe='/')}?limit=1000"
            result = []
            for _ in range(100):
                with urlopen(Request(url, headers={"User-Agent": "planetary-orbital-restore/1"}), timeout=45) as response:
                    result.extend(json.load(response))
                    link = response.headers.get("Link", "")
                match = re.search(r'<([^>]+)>;\s*rel="next"', link)
                if not match:
                    break
                url = match[1]
                if not url.startswith(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/"):
                    raise ValueError("Unexpected source pagination URL")
            else:
                raise ValueError("Source inventory exceeds page limit")
            self.listings[key] = result
        return self.listings[key]

    def files(self, repo, revision, folder=""):
        for item in self.tree(repo, revision, folder):
            if item["type"] == "file":
                yield item
            elif item["type"] == "directory":
                yield from self.files(repo, revision, item["path"])

    def find(self, repo, revision, predicate, folder=""):
        matches = [item for item in self.files(repo, revision, folder) if predicate(item["path"])]
        if len(matches) != 1:
            raise ValueError(f"Expected one pinned source file in {repo}, found {len(matches)}")
        return matches[0]

    def asset(self, repo, revision, filename):
        folder, _, name = filename.rpartition("/")
        item = next((i for i in self.tree(repo, revision, folder) if i["type"] == "file" and i["path"] == filename), None)
        if item is None:
            raise ValueError(f"Source file missing at pinned revision: {repo}/{filename}")
        if not 0 < item["size"] <= 1024**3:
            raise ValueError("Source asset exceeds 1 GiB limit")
        dest = self.cache / repo / revision / filename
        if not dest.resolve().is_relative_to(self.cache):
            raise ValueError("Unsafe source path")
        lfs = item.get("lfs") or {}
        sha = lfs.get("oid")
        download(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(filename, safe='/')}",
            dest, expected_size=item["size"], checksum=f"sha256:{sha}" if sha else None)
        return dest

    def table(self, repo, revision, filename):
        key = (repo, revision, filename)
        if key not in self.tables:
            import pyarrow.parquet as pq
            self.tables[key] = pq.read_table(self.asset(repo, revision, filename)).to_pylist()
        return self.tables[key]

    def coco(self, repo, revision, filename):
        return json.loads(self.asset(repo, revision, filename).read_text(encoding="utf-8"))


def save_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def materialize(row, cache, root, staging):
    """Restore one source-labelled image/target pair using existing converters."""
    import numpy as np
    from PIL import Image, ImageDraw
    repo, revision = row["source_dataset"], row["source_revision"]
    source_id, split = str(row["source_image_id"]), row["source_split"]
    if repo.startswith("Mirali33/"):
        import pyarrow.parquet as pq
        shard = cache.find(repo, revision, lambda p: Path(p).name.startswith(f"{split}-") and p.endswith(".parquet"))
        parquet = pq.ParquetFile(cache.asset(repo, revision, shard["path"]))
        for batch in parquet.iter_batches(batch_size=16):
            for record in batch.to_pylist():
                if Path(record["image"]["path"]).name == source_id:
                    payloads = {"image": record["image"]["bytes"], "mask": record["mask"]["bytes"]}
                    if repo == "Mirali33/mb-boulder_seg":
                        # acquire_mars_boulder_test.py decoded embedded TIFFs
                        # and saved PNGs before the curated hashes were recorded.
                        # Reproduce that conversion for both image and mask.
                        for key, payload in payloads.items():
                            with Image.open(BytesIO(payload)) as image:
                                if image.size != (500, 500):
                                    raise ValueError(f"Unexpected boulder {key} size: {image.size}")
                                payloads[key] = save_image(image)
                    return payloads
        raise ValueError(f"Selected image absent from pinned Parquet: {source_id}")
    if row.get("mask_semantics") == "imp":
        source = cache.asset(repo, revision, f"all/{source_id}_img.tif")
        mask = cache.asset(repo, revision, f"all/{source_id}_mask.tif")
        with Image.open(source) as image:
            values = np.asarray(image, dtype=np.float32)
        valid = np.isfinite(values) & (values > -1e20)
        rendered = np.zeros(values.shape, dtype=np.uint8)
        rendered[valid] = np.rint(np.clip(values[valid] / .12, 0, 1) * 255).astype(np.uint8)
        return {"image": save_image(Image.fromarray(rendered)), "mask": mask.read_bytes()}
    if repo == "F1nnSBK/lunar-debris-and-voids":
        converter = script("acquire_lunar_instance_nontrain", root)
        paths = [item["path"] for item in cache.files(repo, revision)]
        coco = cache.coco(repo, revision, converter.resolve_file(paths, f"instances_{split}.json"))
        info = next(i for i in coco["images"] if i["file_name"] == source_id)
        source = cache.asset(repo, revision, converter.resolve_image_file(paths, split, source_id))
        with Image.open(source) as opened:
            image = opened.convert("L")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        classes = {c["id"]: converter.CLASS_ID[c["name"].lower()] for c in coco["categories"]}
        for ann in coco["annotations"]:
            if ann["image_id"] == info["id"]:
                for polygon in ann["segmentation"]:
                    if len(polygon) >= 6 and len(polygon) % 2 == 0:
                        draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=classes[ann["category_id"]])
        return {"image": save_image(image), "mask": save_image(mask)}
    builder = script("build_orbital_300", root)
    if repo == "MarsLS/MMLSv2":
        # Reuse the established seven-band reader and visible-band renderer.
        token = re.sub(r"^mars_mmlsv2_(train|val|test)_", "", source_id)
        item = cache.find(repo, revision, lambda p: token in Path(p).stem and "mask" not in p.lower() and p.endswith((".tif", ".tiff")), folder=split)
        source = cache.asset(repo, revision, item["path"])
        converter = script("acquire_mmlsv2_subset", root)
        cube = converter.read_multiband(source)
        height = cube[:, :, 3].astype(np.float32, copy=False)
        height_path, valid_path = staging / "height.npy", staging / "valid.npy"
        np.save(height_path, height, allow_pickle=False)
        np.save(valid_path, np.isfinite(height).astype(np.uint8), allow_pickle=False)
        image_path, _, _ = builder.put_image(staging, source)
        return {"image": (Path(root) / image_path).read_bytes(), "height_map": height_path.read_bytes(), "height_valid_mask": valid_path.read_bytes()}
    if repo == "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection":
        converter = script("prepare_wac_orbital_pairs", root)
        coco = cache.coco(repo, revision, f"{split}.json")
        if source_id.isdigit():
            info = next(i for i in coco["images"] if str(i["id"]) == source_id)
        else:
            stem = re.sub(r"^moon_wac_(train|val|test)_", "", source_id)
            info = next(i for i in coco["images"] if Path(i["file_name"]).stem == stem)
        stem = Path(info["file_name"]).stem
        source = cache.asset(repo, revision, f"images_tiff/{stem}.tif")
        image_path, _, _ = builder.put_image(staging, source)
        result = {"image": (Path(root) / image_path).read_bytes()}
        if row.get("mask"):
            mask = Image.new("L", (info["width"], info["height"]), 0)
            draw = ImageDraw.Draw(mask)
            for ann in coco["annotations"]:
                if ann["image_id"] == info["id"]:
                    for polygon in ann.get("segmentation", []):
                        if len(polygon) >= 6:
                            draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=1)
            result["mask"] = save_image(mask)
        else:
            meta = next(r for r in cache.table(repo, revision, "metadata.parquet") if Path(r["WAC_VIS_TILE"]).stem == stem)
            url = f"{converter.AWS}/{stem}.nc"
            dtm = staging / "source_dtm.nc"
            with urlopen(Request(url, method="HEAD"), timeout=45) as response:
                size = int(response.headers["Content-Length"])
            if not 0 < size <= 1024**3:
                raise ValueError("DTM asset exceeds 1 GiB limit")
            download(url, dtm, expected_size=size)
            height, valid, _ = converter.resample_dtm_to_wac_grid(dtm, meta, (info["height"], info["width"]))
            height_path, valid_path = staging / "height.npy", staging / "valid.npy"
            np.save(height_path, height, allow_pickle=False)
            np.save(valid_path, valid, allow_pickle=False)
            result.update(height_map=height_path.read_bytes(), height_valid_mask=valid_path.read_bytes())
        return result
    raise ValueError(f"No orbital restore converter for {repo}")


@contextmanager
def planet_lock(root, planet):
    """OS lock serializes restorations, including requests from another API process."""
    folder = Path(root) / ".state/builtin-datasets"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{planet}.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        if __import__("os").name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if __import__("os").name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def download_dataset(root, planet, subset, progress=lambda done, total: None):
    """Explicit user-selected restoration; do not call from catalog reads/startup."""
    import hashlib
    import os
    import tempfile
    validate_selection(planet, subset)
    root = Path(root).resolve()
    with planet_lock(root, planet):
        info = inspect_subset(root, planet, subset)
        # Keep any local manifest and its original source metadata.
        rows = info["rows"]
        cache = SourceCache(root)
        cache.cache.mkdir(parents=True, exist_ok=True)
        progress(0, len(rows))
        for index, row in enumerate(rows, 1):
            needed = {key: checked_path(root, row[key]) for key in ASSETS if row.get(key)}
            if any(not path.is_file() for path in needed.values()):
                with tempfile.TemporaryDirectory(prefix="restore-", dir=cache.cache) as temp:
                    restored = materialize(row, cache, root, Path(temp))
                    for key, path in needed.items():
                        if path.is_file():
                            continue
                        payload = restored[key]
                        expected = row.get(key + "_sha256") or (row.get("height_sha256") if key == "height_map" else None)
                        if not expected or hashlib.sha256(payload).hexdigest() != expected:
                            raise ValueError(f"Restored {row['sample_id']}/{key} does not match its recorded SHA-256")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        partial = path.with_name(path.name + ".restore")
                        partial.write_bytes(payload)
                        os.replace(partial, path)
            progress(index, len(rows))
        if not info["manifest"].exists():
            # IMP is a subset of the original segmentation manifest; retain all
            # references so restoring another lunar subset later remains possible.
            track = "segmentation" if subset == "imp" else subset
            original = recovery_rows(planet, track)
            info["manifest"].parent.mkdir(parents=True, exist_ok=True)
            temporary = info["manifest"].with_suffix(".restore")
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in original), encoding="utf-8")
            os.replace(temporary, info["manifest"])
        if not inspect_subset(root, planet, subset)["ready"]:
            raise ValueError("Restoration did not produce a complete selected subset")
        verify_assets(root, rows)
