"""Download pinned orbital benchmark sources without changing rover data.

Run explicitly: python scripts/acquire_orbital.py
Only source files and provenance are stored here; no model weights are fetched.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1] / "data_orbital"
SOURCES = (
    {
        "name": "mars_crater_binary_seg",
        "repo": "Mirali33/mb-crater_binary_seg",
        "revision": "127ebfb30f2ec68781534a3071093a234ec90257",
        "license": "CC-BY-4.0 (dataset card)",
        "planet": "mars",
        "include": lambda p: p in {"README.md", "mapping.json", "metadata.zip"}
        or p in {"data/val-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"},
    },
    {
        "name": "mars_conequest_seg",
        "repo": "Mirali33/mb-conequest_seg",
        "revision": "4f2b15c07dcaab4f9b277a0744c28dae14a9c323",
        "license": "CC-BY-4.0 (dataset card; verify original ConeQuest terms)",
        "planet": "mars",
        "include": lambda p: p in {"README.md", "mapping.json", "metadata.csv"}
        or p in {"data/val-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"},
    },
    {
        "name": "moon_imp_seg",
        "repo": "nasa-ibm-ai4science/Sombench-IMP-Segmentation",
        "revision": "1418851011f4b1d4fd8140a8d765b17fda86ebc0",
        "license": "CC-BY-4.0 (dataset card)",
        "planet": "moon",
        "include": lambda p: p != ".gitattributes",
    },
    {
        "name": "moon_nac_crater_det",
        "repo": "nasa-ibm-ai4science/Sombench-NAC-Crater-Detection",
        "revision": "79419eb3486b2dd9ad09a283802e0bb8d75c767d",
        "license": "CC-BY-4.0 (dataset card)",
        "planet": "moon",
        "include": lambda p: p != ".gitattributes" and not p.startswith("previews/"),
    },
    {
        "name": "moon_wac_crater_det",
        "repo": "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection",
        "revision": "20f800becb64a0c7ad314652a683f97f0281fe67",
        "license": "CC-BY-4.0 (dataset card)",
        "planet": "moon",
        "wac_selected_splits": True,
    },
)


def fetch_json(url):
    request = Request(url, headers={"User-Agent": "planetary-vlm-orbital-acquisition/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download_one(source, item):
    relative = item["path"]
    destination = ROOT / "raw" / source["name"] / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = item["size"]
    expected_sha = item.get("lfs", {}).get("oid")
    if destination.is_file() and destination.stat().st_size == expected_size:
        local_sha = digest(destination)
        if expected_sha is None or local_sha == expected_sha:
            return {"path": relative, "size": expected_size, "sha256": local_sha, "status": "reused"}
    url = ("https://huggingface.co/datasets/" + source["repo"] + "/resolve/" +
           source["revision"] + "/" + quote(relative, safe="/") + "?download=true")
    partial = destination.with_name(destination.name + ".part")
    last_error = None
    for attempt in range(4):
        try:
            subprocess.run(
                ["curl.exe", "--location", "--fail", "--silent", "--show-error",
                 "--max-time", "300", "--output", str(partial), url],
                check=True, timeout=320,
            )
            if partial.stat().st_size != expected_size:
                raise ValueError(f"Size mismatch: {relative}")
            local_sha = digest(partial)
            if expected_sha and local_sha != expected_sha:
                raise ValueError(f"SHA-256 mismatch: {relative}")
            partial.replace(destination)
            return {"path": relative, "size": expected_size, "sha256": local_sha, "status": "downloaded"}
        except (HTTPError, URLError, TimeoutError, ValueError, OSError,
                subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {source['repo']}/{relative}: {last_error}")


def acquire(source):
    api = ("https://huggingface.co/api/datasets/" + source["repo"] +
           "/tree/" + source["revision"] +
           ("" if source.get("wac_selected_splits") else "?recursive=true&limit=1000"))
    items = fetch_json(api)
    if len(items) >= 1000:
        raise ValueError(f"Source listing may be paginated: {source['repo']}")
    if source.get("wac_selected_splits"):
        chosen = [item for item in items if item["type"] == "file" and
                  item["path"] in {"README.md", "metadata.parquet", "val.json", "test.json"}]
        for item in chosen:
            download_one(source, item)
        filenames = set()
        for split in ("val", "test"):
            coco = json.loads((ROOT / "raw" / source["name"] / f"{split}.json").read_text())
            filenames.update("images_tiff/" + Path(row["file_name"]).stem + ".tif"
                             for row in coco["images"])
        image_api = api + "/images_tiff?limit=1000"
        image_items = fetch_json(image_api)
        indexed = {item["path"]: item for item in image_items if item["type"] == "file"}
        missing = filenames - indexed.keys()
        if missing:
            raise ValueError(f"Missing WAC images in pinned repository: {sorted(missing)[:8]}")
        chosen.extend(indexed[name] for name in sorted(filenames))
    else:
        chosen = [item for item in items if item["type"] == "file" and source["include"](item["path"])]
    if not chosen:
        raise ValueError(f"No source files selected: {source['repo']}")
    completed = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(download_one, source, item): item for item in chosen}
        for future in as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda row: row["path"])
    manifest = {
        "schema_version": "orbital-acquisition-v1",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "planet": source["planet"],
        "repo": source["repo"],
        "revision": source["revision"],
        "source_url": "https://huggingface.co/datasets/" + source["repo"],
        "license_claim": source["license"],
        "selection": "Mars: official validation and test, plus metadata; Moon: full source except previews",
        "files": completed,
    }
    destination = ROOT / "raw" / source["name"] / "acquisition.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{source['name']}: {len(completed)} files, {sum(x['size'] for x in completed)} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-wac", action="store_true",
                        help="Also fetch optional WAC val/test images and COCO annotations")
    options = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        if source.get("wac_selected_splits") and not options.include_wac:
            continue
        acquire(source)
