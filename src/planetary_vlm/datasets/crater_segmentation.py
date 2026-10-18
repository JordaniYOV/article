"""Bounded, reproducible 50 Moon + 50 Mars crater-segmentation collection.

No models, training, inferred masks, split relabeling or protected archives.
Only publisher test examples are admitted. The original orbital collection and
the already-prepared training cohort are exclusion sources, never new inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from urllib.parse import quote
from urllib.request import Request

from planetary_vlm.network import urlopen
from .sources import download


MOON = "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection"
MOON_REV = "20f800becb64a0c7ad314652a683f97f0281fe67"
MARS = "Mirali33/mb-crater_binary_seg"
MARS_REV = "127ebfb30f2ec68781534a3071093a234ec90257"
MOON_COCO_SHA = "caf72eac333d62f06041949c3b4e4db67672b273a96694e2ac329b096bc82dff"
MARS_PARQUET = "data/test-00000-of-00001.parquet"
MARS_PARQUET_SHA = "c3e0d5f5407fac371d094b5f267875aad86f558e890974430a99216d14383172"
MARS_PARQUET_SIZE = 168512824
SCHEMA = "crater-segmentation-100-v1"
DEFAULT_OUTPUT = "data_orbital/crater_seg_100_v1"
NEAR_DUPLICATE_DISTANCE = 2
SEED = "crater-seg-100-v1-source-test"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def orbital_path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root / "data_orbital"):
        raise ValueError("Only data_orbital paths are allowed")
    return path


def collection_path(root, value):
    root = Path(root).resolve()
    path = orbital_path(root, value)
    forbidden = [root / "data_orbital" / name for name in
                 ("orbital_300_v3", "orbital_300_v4", "head_training_v2", "crater_source_annotations_v1")]
    if path == root / "data_orbital" or any(path.is_relative_to(p) or p.is_relative_to(path) for p in forbidden):
        raise ValueError("Use a new dedicated collection, never an original dataset folder")
    return path


def atomic_json(path, value, *, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in value)
            if lines else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Immutable collection file changed: {path.name}")
        return
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def source_stem(value):
    name = PurePosixPath(str(value).replace("\\", "/")).stem
    return re.sub(r"^moon_wac_(?:train|val|test)_", "", name).casefold()


def moon_group(stem):
    match = re.fullmatch(r"(M\d+[A-Z]+)_r\d+_c\d+", stem)
    if not match:
        raise ValueError("Unrecognized WAC product/tile identifier")
    return match[1]


def mars_group(stem):
    match = re.fullmatch(r"(thm_dir_N-?\d+_\d+)_\d+", stem)
    if not match:
        raise ValueError("Unrecognized Mars-Bench mosaic/tile identifier")
    return match[1]


def image_signature(image):
    """Exact decoded RGB hash + fixed 64-bit difference hash (not a geography audit)."""
    import numpy as np
    from PIL import Image
    rgb = image.convert("RGB")
    exact = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = small[:, 1:] > small[:, :-1]
    dhash = sum(int(value) << i for i, value in enumerate(bits.flat))
    return exact, f"{dhash:016x}"


class Exclusions:
    def __init__(self):
        self.ids = set()
        self.groups = set()
        self.files = set()
        self.pixels = set()
        self.dhashes = defaultdict(list)
        self.provenance = []

    def add(self, planet, source_id, group, file_hash, signature):
        self.ids.add((planet, source_stem(source_id)))
        if group:
            self.groups.add((planet, str(group)))
        self.files.add(file_hash)
        self.pixels.add(signature[0])
        self.dhashes[planet].append(int(signature[1], 16))

    def reason(self, planet, source_id, group, image_hash=None, signature=None, *, check_groups=True):
        if (planet, source_stem(source_id)) in self.ids:
            return "source_id_overlap"
        if check_groups and (planet, str(group)) in self.groups:
            return "scene_group_overlap"
        if image_hash in self.files:
            return "file_hash_overlap"
        if signature:
            if signature[0] in self.pixels:
                return "decoded_pixels_overlap"
            value = int(signature[1], 16)
            if any((value ^ old).bit_count() <= NEAR_DUPLICATE_DISTANCE for old in self.dhashes[planet]):
                return "near_duplicate_dhash"
        return None


def load_exclusions(root):
    """Read only explicitly named orbital manifests/assets; no broad directory walks."""
    from PIL import Image
    root = Path(root).resolve()
    result = Exclusions()
    for collection in ("orbital_300_v3", "orbital_300_v4"):
        for planet in ("moon", "mars"):
            for track in ("segmentation", "height", "train"):
                relative = f"data_orbital/{collection}/{planet}/{track}_manifest.jsonl"
                manifest = orbital_path(root, relative)
                if not manifest.exists():
                    if collection == "orbital_300_v3":
                        raise ValueError("All six orbital_300_v3 exclusion manifests are required")
                    continue
                source_rows = rows(manifest)
                result.provenance.append({"path": relative, "sha256": sha(manifest), "samples": len(source_rows)})
                for row in source_rows:
                    path = orbital_path(root, row["image"])
                    digest = sha(path)
                    if digest != row["image_sha256"]:
                        raise ValueError("Excluded source image changed: " + row["sample_id"])
                    with Image.open(path) as image:
                        signature = image_signature(image)
                    result.add(planet, row["source_image_id"], row.get("scene_group_id"), digest, signature)
    # Guard against any existing head-training selection outside the original 300.
    training = orbital_path(root, "data_orbital/head_training_v2/inputs.jsonl")
    if training.exists():
        result.provenance.append({"path": training.relative_to(root).as_posix(), "sha256": sha(training), "samples": len(rows(training))})
        for row in rows(training):
            path = orbital_path(root, row["rgb_image"])
            digest = sha(path)
            if digest != row["rgb_image_sha256"]:
                raise ValueError("Existing head input changed")
            with Image.open(path) as image:
                signature = image_signature(image)
            result.add("moon", row["source_image_id"], row["scene_group_id"], digest, signature)
    return result


def source_url(repo, revision, name, *, raw=False):
    return f"https://huggingface.co/datasets/{repo}/{'raw' if raw else 'resolve'}/{revision}/{quote(name, safe='/')}"


def get_json(url, max_bytes=8 * 1024**2):
    with urlopen(Request(url, headers={"User-Agent": "planetary-crater-collection/1"}), timeout=45) as response:
        content = response.read(max_bytes + 1)
        if len(content) > max_bytes or 'rel="next"' in response.headers.get("Link", ""):
            raise ValueError("Source inventory exceeds bounds or needs pagination")
        return json.loads(content)


def get_asset(repo, revision, name, path, size, digest, allow_download, cached=None):
    if not path.exists() and cached is not None and cached.is_file():
        if cached.stat().st_size != size or sha(cached) != digest:
            raise ValueError("Reusable source cache differs: " + cached.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, path)
    if path.exists():
        if path.stat().st_size != size or sha(path) != digest:
            raise ValueError("Existing source asset differs: " + path.name)
    elif not allow_download:
        raise ValueError("Missing source asset; use --allow-download explicitly")
    else:
        download(source_url(repo, revision, name), path, size, "sha256:" + digest)
    return path


def inventory(root, output, allow_download, source_cache=None):
    sources = output / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    if source_cache:
        for name in ("moon_tiff_inventory.json", "moon_README.md", "mars_README.md"):
            cached = source_cache / name
            if not (sources / name).exists() and cached.is_file():
                shutil.copyfile(cached, sources / name)
    coco_path = sources / "moon_test.json"
    if not coco_path.exists():
        existing = orbital_path(root, "data_orbital/crater_source_annotations_v1/moon_wac_test.json")
        if existing.exists() and sha(existing) == MOON_COCO_SHA:
            shutil.copyfile(existing, coco_path)
    get_asset(MOON, MOON_REV, "test.json", coco_path, 14159333, MOON_COCO_SHA, allow_download,
              source_cache / "moon_test.json" if source_cache else None)
    listing_path = sources / "moon_tiff_inventory.json"
    if not listing_path.exists():
        if not allow_download:
            raise ValueError("Missing pinned source inventory; use --allow-download")
        listing = get_json(f"https://huggingface.co/api/datasets/{MOON}/tree/{MOON_REV}/images_tiff?limit=1000")
        atomic_json(listing_path, listing)
    listing = json.loads(listing_path.read_text(encoding="utf-8"))
    if len(listing) != 1000 or any(r.get("type") != "file" for r in listing):
        raise ValueError("Unexpected WAC source inventory")
    for name, repo, revision in (("moon", MOON, MOON_REV), ("mars", MARS, MARS_REV)):
        card = sources / (name + "_README.md")
        if not card.exists():
            if not allow_download:
                raise ValueError("Missing pinned dataset card")
            with urlopen(Request(source_url(repo, revision, "README.md", raw=True)), timeout=45) as response:
                payload = response.read(256 * 1024 + 1)
            if len(payload) > 256 * 1024:
                raise ValueError("Source card exceeds size limit")
            card.write_bytes(payload)
        if "cc-by-4.0" not in card.read_text(encoding="utf-8").casefold():
            raise ValueError("Expected source license claim missing")
    parquet = get_asset(MARS, MARS_REV, MARS_PARQUET, sources / "mars_test.parquet",
                        MARS_PARQUET_SIZE, MARS_PARQUET_SHA, allow_download,
                        source_cache / "mars_test.parquet" if source_cache else None)
    return json.loads(coco_path.read_text(encoding="utf-8")), {r["path"]: r for r in listing}, parquet


def stable_order(source_id):
    return hashlib.sha256((SEED + ":" + source_id).encode()).hexdigest()


def spread(candidates):
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["scene_group_id"]].append(candidate)
    for group in grouped.values():
        group.sort(key=lambda r: stable_order(r["source_image_id"]))
    ordered_groups = sorted(grouped, key=stable_order)
    for offset in range(max((len(v) for v in grouped.values()), default=0)):
        for key in ordered_groups:
            if offset < len(grouped[key]):
                yield grouped[key][offset]


def save_png(path, image):
    content = BytesIO()
    image.save(content, format="PNG")
    payload = content.getvalue()
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("Existing derived image differs: " + path.name)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return sha(path)


def render_wac(cube):
    """Same visible-band rendering as the original collection; native TIFF is retained."""
    import numpy as np
    if cube.shape != (512, 512, 5) or np.isinf(cube).any():
        raise ValueError("Expected native 512x512x5 WAC VIS raster")
    channels = []
    limits = []
    for band in range(3):
        values = np.asarray(cube[..., band], dtype=np.float32)
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError("Empty WAC band")
        low, high = np.percentile(values[finite], (2, 98))
        scale = np.clip((values - low) / max(float(high - low), 1e-8), 0, 1)
        channels.append((np.nan_to_num(scale) * 255).astype(np.uint8))
        limits.append([float(low), float(high)])
    return np.stack(channels, axis=-1), {"method": "per_tile_visible_rgb_percentile_v1", "bands_zero_based": [0, 1, 2], "percentiles": [2, 98], "limits": limits}


def rasterize_craters(coco, info):
    from PIL import Image, ImageDraw
    categories = {r["id"]: r["name"].casefold() for r in coco["categories"]}
    mask = Image.new("L", (info["width"], info["height"]), 0)
    draw = ImageDraw.Draw(mask)
    annotations = []
    for ann in coco["annotations"]:
        if ann["image_id"] != info["id"]:
            continue
        if categories.get(ann["category_id"]) != "crater" or ann.get("iscrowd", 0):
            raise ValueError("Only individual publisher crater polygons are admitted")
        polygons = ann.get("segmentation")
        if not isinstance(polygons, list):
            raise ValueError("Unsupported COCO segmentation")
        for polygon in polygons:
            if len(polygon) < 6 or len(polygon) % 2:
                raise ValueError("Malformed publisher polygon")
            draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=1)
        annotations.append(ann)
    return mask, annotations


def validate_pair(image, mask):
    import numpy as np
    if image.size != mask.size or min(image.size) < 32:
        raise ValueError("Image/mask dimensions differ or are too small")
    values = np.asarray(mask)
    if values.ndim != 2 or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected a binary 0/1 crater mask; no relabeling is implicit")
    return {"size": list(image.size), "foreground_pixels": int(values.sum()),
            "foreground_fraction": float(values.mean()), "mask_values": [int(v) for v in np.unique(values)]}


def build_collection(root, output=DEFAULT_OUTPUT, count=50, allow_download=False, source_cache=None):
    """Acquire only held-out test samples, with restart-safe immutable outputs."""
    import numpy as np
    import pyarrow.parquet as pq
    import tifffile
    from PIL import Image
    root = Path(root).resolve()
    source_cache = orbital_path(root, source_cache) if source_cache else None
    destination = collection_path(root, output)
    if not 1 <= count <= 50:
        raise ValueError("Bounded collection supports 1..50 images per planet")
    if (destination / "metadata.json").exists():
        verified = verify_collection(root, output)
        if verified["counts"] != {"moon": count, "mars": count}:
            raise ValueError("Existing collection has another requested count")
        return verified
    exclusions = load_exclusions(root)
    identity = {"schema": SCHEMA, "count_per_planet": count, "source_split": "test",
                "source_revisions": {MOON: MOON_REV, MARS: MARS_REV},
                "exclusions": exclusions.provenance, "converter_sha256": sha(__file__),
                "dependencies": {name: version(name) for name in ("numpy", "Pillow", "pyarrow", "tifffile")},
                "source_group_policy": {"moon": "exclude all original products", "mars": "retain and report shared parent mosaics; exclude original tiles and duplicate pixels"},
                "seed": SEED, "near_duplicate_dhash_max_distance": NEAR_DUPLICATE_DISTANCE}
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "identity.json", identity)
    coco, tiff_inventory, parquet = inventory(root, destination, allow_download, source_cache)
    selected = {"moon": [], "mars": []}
    accepted = Exclusions()
    rejection_counts = Counter()
    rejection_rows = []

    def reject(planet, candidate, reason):
        rejection_counts[planet + ":" + reason] += 1
        rejection_rows.append({"planet": planet, "source_image_id": candidate["source_image_id"], "reason": reason})

    def accept(planet, candidate, image, mask, extra):
        digest, dhash = image_signature(image)
        reason = exclusions.reason(planet, candidate["source_image_id"], candidate["scene_group_id"],
                                   signature=(digest, dhash), check_groups=planet == "moon")
        reason = reason or accepted.reason(planet, candidate["source_image_id"], "", signature=(digest, dhash))
        if reason:
            reject(planet, candidate, reason)
            return False
        qa = validate_pair(image, mask)
        sid = f"{planet}_crater_{len(selected[planet]) + 1:03d}"
        folder = destination / planet / f"{len(selected[planet]) + 1:03d}"
        image_path, mask_path = folder / "image.png", folder / "mask.png"
        image_sha, mask_sha = save_png(image_path, image), save_png(mask_path, mask)
        record = {"sample_id": sid, "planet": planet, "track": "crater_segmentation",
                  "source_dataset": MOON if planet == "moon" else MARS,
                  "source_revision": MOON_REV if planet == "moon" else MARS_REV,
                  "source_split": "test", "evaluation_eligible": True, "final_test_eligible": True,
                  "source_image_id": candidate["source_image_id"], "scene_group_id": candidate["scene_group_id"],
                  "image": image_path.relative_to(root).as_posix(), "image_sha256": image_sha,
                  "mask": mask_path.relative_to(root).as_posix(), "mask_sha256": mask_sha,
                  "rgb_pixels_sha256": digest, "dhash64": dhash, "classes": ["Background", "Crater"],
                  "license_claim": "CC-BY-4.0 (pinned dataset card)", "reference_status": "ready",
                  "reference_scope": "publisher annotations, not guaranteed exhaustive visible-crater census",
                  "mask_semantics": "catalog_crater_polygon" if planet == "moon" else "publisher_binary_crater_segmentation",
                  "paired_ibm_status": "native_vis5_available_not_registered_for_existing_heads" if planet == "moon" else "blocked_missing_native_vis5_requires_separate_input_protocol",
                  "qa": qa, **extra}
        selected[planet].append(record)
        accepted.add(planet, candidate["source_image_id"], candidate["scene_group_id"], image_sha, (digest, dhash))
        print(f"{planet}: {len(selected[planet])}/{count} {candidate['source_image_id']}", flush=True)
        return True

    candidates = []
    for info in coco["images"]:
        stem = PurePosixPath(info["file_name"]).stem
        candidate = {"source_image_id": stem, "scene_group_id": moon_group(stem), "info": info}
        reason = exclusions.reason("moon", stem, candidate["scene_group_id"])
        if reason:
            reject("moon", candidate, reason)
        else:
            candidates.append(candidate)
    used_products = set()
    for candidate in spread(candidates):
        if len(selected["moon"]) >= count:
            break
        if candidate["scene_group_id"] in used_products:
            reject("moon", candidate, "one_tile_per_new_product")
            continue
        stem = candidate["source_image_id"]
        name = f"images_tiff/{stem}.tif"
        item = tiff_inventory.get(name)
        if not item or not item.get("lfs", {}).get("oid"):
            raise ValueError("Native WAC tile not in verified inventory")
        native = destination / "sources" / "moon_native" / (stem + ".tif")
        get_asset(MOON, MOON_REV, name, native, item["size"], item["lfs"]["oid"], allow_download,
                  source_cache / "moon_native" / (stem + ".tif") if source_cache else None)
        cube = tifffile.imread(native)
        rgb, transform = render_wac(cube)
        mask, annotations = rasterize_craters(coco, candidate["info"])
        with Image.fromarray(rgb) as image:
            if accept("moon", candidate, image, mask, {
                "native_vis": native.relative_to(root).as_posix(), "native_vis_sha256": sha(native),
                "native_vis_shape": list(cube.shape), "native_vis_finite_fraction": float(np.isfinite(cube).mean()),
                "display_transform": transform, "source_asset_url": source_url(MOON, MOON_REV, name),
                "source_annotation_file": "sources/moon_test.json", "source_annotation_sha256": MOON_COCO_SHA,
                "source_annotation_ids": [a["id"] for a in annotations], "source_coco_image_id": candidate["info"]["id"],
                "mask_conversion": "Pillow ImageDraw polygon fill=1, native grid; overlaps use union"}):
                used_products.add(candidate["scene_group_id"])
    if len(selected["moon"]) != count:
        raise ValueError(f"Only {len(selected['moon'])} new group-disjoint lunar images survived; no exclusions were relaxed")

    candidates = []
    table = pq.ParquetFile(parquet)
    if table.metadata.num_rows != 900:
        raise ValueError("Unexpected Mars publisher-test row count")
    index = 0
    for batch in table.iter_batches(batch_size=16):
        for record in batch.to_pylist():
            name = PurePosixPath(record["image"]["path"]).name
            candidate = {"source_image_id": name, "scene_group_id": mars_group(PurePosixPath(name).stem),
                         "parquet_row": index, "record": record}
            index += 1
            reason = exclusions.reason("mars", name, candidate["scene_group_id"], check_groups=False)
            if reason:
                reject("mars", candidate, reason)
            else:
                candidates.append(candidate)
    for candidate in spread(candidates):
        if len(selected["mars"]) >= count:
            break
        record = candidate["record"]
        with Image.open(BytesIO(record["image"]["bytes"])) as image, Image.open(BytesIO(record["mask"]["bytes"])) as mask:
            accept("mars", candidate, image, mask, {
                "source_asset_url": source_url(MARS, MARS_REV, MARS_PARQUET),
                "source_container_sha256": MARS_PARQUET_SHA, "source_parquet_row": candidate["parquet_row"],
                "source_image_bytes_sha256": hashlib.sha256(record["image"]["bytes"]).hexdigest(),
                "source_mask_bytes_sha256": hashlib.sha256(record["mask"]["bytes"]).hexdigest(),
                "display_transform": None, "mask_conversion": "publisher 0/1 labels unchanged, PNG serialization only"})
    if len(selected["mars"]) != count:
        raise ValueError(f"Only {len(selected['mars'])} new Mars images survived; no exclusions were relaxed")
    for planet in selected:
        atomic_json(destination / planet / "manifest.jsonl", selected[planet], lines=True)
    combined = selected["moon"] + selected["mars"]
    atomic_json(destination / "manifest.jsonl", combined, lines=True)
    atomic_json(destination / "rejections.jsonl", rejection_rows, lines=True)
    atomic_json(destination / "source_attribution.json", {
        "sources": [{"repo": repo, "revision": revision, "url": f"https://huggingface.co/datasets/{repo}", "license_claim": "CC-BY-4.0"}
                    for repo, revision in ((MOON, MOON_REV), (MARS, MARS_REV))],
        "cards": {name: sha(destination / "sources" / name) for name in ("moon_README.md", "mars_README.md")},
        "processing": "Visible-band rendering + publisher polygon rasterization for Moon; original binary labels for Mars.",
        "model_inference": False, "training": False, "weights_downloaded": False})
    bound_files = ("manifest.jsonl", "moon/manifest.jsonl", "mars/manifest.jsonl", "rejections.jsonl", "source_attribution.json", "identity.json",
                   "sources/moon_test.json", "sources/moon_tiff_inventory.json", "sources/moon_README.md", "sources/mars_README.md", "sources/mars_test.parquet")
    metadata = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(), "identity": identity,
                "counts": dict(Counter(r["planet"] for r in combined)), "source_splits": {"test": len(combined)},
                "scene_groups": {p: len({r["scene_group_id"] for r in selected[p]}) for p in selected},
                "shared_original_source_groups": {p: sorted({r["scene_group_id"] for r in selected[p] if (p, r["scene_group_id"]) in exclusions.groups}) for p in selected},
                "rejection_counts": dict(rejection_counts), "files_sha256": {name: sha(destination / name) for name in bound_files},
                "limitations": ["Original source tile IDs, decoded pixels and a fixed near-duplicate heuristic are excluded. Lunar products are also excluded, but all four Mars source-test parent mosaics are shared with orbital_300_v3.",
                    "New Mars tiles are not new independent regions; preserve mosaic IDs for clustered analysis. Geographic footprint overlap across different products is not ruled out.",
                    "Pretraining independence is not demonstrated.", "Publisher masks need not include every visually identifiable crater.",
                    "New collection is NOT registered in the existing WAC head deployment allowlist.",
                    "Mars does not supply native five-band WAC VIS; current IBM frozen-head inference is unsupported on Mars.",
                    "All sources retain original test split. Never tune thresholds or transformations on these new test images."]}
    atomic_json(destination / "metadata.json", metadata)
    write_previews(destination, combined, root)
    write_readme(destination, metadata)
    return verify_collection(root, output)


def write_previews(destination, records, root):
    from PIL import Image, ImageDraw
    for planet in ("moon", "mars"):
        subset = [r for r in records if r["planet"] == planet]
        width, height = 160, 180
        canvas = Image.new("RGB", (width * 5, height * ((len(subset) + 4) // 5)), "#161a20")
        draw = ImageDraw.Draw(canvas)
        for index, record in enumerate(subset):
            with Image.open(orbital_path(root, record["image"])) as source, Image.open(orbital_path(root, record["mask"])) as mask:
                thumb = source.convert("RGB").resize((width, 160))
                selected = mask.point(lambda x: 100 if x else 0).resize((width, 160))
                thumb.paste(Image.new("RGB", thumb.size, "#12d9de"), (0, 0), selected)
            x, y = index % 5 * width, index // 5 * height
            canvas.paste(thumb, (x, y))
            draw.text((x + 4, y + 162), record["sample_id"], fill="white")
        save_png(destination / "previews" / (planet + "_masks.png"), canvas)


def write_readme(destination, metadata):
    text = f"""# Crater segmentation 100 v1

{metadata['counts']['moon']} Moon + {metadata['counts']['mars']} Mars image/mask pairs.
Binary mask IDs: 0 Background, 1 Crater. Original source-test splits preserved.
No original image IDs or duplicate decoded pixels from orbital_300_v3/v4.
Existing lunar head-training products are excluded. Mars retains four shared
parent mosaics, not shared selected tiles: these are NOT new independent regions.
Exclusion details and limitations are in metadata.json; all sample assets have SHA-256.

Moon: native five-band WAC TIFF retained; visible bands 0..2 rendered exactly as
in the earlier collection; publisher COCO polygons rasterized at native 512x512.
Mars: publisher binary crater masks, without relabeling other semantic classes.
These are source annotation targets, not a claim of exhaustive visible-crater truth.

No training, model inference or weight download was performed. Old datasets and
annotations were not modified. Current WAC head deployments admit a different
fixed input list: this collection needs a separate registration/protocol.
Mars lacks native WAC VIS5 and is NOT ready for the existing IBM head adapter.

Manifests: manifest.jsonl and moon/manifest.jsonl, mars/manifest.jsonl.
Each image folder has image.png + mask.png. Moon native TIFF paths are recorded
in the manifest, under sources/moon_native/. Source cards are retained in sources/.
The Mars test Parquet is a verified containing shard for these selected examples,
not an additional evaluation set or permission to use its other samples.

Mask overlays: previews/moon_masks.png, previews/mars_masks.png.
Restore/rebuild: python -m planetary_vlm crater-seg-build --allow-download
Verify without network/models: python -m planetary_vlm crater-seg-verify

Do not tune on these new source-test samples. More independent pretraining and
geographic-overlap audits remain necessary before broad generalization claims.
"""
    path = destination / "README.md"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("Existing collection README differs")
    path.write_text(text, encoding="utf-8")


def verify_collection(root, output=DEFAULT_OUTPUT):
    from PIL import Image
    import tifffile
    import numpy as np
    root = Path(root).resolve()
    destination = collection_path(root, output)
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Unknown collection schema")
    for name, digest in metadata["files_sha256"].items():
        path = (destination / name).resolve()
        if not path.is_relative_to(destination) or sha(path) != digest:
            raise ValueError("Collection metadata hash changed: " + name)
    if (sha(destination / "sources/moon_test.json") != MOON_COCO_SHA
            or sha(destination / "sources/mars_test.parquet") != MARS_PARQUET_SHA):
        raise ValueError("Source annotations/container differ from pinned publisher files")
    if json.loads((destination / "identity.json").read_text()) != metadata["identity"]:
        raise ValueError("Collection identities differ")
    exclusions = load_exclusions(root)
    if exclusions.provenance != metadata["identity"]["exclusions"]:
        raise ValueError("Original exclusion manifests changed")
    records = rows(destination / "manifest.jsonl")
    if len(records) != 2 * metadata["identity"]["count_per_planet"]:
        raise ValueError("Unexpected sample count")
    seen = Exclusions()
    lunar_coco = json.loads((destination / "sources/moon_test.json").read_text(encoding="utf-8"))
    lunar_images = {r["id"]: r for r in lunar_coco["images"]}
    lunar_inventory = {r["path"]: r for r in json.loads((destination / "sources/moon_tiff_inventory.json").read_text())}
    mars_records = {r["source_parquet_row"]: r for r in records if r["planet"] == "mars"}
    import pyarrow.parquet as pq
    source_index = 0
    for batch in pq.ParquetFile(destination / "sources/mars_test.parquet").iter_batches(batch_size=16):
        for source in batch.to_pylist():
            record = mars_records.get(source_index)
            source_index += 1
            if record is None:
                continue
            if (hashlib.sha256(source["image"]["bytes"]).hexdigest() != record["source_image_bytes_sha256"]
                    or hashlib.sha256(source["mask"]["bytes"]).hexdigest() != record["source_mask_bytes_sha256"]):
                raise ValueError("Selected Mars source bytes differ")
            with Image.open(BytesIO(source["image"]["bytes"])) as source_image, Image.open(BytesIO(source["mask"]["bytes"])) as source_mask:
                with Image.open(orbital_path(root, record["image"])) as image, Image.open(orbital_path(root, record["mask"])) as mask:
                    if (source_image.mode != image.mode or source_mask.mode != mask.mode or source_image.size != image.size
                            or source_mask.size != mask.size or source_image.tobytes() != image.tobytes() or source_mask.tobytes() != mask.tobytes()):
                        raise ValueError("Mars pixels/labels differ from publisher")
    ids = set()
    actual_counts = Counter()
    for record in records:
        if record["sample_id"] in ids or record["source_split"] != "test" or record["classes"] != ["Background", "Crater"]:
            raise ValueError("Duplicate ID or target/split substitution")
        ids.add(record["sample_id"])
        planet = record["planet"]
        if (record["source_dataset"], record["source_revision"]) != ((MOON, MOON_REV) if planet == "moon" else (MARS, MARS_REV)):
            raise ValueError("Unexpected source identity")
        for key in ("image", "mask"):
            if sha(orbital_path(root, record[key])) != record[key + "_sha256"]:
                raise ValueError("Sample asset changed: " + record["sample_id"])
        with Image.open(orbital_path(root, record["image"])) as image, Image.open(orbital_path(root, record["mask"])) as mask:
            if validate_pair(image, mask) != record["qa"]:
                raise ValueError("Pair QA differs")
            signature = image_signature(image)
        if signature != (record["rgb_pixels_sha256"], record["dhash64"]):
            raise ValueError("Decoded image identity changed")
        for excluded in (exclusions, seen):
            group = record["scene_group_id"] if excluded is exclusions else ""
            reason = excluded.reason(planet, record["source_image_id"], group, record["image_sha256"], signature,
                                     check_groups=excluded is exclusions and planet == "moon")
            if reason:
                raise ValueError("Collection overlap: " + record["sample_id"] + ":" + reason)
        seen.add(planet, record["source_image_id"], record["scene_group_id"], record["image_sha256"], signature)
        if planet == "moon":
            native = orbital_path(root, record["native_vis"])
            inventory_row = lunar_inventory["images_tiff/" + record["source_image_id"] + ".tif"]
            if sha(native) != record["native_vis_sha256"] or record["native_vis_sha256"] != inventory_row["lfs"]["oid"]:
                raise ValueError("Native WAC changed")
            cube = tifffile.imread(native)
            if cube.shape != (512, 512, 5) or np.isinf(cube).any():
                raise ValueError("Invalid native VIS input")
            rgb, transform = render_wac(cube)
            expected_mask, annotations = rasterize_craters(lunar_coco, lunar_images[record["source_coco_image_id"]])
            with Image.open(orbital_path(root, record["image"])) as image, Image.open(orbital_path(root, record["mask"])) as mask:
                if image.tobytes() != rgb.tobytes() or mask.tobytes() != expected_mask.tobytes() or transform != record["display_transform"]:
                    raise ValueError("WAC render or polygon mask differs from source")
            if [r["id"] for r in annotations] != record["source_annotation_ids"]:
                raise ValueError("WAC source annotation identity differs")
        actual_counts[planet] += 1
    if dict(actual_counts) != metadata["counts"]:
        raise ValueError("Planet counts differ")
    for planet in ("moon", "mars"):
        if rows(destination / planet / "manifest.jsonl") != [r for r in records if r["planet"] == planet]:
            raise ValueError("Per-planet manifest differs")
    return {"dataset": str(destination), "counts": dict(actual_counts), "source_split": "test",
            "original_image_overlap": 0, "shared_original_source_groups": metadata["shared_original_source_groups"],
            "mask_classes": ["Background", "Crater"],
            "moon_native_vis5": actual_counts["moon"], "mars_ibm_vis5_status": "unsupported_missing_channels",
            "model_inference": False, "training": False, "scientific_scores": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--source-cache", help="Optional verified prior sources directory inside data_orbital")
    args = parser.parse_args(argv)
    result = (build_collection(args.workspace, args.output, allow_download=args.allow_download, source_cache=args.source_cache)
              if args.command == "build" else verify_collection(args.workspace, args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
