"""Selective acquisition helpers for the LuSNAR synthetic lunar dataset.

The source stores each scene as a large ZIP. HTTP byte ranges let this module
inspect its central directory and extract a reproducible, spaced subset without
downloading unrelated trajectories.
"""

from collections import OrderedDict
import hashlib
from http.client import IncompleteRead
import io
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import zipfile
import zlib


class HTTPRangeReader:
    """Small seekable ZIP reader backed by HTTP Range requests."""

    def __init__(self, url, size, *, block_size=1024 * 1024, cache_blocks=8):
        if size <= 0 or block_size <= 0 or cache_blocks <= 0:
            raise ValueError("size, block_size, and cache_blocks must be positive")
        self.url = url
        self.origin_url = url
        self.size = size
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        self.position = 0
        self.cache = OrderedDict()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("invalid whence")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def _block(self, index):
        cached = self.cache.get(index)
        if cached is not None:
            self.cache.move_to_end(index)
            return cached
        start = index * self.block_size
        end = min(self.size, start + self.block_size) - 1
        expected = end - start + 1
        last_error = None
        for attempt in range(6):
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206:
                        raise OSError(f"Expected HTTP 206 for byte range; got {response.status}")
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-"):
                        raise OSError(f"Unexpected Content-Range: {content_range!r}")
                    payload = response.read()
                    final_url = response.geturl()
                if len(payload) != expected:
                    raise OSError(f"Incomplete HTTP range: received {len(payload)} of {expected} bytes")
                # Reuse the signed CDN URL after the first redirect. Refresh it
                # from the immutable Hub URL if it expires during a long run.
                self.url = final_url
                break
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code in {401, 403, 410, 429, 500, 502, 503, 504}:
                    self.url = self.origin_url
                else:
                    raise
            except (IncompleteRead, urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
                if self.url != self.origin_url:
                    self.url = self.origin_url
            if attempt < 5:
                time.sleep(min(0.5 * (attempt + 1), 2.0))
        else:
            raise OSError(f"Failed byte range {start}-{end} after retries") from last_error
        self.cache[index] = payload
        self.cache.move_to_end(index)
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return payload

    def read(self, size=-1):
        if size is None or size < 0:
            size = max(0, self.size - self.position)
        size = min(size, max(0, self.size - self.position))
        chunks = []
        remaining = size
        while remaining:
            index = self.position // self.block_size
            offset = self.position % self.block_size
            block = self._block(index)
            chunk = block[offset:offset + remaining]
            if not chunk:
                break
            chunks.append(chunk)
            self.position += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def matched_frames(archive, camera="image0"):
    """Return frame stems having RGB, depth and label members in one camera."""
    roots = {}
    marker = f"/{camera}/color/"
    for name in archive.namelist():
        normalized = "/" + name.replace("\\", "/").lstrip("/")
        position = normalized.find(marker)
        if position < 0 or not normalized.lower().endswith(".png"):
            continue
        prefix = normalized[:position]
        filename = normalized[position + len(marker):]
        if "/" in filename:
            continue
        stem = filename[:-4]
        rgb = f"{prefix}{marker}{filename}".lstrip("/")
        depth = f"{prefix}/{camera}/depth/{stem}.pfm".lstrip("/")
        label = f"{prefix}/{camera}/label/{filename}".lstrip("/")
        roots.setdefault(stem, {})["rgb"] = rgb
        roots[stem]["depth"] = depth
        roots[stem]["label"] = label
    available = set(archive.namelist())
    return [(stem, assets) for stem, assets in sorted(roots.items())
            if set(assets.values()).issubset(available)]


def evenly_spaced_indices(count, requested):
    """Select unique indices spanning the full ordered sequence."""
    if requested < 1 or count < requested:
        raise ValueError(f"Need at least {requested} complete frames; found {count}")
    if requested == 1:
        return [count // 2]
    indices = [round(index * (count - 1) / (requested - 1)) for index in range(requested)]
    if len(set(indices)) != requested:
        raise ValueError("Could not produce unique evenly spaced samples")
    return indices


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_timestamp(stem):
    match = re.search(r"\d+", stem)
    return int(match.group()) if match else stem


def decode_pfm(payload):
    """Decode LuSNAR's PFM payload into its original image row order.

    The source convention is verified by exact alignment between its 65,504
    depth sentinel and the source sky mask; generic PFM readers normally flip
    rows, which would break that pixel alignment for these files.
    """
    import numpy as np

    position = 0
    headers = []
    while len(headers) < 3:
        end = payload.find(b"\n", position)
        if end < 0:
            raise ValueError("Truncated PFM header")
        line = payload[position:end].strip()
        position = end + 1
        if line and not line.startswith(b"#"):
            headers.append(line)
    magic, dimensions, scale_line = headers
    if magic not in {b"PF", b"Pf"}:
        raise ValueError(f"Unsupported PFM magic: {magic!r}")
    width, height = (int(value) for value in dimensions.split())
    scale = float(scale_line)
    if width <= 0 or height <= 0 or scale == 0:
        raise ValueError("Invalid PFM dimensions or scale")
    dtype = np.dtype("<f4" if scale < 0 else ">f4")
    channels = 3 if magic == b"PF" else 1
    values = np.frombuffer(payload, dtype=dtype, offset=position)
    if values.size != width * height * channels:
        raise ValueError("PFM payload size does not match its header")
    values = values.reshape(height, width, channels) * abs(scale)
    result = values.astype(np.float32, copy=True)
    return result[..., 0] if channels == 1 else result


def collect_scene(archive_path, destination, *, scene_id, requested=None, frame_indices=None,
                  source_url, source_revision, source_archive_sha256, camera="image0"):
    """Extract a time-spaced RGB/depth/label subset from one local ZIP.

    Reading each member to EOF makes ``zipfile`` check its CRC before the
    resulting per-file SHA-256 is entered into the collection index.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    if isinstance(archive_path, zipfile.ZipFile):
        archive_context = archive_path
        close_archive = False
    else:
        archive_path = Path(archive_path)
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        archive_context = zipfile.ZipFile(archive_path)
        close_archive = True
    try:
        archive = archive_context
        frames = matched_frames(archive, camera=camera)
        if frame_indices is None:
            if requested is None:
                raise ValueError("Provide either requested or frame_indices")
            indices = evenly_spaced_indices(len(frames), requested)
        else:
            indices = list(frame_indices)
            if len(set(indices)) != len(indices) or any(
                    not isinstance(index, int) or index < 0 or index >= len(frames)
                    for index in indices):
                raise ValueError("frame_indices must be unique valid frame indices")
        for frame_index in indices:
            stem, assets = frames[frame_index]
            sample_id = f"lusnar_scene_{scene_id}_{frame_timestamp(stem)}"
            sample_dir = destination / f"scene_{scene_id:02d}" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            asset_rows = {}
            for modality, member_name in assets.items():
                info = archive.getinfo(member_name)
                target = sample_dir / f"{modality}{Path(member_name).suffix.lower()}"
                existing_valid = False
                if target.is_file() and target.stat().st_size == info.file_size:
                    crc = byte_count = 0
                    with target.open("rb") as existing:
                        for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                            byte_count += len(chunk)
                            crc = zlib.crc32(chunk, crc)
                    existing_valid = byte_count == info.file_size and crc & 0xffffffff == info.CRC
                    if existing_valid:
                        digest_hex = sha256_file(target)
                if not existing_valid:
                    temporary = target.with_name(target.name + ".part")
                    crc = byte_count = 0
                    digest = hashlib.sha256()
                    with archive.open(info, "r") as incoming, temporary.open("wb") as outgoing:
                        while True:
                            chunk = incoming.read(1024 * 1024)
                            if not chunk:
                                break
                            outgoing.write(chunk)
                            byte_count += len(chunk)
                            crc = zlib.crc32(chunk, crc)
                            digest.update(chunk)
                    if byte_count != info.file_size or crc & 0xffffffff != info.CRC:
                        raise OSError(f"ZIP CRC/size mismatch for {member_name}")
                    os.replace(temporary, target)
                    digest_hex = digest.hexdigest()
                asset_rows[modality] = {
                    "path": str(target.resolve()),
                    "source_member": member_name,
                    "source_crc32": f"{info.CRC:08x}",
                    "source_uncompressed_bytes": info.file_size,
                    "source_compressed_bytes": info.compress_size,
                    "sha256": digest_hex,
                }
            records.append({
                "sample_id": sample_id,
                "scene_group_id": f"lusnar_scene_{scene_id:02d}",
                "sequence_frame_index": frame_index,
                "sequence_frame_count": len(frames),
                "frame_stem": stem,
                "camera": camera,
                "split": LUSNAR_SCENE_SPLITS.get(scene_id, "unknown"),
                "source_url": source_url,
                "source_revision": source_revision,
                "source_archive_sha256": source_archive_sha256,
                "source_archive_sha256_verified": False,
                "depth_origin": "unreal_engine_simulator_ground_truth_native_encoding",
                "synthetic": True,
                "assets": asset_rows,
            })
    finally:
        if close_archive:
            archive_context.close()
    return records


def save_selection_index(records, path):
    """Write one portable JSONL inventory for selected frames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = path.parent.resolve()
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            portable = dict(record)
            portable["assets"] = {}
            for modality, asset in record["assets"].items():
                relative_asset = dict(asset)
                relative_asset["path"] = Path(asset["path"]).resolve().relative_to(root).as_posix()
                portable["assets"][modality] = relative_asset
            stream.write(json.dumps(portable, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def prepare_lusnar_inputs(selection_index, destination):
    """Convert verified raw triplets into scalar masks and numeric depth inputs."""
    import numpy as np
    from PIL import Image
    from planetary_vlm.io import write_json, write_jsonl

    selection_index, destination = Path(selection_index).resolve(), Path(destination).resolve()
    rows = [json.loads(line) for line in selection_index.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 152 or len({row["sample_id"] for row in rows}) != 152:
        raise ValueError("Expected 152 unique selected LuSNAR triplets")
    scene_ids = sorted({int(row["scene_group_id"].rsplit("_", 1)[1]) for row in rows})
    scene_counts = {scene: sum(row["scene_group_id"] == f"lusnar_scene_{scene:02d}" for row in rows)
                    for scene in scene_ids}
    has_selection_labels = ["rock_presence" in row for row in rows]
    if any(has_selection_labels) and not all(has_selection_labels):
        raise ValueError("Rock-presence selection metadata must be present for every frame or none")
    positive_count = sum(bool(row["rock_presence"]) for row in rows if "rock_presence" in row)
    balanced_selection = all(has_selection_labels)
    if balanced_selection and (positive_count != 76 or len(rows) - positive_count != 76):
        raise ValueError(f"Expected 76 frames with and 76 without rock masks; got {positive_count}")
    if destination.exists():
        raise FileExistsError(f"Choose a new version; destination already exists: {destination}")
    colour_to_id = {
        (187, 70, 156): 1,  # lunar regolith
        (120, 0, 200): 2,   # impact crater
        (232, 250, 80): 3,  # rock
        (173, 69, 31): 4,   # mountain
        (34, 201, 248): 5,  # sky
    }
    native = {(r << 16) | (g << 8) | b: label for (r, g, b), label in colour_to_id.items()}
    root = selection_index.parent
    # Preflight the full selection before creating any derived dataset files.
    for record in rows:
        paths = {name: (root / asset["path"]).resolve() if not Path(asset["path"]).is_absolute()
                 else Path(asset["path"]).resolve()
                 for name, asset in record["assets"].items()}
        for modality, path in paths.items():
            if sha256_file(path) != record["assets"][modality]["sha256"]:
                raise ValueError(f"Selected {modality} asset changed: {record['sample_id']}")
        with Image.open(paths["rgb"]) as source_image:
            image_size = source_image.size
        with Image.open(paths["label"]) as source_mask:
            mask_rgb = np.asarray(source_mask.convert("RGB"), dtype=np.uint8)
        if image_size != (1024, 1024) or mask_rgb.shape != (1024, 1024, 3):
            raise ValueError(f"Unexpected RGB/mask dimensions: {record['sample_id']}")
        encoded = (mask_rgb[:, :, 0].astype(np.uint32) << 16) | (
            mask_rgb[:, :, 1].astype(np.uint32) << 8) | mask_rgb[:, :, 2]
        unknown = set(int(value) for value in np.unique(encoded)) - set(native)
        if unknown:
            raise ValueError(f"Unknown LuSNAR semantic colours {sorted(unknown)} in {record['sample_id']}")
        rock_present = bool(np.any(encoded == ((232 << 16) | (250 << 8) | 80)))
        if balanced_selection and rock_present != bool(record["rock_presence"]):
            raise ValueError(f"Rock-presence selection does not match source mask: {record['sample_id']}")
        scalar_mask = np.zeros(encoded.shape, dtype=np.uint8)
        for colour_code, label_id in native.items():
            scalar_mask[encoded == colour_code] = label_id
        depth = decode_pfm(paths["depth"].read_bytes())
        if depth.shape != scalar_mask.shape:
            raise ValueError(f"RGB/mask/depth dimensions differ: {record['sample_id']}")
        sky = scalar_mask == colour_to_id[(34, 201, 248)]
        invalid_sentinel = depth == 65504.0
        if not np.array_equal(sky, invalid_sentinel):
            raise ValueError(f"PFM sentinel is not pixel-aligned to the sky label: {record['sample_id']}")
        if float((np.isfinite(depth) & (depth > 0) & ~invalid_sentinel).mean()) < 0.5:
            raise ValueError(f"Insufficient valid depth coverage: {record['sample_id']}")

    destination.mkdir(parents=True, exist_ok=False)
    for folder in ("images", "masks", "depth", "validity"):
        (destination / folder).mkdir()

    samples, depths, qa_rows = [], [], []
    for record in rows:
        asset_paths = {name: (root / asset["path"]).resolve() if not Path(asset["path"]).is_absolute()
                       else Path(asset["path"]).resolve()
                       for name, asset in record["assets"].items()}
        for modality, path in asset_paths.items():
            if sha256_file(path) != record["assets"][modality]["sha256"]:
                raise ValueError(f"Selected {modality} asset changed: {record['sample_id']}")
        sample_id = record["sample_id"]
        image_out = destination / "images" / f"{sample_id}.png"
        mask_out = destination / "masks" / f"{sample_id}.png"
        depth_out = destination / "depth" / f"{sample_id}.source_values.npy"
        valid_out = destination / "validity" / f"{sample_id}.validity.npy"
        with Image.open(asset_paths["rgb"]) as source_image:
            image = source_image.convert("RGB")
        with Image.open(asset_paths["label"]) as source_mask:
            mask_rgb = np.asarray(source_mask.convert("RGB"), dtype=np.uint8)
        if image.size != (1024, 1024) or mask_rgb.shape != (1024, 1024, 3):
            raise ValueError(f"Unexpected RGB/mask dimensions: {sample_id}")
        encoded = (mask_rgb[:, :, 0].astype(np.uint32) << 16) | (
            mask_rgb[:, :, 1].astype(np.uint32) << 8) | mask_rgb[:, :, 2]
        unknown = set(int(value) for value in np.unique(encoded)) - set(native)
        if unknown:
            raise ValueError(f"Unknown LuSNAR semantic colours {sorted(unknown)} in {sample_id}")
        rock_present = bool(np.any(encoded == ((232 << 16) | (250 << 8) | 80)))
        if balanced_selection and rock_present != bool(record["rock_presence"]):
            raise ValueError(f"Rock-presence selection does not match source mask: {sample_id}")
        scalar_mask = np.zeros(encoded.shape, dtype=np.uint8)
        for colour_code, label_id in native.items():
            scalar_mask[encoded == colour_code] = label_id
        depth = decode_pfm(asset_paths["depth"].read_bytes())
        if depth.shape != scalar_mask.shape:
            raise ValueError(f"RGB/mask/depth dimensions differ: {sample_id}")
        sky = scalar_mask == colour_to_id[(34, 201, 248)]
        invalid_sentinel = depth == 65504.0
        if not np.array_equal(sky, invalid_sentinel):
            raise ValueError(f"PFM sentinel is not pixel-aligned to the sky label: {sample_id}")
        valid = np.isfinite(depth) & (depth > 0) & ~invalid_sentinel
        valid_fraction = float(valid.mean())
        if valid_fraction < 0.5:
            raise ValueError(f"Insufficient valid depth coverage ({valid_fraction:.3f}): {sample_id}")
        image.save(image_out, format="PNG")
        Image.fromarray(scalar_mask, mode="L").save(mask_out, format="PNG")
        np.save(depth_out, depth.astype(np.float32, copy=False), allow_pickle=False)
        np.save(valid_out, valid, allow_pickle=False)
        image_hash = sha256_file(image_out)
        depth_hash, validity_hash = sha256_file(depth_out), sha256_file(valid_out)
        samples.append({"sample_id": sample_id,
            "image_path": image_out.relative_to(destination).as_posix(),
            "mask_path": mask_out.relative_to(destination).as_posix(),
            "group_id": record["scene_group_id"], "split": record["split"],
            "license": "MIT (publisher claim; publication review pending)",
            "source_url": "https://huggingface.co/datasets/xuboluo2001/LuSNAR"})
        depths.append({"sample_id": sample_id, "status": "technical_depth_qa_pass",
            "image_path": str(image_out.resolve()),
            "registration": {"benchmark_sha256": image_hash},
            "depth_path": str(depth_out.resolve()), "depth_sha256": depth_hash,
            "validity_path": str(valid_out.resolve()), "validity_sha256": validity_hash,
            "depth_origin": record["depth_origin"], "depth_units": "unspecified_simulator_native_values",
            "source_depth_sha256": record["assets"]["depth"]["sha256"],
            "source_depth_member": record["assets"]["depth"]["source_member"],
            "source_label_sha256": record["assets"]["label"]["sha256"],
            "source_label_member": record["assets"]["label"]["source_member"],
            "source_rgb_sha256": record["assets"]["rgb"]["sha256"],
            "source_rgb_member": record["assets"]["rgb"]["source_member"],
            "source_archive_sha256": record["source_archive_sha256"],
            "source_archive_sha256_verified": False,
            "scene_group_id": record["scene_group_id"],
            "sequence_frame_index": record["sequence_frame_index"],
            "sequence_frame_count": record["sequence_frame_count"],
            "valid_fraction": valid_fraction, "invalid_sentinel": 65504.0})
        qa_rows.append({"sample_id": sample_id, "scene_group_id": record["scene_group_id"],
            "image_size": list(image.size), "mask_values": sorted(int(x) for x in np.unique(scalar_mask)),
            "valid_depth_fraction": valid_fraction, "sky_sentinel_pixel_exact_match": True})

    write_jsonl(destination / "samples.jsonl", samples)
    write_jsonl(destination / "depth_index.jsonl", depths)
    write_json(destination / "source_validation.json", {
        "source": "LuSNAR", "revision": LUSNAR_REVISION, "synthetic": True,
        "selected_triplets": len(rows), "scene_counts": scene_counts,
        "rock_presence_definition": "at least one source pixel with the exact ROCK semantic colour",
        "rock_presence_counts": ({"present": positive_count, "absent": len(rows) - positive_count}
                                  if balanced_selection else None),
        "camera": "image0", "resolution": [1024, 1024],
        "depth_format": "PFM float32 source values; metric unit not documented in source README",
        "pfm_row_order": "kept as stored; exact 65,504 sentinel/sky alignment verified per frame",
        "segmentation": "5 exact source RGB colours mapped to scalar IDs 1-5",
        "source_archive_hashes_fully_verified": False,
        "per_asset_sha256_checked": True,
        "per_member_zip_crc_checked_during_extraction": True,
        "qa": qa_rows,
    })
    return {"samples": destination / "samples.jsonl",
            "depth_index": destination / "depth_index.jsonl",
            "validation": destination / "source_validation.json"}


def prepare_lusnar_balanced_crops(selection_index, destination, *, crop_size=500,
                                  target_per_class=76, min_rock_pixels=256,
                                  min_valid_fraction=0.5, min_surface_fraction=0.5):
    """Create balanced, same-frame 500px RGB/mask/depth crops from LuSNAR.

    Positive crops contain at least ``min_rock_pixels`` ROCK-labeled pixels;
    negative crops contain none. One crop at most is selected per source frame.
    The source labels are used only for selection and evaluation targets.
    """
    import numpy as np
    from PIL import Image
    from planetary_vlm.io import write_json, write_jsonl

    selection_index = Path(selection_index).resolve()
    destination = Path(destination).resolve()
    rows = [json.loads(line) for line in selection_index.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Selection index must contain unique source frames")
    if crop_size < 1 or target_per_class < 1 or min_rock_pixels < 1:
        raise ValueError("crop_size, target_per_class and min_rock_pixels must be positive")
    if not 0 < min_valid_fraction <= 1 or not 0 < min_surface_fraction <= 1:
        raise ValueError("Depth and surface fractions must be in (0,1]")
    if destination.exists():
        raise FileExistsError(f"Choose a new crop dataset output: {destination}")

    palette = {
        (187, 70, 156): 1, (120, 0, 200): 2, (232, 250, 80): 3,
        (173, 69, 31): 4, (34, 201, 248): 5,
    }
    palette_array = np.asarray(list(palette), dtype=np.uint8)
    root = selection_index.parent

    def integral_windows(binary):
        integral = np.pad(binary.astype(np.int64).cumsum(0).cumsum(1),
                          ((1, 0), (1, 0)))
        return (integral[crop_size:, crop_size:] - integral[:-crop_size, crop_size:]
                - integral[crop_size:, :-crop_size] + integral[:-crop_size, :-crop_size])

    candidates = []
    for record in rows:
        paths = {name: (root / asset["path"]).resolve() if not Path(asset["path"]).is_absolute()
                 else Path(asset["path"]).resolve()
                 for name, asset in record["assets"].items()}
        for modality, path in paths.items():
            if sha256_file(path) != record["assets"][modality]["sha256"]:
                raise ValueError(f"Selected {modality} asset changed: {record['sample_id']}")
        with Image.open(paths["rgb"]) as opened:
            rgb_size = opened.size
        with Image.open(paths["label"]) as opened:
            label_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        if rgb_size != (1024, 1024) or label_rgb.shape != (1024, 1024, 3):
            raise ValueError(f"Unexpected LuSNAR image/mask size: {record['sample_id']}")
        known_colour = np.zeros((1024, 1024), dtype=bool)
        for colour in palette_array:
            known_colour |= np.all(label_rgb == colour, axis=2)
        if not known_colour.all():
            raise ValueError(f"Unknown semantic palette pixels: {record['sample_id']}")
        rock = np.all(label_rgb == np.asarray((232, 250, 80), dtype=np.uint8), axis=2)
        sky = np.all(label_rgb == np.asarray((34, 201, 248), dtype=np.uint8), axis=2)
        depth = decode_pfm(paths["depth"].read_bytes())
        if depth.shape != rock.shape:
            raise ValueError(f"RGB/mask/depth grids differ: {record['sample_id']}")
        invalid = depth == 65504.0
        if not np.array_equal(sky, invalid):
            raise ValueError(f"Depth sentinel/sky alignment failed: {record['sample_id']}")
        valid = np.isfinite(depth) & (depth > 0) & ~invalid
        rock_counts = integral_windows(rock)
        valid_counts = integral_windows(valid)
        surface_counts = integral_windows(~sky)
        area = crop_size * crop_size
        good = ((valid_counts >= int(np.ceil(min_valid_fraction * area))) &
                (surface_counts >= int(np.ceil(min_surface_fraction * area))))
        positive = np.argwhere(good & (rock_counts >= min_rock_pixels))
        negative = np.argwhere(good & (rock_counts == 0))

        def stable_pick(options, class_name):
            if not len(options):
                return None
            digest = hashlib.sha256(
                f"{LUSNAR_REVISION}:{record['sample_id']}:{class_name}:crop{crop_size}".encode()
            ).digest()
            y, x = options[int.from_bytes(digest[:8], "big") % len(options)]
            return {"x": int(x), "y": int(y), "candidate_windows": int(len(options))}

        candidates.append({
            "record": record,
            "positive_crop": stable_pick(positive, "rock_present"),
            "negative_crop": stable_pick(negative, "rock_absent"),
        })

    positive_only = [item for item in candidates if item["positive_crop"] and not item["negative_crop"]]
    negative_only = [item for item in candidates if item["negative_crop"] and not item["positive_crop"]]
    both = [item for item in candidates if item["positive_crop"] and item["negative_crop"]]
    available = len(positive_only) + len(negative_only) + len(both)
    sample_count = min(target_per_class, available // 2,
                       len(positive_only) + len(both), len(negative_only) + len(both))
    if sample_count < 1:
        summary = {"frames": len(rows), "positive_only": len(positive_only),
                   "negative_only": len(negative_only), "both": len(both),
                   "neither": len(rows) - available}
        raise ValueError(f"No balanced rock-positive/negative crop set possible: {summary}")

    def ordered(items, label):
        return sorted(items, key=lambda item: hashlib.sha256(
            f"{LUSNAR_REVISION}:{item['record']['sample_id']}:{label}:frame-order".encode()
        ).digest())

    selected = []
    by_scene = {}
    for item in candidates:
        if item["positive_crop"] or item["negative_crop"]:
            by_scene.setdefault(item["record"]["scene_group_id"], []).append(item)
    scene_frame_quotas = _proportional_scene_quotas(by_scene, 2 * sample_count)
    positive_targets = {scene: count // 2 for scene, count in scene_frame_quotas.items()}
    extra_positive = sample_count - sum(positive_targets.values())
    odd_scenes = sorted((scene for scene, count in scene_frame_quotas.items() if count % 2),
                        key=lambda scene: hashlib.sha256(
                            f"{LUSNAR_REVISION}:{scene}:positive-class-extra".encode()).digest())
    if extra_positive > len(odd_scenes):
        raise ValueError("Unable to stratify balanced classes across source scenes")
    for scene in odd_scenes[:extra_positive]:
        positive_targets[scene] += 1
    scene_class_targets = {}
    for scene, frame_quota in scene_frame_quotas.items():
        positive_quota = positive_targets[scene]
        negative_quota = frame_quota - positive_quota
        local = by_scene[scene]
        local_positive_only = ordered([item for item in local
                                       if item["positive_crop"] and not item["negative_crop"]],
                                      f"{scene}:positive")
        local_negative_only = ordered([item for item in local
                                       if item["negative_crop"] and not item["positive_crop"]],
                                      f"{scene}:negative")
        local_both = ordered([item for item in local
                              if item["positive_crop"] and item["negative_crop"]], f"{scene}:both")
        if len(local_positive_only) > positive_quota or len(local_negative_only) > negative_quota:
            raise ValueError(f"Cannot stratify scene {scene} to {positive_quota}/{negative_quota}")
        take_positive_only = local_positive_only[:positive_quota]
        take_negative_only = local_negative_only[:negative_quota]
        need_positive = positive_quota - len(take_positive_only)
        need_negative = negative_quota - len(take_negative_only)
        if need_positive + need_negative > len(local_both):
            raise ValueError(f"Insufficient dual-class source frames in scene {scene}")
        selected.extend((item, True) for item in take_positive_only)
        selected.extend((item, False) for item in take_negative_only)
        selected.extend((item, True) for item in local_both[:need_positive])
        selected.extend((item, False) for item in local_both[need_positive:need_positive + need_negative])
        scene_class_targets[scene] = {"rock_present": positive_quota,
                                      "rock_absent": negative_quota}
    if len({item["record"]["sample_id"] for item, _ in selected}) != len(selected):
        raise ValueError("Balanced crop selection reused a source frame")

    destination.mkdir(parents=True, exist_ok=False)
    for folder in ("images", "masks", "depth", "validity"):
        (destination / folder).mkdir()
    samples, depth_rows, qa_rows = [], [], []
    for item, has_rock in selected:
        record = item["record"]
        box = item["positive_crop"] if has_rock else item["negative_crop"]
        paths = {name: (root / asset["path"]).resolve() if not Path(asset["path"]).is_absolute()
                 else Path(asset["path"]).resolve()
                 for name, asset in record["assets"].items()}
        with Image.open(paths["rgb"]) as opened:
            image = opened.convert("RGB").crop((box["x"], box["y"],
                                                  box["x"] + crop_size, box["y"] + crop_size))
        with Image.open(paths["label"]) as opened:
            label_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        y, x = box["y"], box["x"]
        label_crop = label_rgb[y:y + crop_size, x:x + crop_size]
        scalar_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        for colour, label_id in palette.items():
            scalar_mask[np.all(label_crop == np.asarray(colour, dtype=np.uint8), axis=2)] = label_id
        rock_pixels = int((scalar_mask == 3).sum())
        if (has_rock and rock_pixels < min_rock_pixels) or (not has_rock and rock_pixels != 0):
            raise ValueError(f"Crop class failed source-mask check: {record['sample_id']}")
        depth = decode_pfm(paths["depth"].read_bytes())[y:y + crop_size, x:x + crop_size]
        sky = scalar_mask == 5
        invalid = depth == 65504.0
        if not np.array_equal(sky, invalid):
            raise ValueError(f"Crop depth/sky alignment failed: {record['sample_id']}")
        valid = np.isfinite(depth) & (depth > 0) & ~invalid
        if float(valid.mean()) < min_valid_fraction or float((~sky).mean()) < min_surface_fraction:
            raise ValueError(f"Crop depth/surface coverage failed: {record['sample_id']}")

        crop_class = "rock_present" if has_rock else "rock_absent"
        sample_id = f"{record['sample_id']}_crop500_{crop_class}"
        image_path = destination / "images" / f"{sample_id}.png"
        mask_path = destination / "masks" / f"{sample_id}.png"
        depth_path = destination / "depth" / f"{sample_id}.source_values.npy"
        validity_path = destination / "validity" / f"{sample_id}.validity.npy"
        image.save(image_path, format="PNG")
        Image.fromarray(scalar_mask, mode="L").save(mask_path, format="PNG")
        np.save(depth_path, depth.astype(np.float32, copy=False), allow_pickle=False)
        np.save(validity_path, valid, allow_pickle=False)
        image_sha, mask_sha = sha256_file(image_path), sha256_file(mask_path)
        depth_sha, validity_sha = sha256_file(depth_path), sha256_file(validity_path)
        samples.append({"sample_id": sample_id, "image_path": image_path.relative_to(destination).as_posix(),
            "mask_path": mask_path.relative_to(destination).as_posix(),
            "group_id": record["scene_group_id"], "split": "test",
            "license": "MIT (publisher claim; publication review pending)",
            "source_url": "https://huggingface.co/datasets/xuboluo2001/LuSNAR"})
        depth_rows.append({"sample_id": sample_id, "status": "technical_depth_qa_pass",
            "image_path": str(image_path.resolve()),
            "registration": {"benchmark_sha256": image_sha},
            "depth_path": str(depth_path.resolve()), "depth_sha256": depth_sha,
            "validity_path": str(validity_path.resolve()), "validity_sha256": validity_sha,
            "depth_origin": record["depth_origin"], "depth_units": "unspecified_simulator_native_values",
            "source_rgb_sha256": record["assets"]["rgb"]["sha256"],
            "source_rgb_member": record["assets"]["rgb"]["source_member"],
            "source_label_sha256": record["assets"]["label"]["sha256"],
            "source_label_member": record["assets"]["label"]["source_member"],
            "source_depth_sha256": record["assets"]["depth"]["sha256"],
            "source_depth_member": record["assets"]["depth"]["source_member"],
            "source_archive_sha256": record["source_archive_sha256"],
            "source_archive_sha256_verified": False,
            "scene_group_id": record["scene_group_id"],
            "source_sample_id": record["sample_id"], "source_split": record["split"],
            "crop_xyxy": [x, y, x + crop_size, y + crop_size],
            "crop_class": crop_class, "rock_pixels": rock_pixels,
            "valid_fraction": float(valid.mean()), "surface_fraction": float((~sky).mean()),
            "source_crop_sha256": {"rgb": image_sha, "mask": mask_sha,
                                    "depth": depth_sha, "validity": validity_sha}})
        qa_rows.append({"sample_id": sample_id, "source_sample_id": record["sample_id"],
            "scene_group_id": record["scene_group_id"], "source_split": record["split"],
            "crop_xyxy": [x, y, x + crop_size, y + crop_size],
            "dimensions": [crop_size, crop_size], "crop_class": crop_class,
            "rock_pixels": rock_pixels, "valid_depth_fraction": float(valid.mean()),
            "surface_fraction": float((~sky).mean())})

    write_jsonl(destination / "samples.jsonl", samples)
    write_jsonl(destination / "depth_index.jsonl", depth_rows)
    write_json(destination / "source_validation.json", {
        "source": "LuSNAR", "revision": LUSNAR_REVISION, "synthetic": True,
        "crop_size": crop_size, "selected_crops": len(selected),
        "rock_present": sum(row["crop_class"] == "rock_present" for row in depth_rows),
        "rock_absent": sum(row["crop_class"] == "rock_absent" for row in depth_rows),
        "positive_definition": f"at least {min_rock_pixels} ROCK semantic pixels in crop",
        "negative_definition": "zero ROCK semantic pixels in crop",
        "selection_candidate_counts": {"positive_only_frames": len(positive_only),
            "negative_only_frames": len(negative_only), "both_classes_available_frames": len(both),
            "frames_with_no_eligible_crop": len(rows) - available},
        "selected_scene_class_counts": scene_class_targets,
        "one_crop_per_source_frame": True,
        "valid_depth_min_fraction": min_valid_fraction,
        "surface_min_fraction": min_surface_fraction,
        "depth_units": "native simulator PFM values; metric units unverified",
        "per_asset_sha256_checked": True,
        "source_archive_hashes_fully_verified": False,
        "qa": qa_rows,
    })
    return {"samples": destination / "samples.jsonl",
            "depth_index": destination / "depth_index.jsonl",
            "validation": destination / "source_validation.json"}


def make_lusnar_track_portable(dataset_root, *, source_validation=None):
    """Rebase model/depth assets to track-relative paths for folder transfer."""
    from planetary_vlm.io import read_jsonl

    dataset_root = Path(dataset_root).resolve()
    track = dataset_root / "moon" / "test"
    requests_path = track / "requests.jsonl"
    depth_path = track / "depth_index.jsonl"
    provenance_path = track / "provenance.jsonl"
    samples_path = dataset_root / "source_samples.jsonl"
    manifest_path = dataset_root / "manifest.json"
    requests = read_jsonl(requests_path)
    depths = read_jsonl(depth_path)
    provenance = read_jsonl(provenance_path)
    by_sample = {row["sample_id"]: row for row in depths}
    provenance_by_sample = {row["sample_id"]: row for row in provenance}

    def local_relative(value, base):
        candidate = Path(value)
        absolute = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
        relative = absolute.relative_to(base.resolve()).as_posix()
        if not (base / relative).is_file():
            raise FileNotFoundError(base / relative)
        return relative

    for request in requests:
        request["image_paths"] = [local_relative(path, track) for path in request["image_paths"]]
        request["modality_paths"] = [[name, local_relative(path, track)]
                                      for name, path in request.get("modality_paths", [])]
    for entry in depths:
        for field in ("depth_path", "validity_path", "image_path", "mask_path"):
            entry[field] = local_relative(entry[field], track)
        entry.pop("full_frame_depth_path", None)
    for row in provenance:
        entry = by_sample[row["sample_id"]]
        row["source_image_path"] = entry["source_rgb_member"]
        row["source_mask_path"] = entry["source_label_member"]
        row["source_depth_path"] = entry["source_depth_member"]
        row["source_path_kind"] = "member_of_pinned_lusnar_scene_zip"
        row["source_archive_sha256"] = entry["source_archive_sha256"]
        row["source_archive_sha256_verified"] = False
        row["source_rgb_sha256"] = entry["source_rgb_sha256"]
        row["source_mask_sha256"] = entry["source_label_sha256"]
        row["source_depth_sha256"] = entry["source_depth_sha256"]
        row["image_path"] = entry["source_rgb_member"]
        row["mask_path"] = entry["source_label_member"]
        row["prepared_image"] = local_relative(row["prepared_image"], track)
        row["prepared_mask"] = local_relative(row["prepared_mask"], track)

    def replace_jsonl(path, rows):
        temporary = path.with_name(path.name + ".portable.tmp")
        if temporary.exists():
            raise FileExistsError(temporary)
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                             for row in rows), encoding="utf-8")
        os.replace(temporary, path)

    replace_jsonl(requests_path, requests)
    replace_jsonl(depth_path, depths)
    replace_jsonl(provenance_path, provenance)
    if samples_path.is_file():
        samples = read_jsonl(samples_path)
        for sample in samples:
            row = provenance_by_sample[sample["sample_id"]]
            sample["image_path"] = f"moon/test/{row['prepared_image']}"
            sample["mask_path"] = f"moon/test/{row['prepared_mask']}"
        replace_jsonl(samples_path, samples)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["portable_paths_relative_to_track"] = True
    manifest["source_assets_included"] = False
    manifest["prepared_track_assets_included"] = True
    manifest["source_lineage"] = "pinned archive members and SHA-256; archive-level digest not verified"
    if samples_path.is_file():
        manifest["portable_source_samples_sha256"] = sha256_file(samples_path)
    if source_validation is not None:
        import shutil
        source_validation = Path(source_validation).resolve()
        validation_copy = dataset_root / "source_validation.json"
        shutil.copyfile(source_validation, validation_copy)
        manifest["source_validation"] = validation_copy.name
        manifest["source_validation_sha256"] = sha256_file(validation_copy)
    manifest["portability_rewriter_sha256"] = sha256_file(Path(__file__))
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".portable.tmp")
    if temporary_manifest.exists():
        raise FileExistsError(temporary_manifest)
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return dataset_root


LUSNAR_REVISION = "df281ef5139f78a21a162d684186efd3c2ae3437"
LUSNAR_SCENES = {
    1: {"size": 4243344460, "sha256": None},
    2: {"size": 6361047810, "sha256": None},
    3: {"size": 4864911294, "sha256": "3b298919fe5e8500c5a7991370b46f0b4637f46e4c438926331068c593e996e0"},
    4: {"size": 5194595554, "sha256": None},
    5: {"size": 6651319456, "sha256": "b6c03288cd2aeb5bf8039d72ea5da1890c6544c1eede7b719f40d3645402cb0b"},
    6: {"size": 4463226364, "sha256": None},
    7: {"size": 6406486542, "sha256": "85f51e5f1ffc3615f1acc3d0a59224bcdbfa78f7bee68ac691d5ce3ce860c595"},
    8: {"size": 5422201602, "sha256": None},
    9: {"size": 8694422670, "sha256": None},
}
LUSNAR_TEST_SCENES = (3, 5, 7)
LUSNAR_SCENE_SPLITS = {
    1: "official_semantic_train", 2: "official_semantic_train",
    3: "official_semantic_test", 4: "official_semantic_train",
    5: "official_semantic_test", 6: "official_semantic_train",
    7: "official_semantic_test", 8: "official_semantic_train",
    9: "official_semantic_train",
}


def collect_lusnar_subset(destination, *, quotas=None):
    """Fetch 152 evenly spaced complete triplets from held-out scenes 3/5/7."""
    destination = Path(destination)
    if destination.exists():
        allowed = {"scene_03_extract", "scene_05_extract", "scene_07_extract"}
        unexpected = {item.name for item in destination.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(f"Unexpected existing collection files: {sorted(unexpected)}")
    destination.mkdir(parents=True, exist_ok=True)
    quotas = dict(quotas or {3: 50, 5: 51, 7: 51})
    if set(quotas) != set(LUSNAR_TEST_SCENES) or sum(quotas.values()) != 152:
        raise ValueError("Quotas must cover test scenes 3, 5, 7 and sum to exactly 152")
    records = []
    for scene_id, quota in quotas.items():
        source_url = (f"https://huggingface.co/datasets/xuboluo2001/LuSNAR/resolve/"
                      f"{LUSNAR_REVISION}/Moon_{scene_id}.zip?download=true")
        remote = HTTPRangeReader(source_url, LUSNAR_SCENES[scene_id]["size"],
                                 block_size=1024 * 1024, cache_blocks=8)
        with zipfile.ZipFile(remote) as archive:
            records.extend(collect_scene(
                archive, destination / f"scene_{scene_id:02d}_extract",
                scene_id=scene_id, requested=quota, source_url=source_url,
                source_revision=LUSNAR_REVISION,
                source_archive_sha256=LUSNAR_SCENES[scene_id]["sha256"], camera="image0"))
    if len(records) != 152 or len({row["sample_id"] for row in records}) != 152:
        raise ValueError("Collection did not produce exactly 152 unique frames")
    save_selection_index(records, destination / "selected_assets.jsonl")
    return records


def scan_lusnar_rock_presence(archive, *, camera="image0", progress_path=None):
    """Read semantic label members, optionally checkpointing for resume."""
    import numpy as np
    from PIL import Image

    rock_colour = np.asarray((232, 250, 80), dtype=np.uint8)
    frames = matched_frames(archive, camera=camera)
    progress_path = Path(progress_path) if progress_path is not None else None
    rows_by_index = {}
    if progress_path is not None and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (progress.get("camera") != camera or progress.get("frame_count") != len(frames)
                or progress.get("source_revision") != LUSNAR_REVISION):
            raise ValueError(f"Rock-scan checkpoint does not match source: {progress_path}")
        rows_by_index = {int(row["frame_index"]): row for row in progress["frames"]}

    def save_progress():
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"source_revision": LUSNAR_REVISION, "camera": camera,
                   "frame_count": len(frames),
                   "frames": [rows_by_index[index] for index in sorted(rows_by_index)]}
        temporary = progress_path.with_name(progress_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, progress_path)

    for frame_index, (stem, assets) in enumerate(frames):
        if frame_index in rows_by_index:
            if rows_by_index[frame_index]["frame_stem"] != stem:
                raise ValueError(f"Rock-scan checkpoint frame changed at index {frame_index}")
            continue
        with archive.open(assets["label"], "r") as stream:
            with Image.open(stream) as label_image:
                labels = np.asarray(label_image.convert("RGB"), dtype=np.uint8)
        if labels.shape != (1024, 1024, 3):
            raise ValueError(f"Unexpected label dimensions for {stem}: {labels.shape}")
        rock_pixels = int(np.all(labels == rock_colour, axis=2).sum())
        rows_by_index[frame_index] = {"frame_index": frame_index, "frame_stem": stem,
                                      "rock_pixels": rock_pixels, "has_rock": rock_pixels > 0}
        if (frame_index + 1) % 25 == 0:
            save_progress()
    save_progress()
    if len(rows_by_index) != len(frames):
        raise ValueError(f"Incomplete rock scan: {len(rows_by_index)} of {len(frames)} frames")
    return [rows_by_index[index] for index in range(len(frames))]


def _proportional_scene_quotas(candidates_by_scene, requested):
    """Allocate a sample count proportional to class availability, with caps."""
    capacities = {scene: len(rows) for scene, rows in candidates_by_scene.items()}
    if sum(capacities.values()) < requested:
        raise ValueError(f"Need {requested} frames in class; found {sum(capacities.values())}")
    quotas = {scene: 0 for scene in capacities}
    remaining = requested
    while remaining:
        active = [scene for scene, capacity in capacities.items() if quotas[scene] < capacity]
        weights = sum(capacities[scene] for scene in active)
        if not active or weights == 0:
            raise ValueError("Unable to allocate requested samples across scenes")
        ideal = {scene: remaining * capacities[scene] / weights for scene in active}
        additions = {scene: min(capacities[scene] - quotas[scene], int(ideal[scene]))
                     for scene in active}
        placed = sum(additions.values())
        for scene, addition in additions.items():
            quotas[scene] += addition
        remaining -= placed
        if remaining:
            order = sorted((scene for scene in active if quotas[scene] < capacities[scene]),
                           key=lambda scene: (-(ideal[scene] - int(ideal[scene])), scene))
            if not order:
                continue
            for scene in order:
                if remaining == 0:
                    break
                quotas[scene] += 1
                remaining -= 1
    return quotas


def collect_lusnar_balanced_subset(destination, *, total_samples=152,
                                   positive_fraction=0.5, scene_ids=LUSNAR_TEST_SCENES):
    """Build a class-balanced complete-triplet subset from selected LuSNAR scenes.

    Rock presence means at least one pixel carrying the publisher's exact ROCK
    semantic colour. All selected frames retain their original RGB/depth/mask.
    """
    destination = Path(destination)
    if destination.exists():
        allowed = {f"scene_{scene:02d}_rock_scan.json" for scene in scene_ids}
        allowed.update({f"scene_{scene:02d}_extract" for scene in scene_ids})
        allowed.update({"selected_assets.jsonl", "rock_presence_scan.json"})
        unexpected = {item.name for item in destination.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(f"Unexpected files in collection directory: {sorted(unexpected)}")
    if total_samples < 2 or total_samples % 2:
        raise ValueError("total_samples must be an even integer")
    if not 0 < positive_fraction < 1:
        raise ValueError("positive_fraction must be between zero and one")
    positive_n = round(total_samples * positive_fraction)
    negative_n = total_samples - positive_n
    if positive_n != negative_n:
        raise ValueError("This collector currently requires an exactly balanced 50/50 selection")
    if not scene_ids or set(scene_ids) - set(LUSNAR_SCENES):
        raise ValueError("Scene IDs must be selected from the pinned LuSNAR release")
    destination.mkdir(parents=True, exist_ok=True)
    scene_candidates = {}
    scene_archives = {}
    scene_urls = {}
    for scene_id in scene_ids:
        source_url = (f"https://huggingface.co/datasets/xuboluo2001/LuSNAR/resolve/"
                      f"{LUSNAR_REVISION}/Moon_{scene_id}.zip?download=true")
        remote = HTTPRangeReader(source_url, LUSNAR_SCENES[scene_id]["size"],
                                 block_size=1024 * 1024, cache_blocks=8)
        archive = zipfile.ZipFile(remote)
        scene_archives[scene_id] = archive
        scene_urls[scene_id] = source_url
        scene_candidates[scene_id] = scan_lusnar_rock_presence(
            archive, progress_path=destination / f"scene_{scene_id:02d}_rock_scan.json")

    class_scenes = {
        True: {scene: [row for row in rows if row["has_rock"]]
               for scene, rows in scene_candidates.items()},
        False: {scene: [row for row in rows if not row["has_rock"]]
                for scene, rows in scene_candidates.items()},
    }
    scan_counts = {str(scene): {
        "frames": len(rows), "with_rock": sum(row["has_rock"] for row in rows),
        "without_rock": sum(not row["has_rock"] for row in rows)}
        for scene, rows in scene_candidates.items()}
    scan_report = {
        "source_revision": LUSNAR_REVISION, "scenes": list(scene_ids),
        "definition": "has_rock = at least one pixel with exact LuSNAR ROCK palette colour",
        "candidate_counts": scan_counts,
        "selected_total": 0, "selected_positive": 0, "selected_negative": 0,
        "selection_status": "not_built_no_valid_class_balance",
    }
    report_path = destination / "rock_presence_scan.json"
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    temporary_report.write_text(json.dumps(scan_report, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    os.replace(temporary_report, report_path)
    selected_indices = {scene: [] for scene in scene_ids}
    selected_rows = []
    for has_rock, requested in ((True, positive_n), (False, negative_n)):
        try:
            quotas = _proportional_scene_quotas(class_scenes[has_rock], requested)
        except ValueError as error:
            raise ValueError(f"Cannot form balanced selection; candidate counts: {scan_counts}; {error}") from error
        for scene, count in quotas.items():
            if not count:
                continue
            ordered = class_scenes[has_rock][scene]
            chosen_positions = evenly_spaced_indices(len(ordered), count)
            for position in chosen_positions:
                row = ordered[position]
                selected_indices[scene].append(row["frame_index"])
                selected_rows.append({**row, "scene_id": scene})

    records = []
    for scene_id in scene_ids:
        quota_indices = sorted(selected_indices[scene_id])
        if not quota_indices:
            continue
        records.extend(collect_scene(
            scene_archives[scene_id], destination / f"scene_{scene_id:02d}_extract",
            scene_id=scene_id, frame_indices=quota_indices,
            source_url=scene_urls[scene_id], source_revision=LUSNAR_REVISION,
            source_archive_sha256=LUSNAR_SCENES[scene_id]["sha256"], camera="image0"))
    for archive in scene_archives.values():
        archive.close()
    if len(records) != total_samples or len({row["sample_id"] for row in records}) != total_samples:
        raise ValueError(f"Expected {total_samples} unique frames; collected {len(records)}")
    metadata_by_id = {(row["scene_id"], row["frame_stem"]): row for row in selected_rows}
    for record in records:
        annotation = metadata_by_id[(int(record["scene_group_id"].rsplit("_", 1)[1]),
                                     record["frame_stem"])]
        record["rock_pixels_in_source_mask"] = annotation["rock_pixels"]
        record["rock_presence"] = annotation["has_rock"]
    save_selection_index(records, destination / "selected_assets.jsonl")
    from planetary_vlm.io import write_json
    write_json(destination / "rock_presence_scan.json", {
        "source_revision": LUSNAR_REVISION, "scenes": list(scene_ids),
        "definition": "has_rock = at least one pixel with exact LuSNAR ROCK palette colour",
        "selected_total": total_samples, "selected_positive": positive_n,
        "selected_negative": negative_n,
        "selection_status": "balanced_complete_triplets_collected",
        "candidate_counts": scan_counts,
        "selected_counts_by_scene": {str(scene): {
            "positive": sum(row["scene_id"] == scene and row["has_rock"] for row in selected_rows),
            "negative": sum(row["scene_id"] == scene and not row["has_rock"] for row in selected_rows)}
            for scene in scene_ids},
        "selected_frames": selected_rows,
    })
    return records
