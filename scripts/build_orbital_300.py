"""Build a bounded, source-labelled 300-image orbital collection from local caches.

Run before pruning source caches. This never reads data_rover.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_orbital"
OLD = DATA / "working" / "selected_v2"
OUT = DATA / "orbital_300_v3"
MULTI = DATA / "_staging" / "mars_multi_crater" / "test.parquet"
MULTI_REV = "9a26ac9001532bc9f4fe37abfa3fce13e4ea1b74"
MULTI_CLASSES = ["Background", "Other", "Layered", "Buried", "Secondary"]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_path(value: str) -> Path:
    p = ROOT / value if value.startswith("data_orbital/") else DATA / value
    p = p.resolve()
    if DATA.resolve() not in p.parents or not p.is_file():
        raise ValueError(f"Missing or unsafe source: {value}")
    return p


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_bytes(folder: Path, name: str, data: bytes) -> tuple[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(data)
    return path.relative_to(ROOT).as_posix(), digest(data)


def put_file(folder: Path, name: str, source: Path) -> tuple[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    shutil.copy2(source, target)
    return target.relative_to(ROOT).as_posix(), digest(target.read_bytes())


def image_size(source: Path) -> tuple[int, int]:
    if source.suffix.lower() in (".tif", ".tiff"):
        array = tifffile.imread(source)
        if array.ndim == 3:
            return int(array.shape[1]), int(array.shape[0])
    with Image.open(source) as image:
        return image.size


def put_image(folder: Path, source: Path) -> tuple[str, str, dict | None]:
    if source.suffix.lower() in (".tif", ".tiff"):
        array = tifffile.imread(source)
        if array.ndim == 3 and array.shape[-1] >= 3:
            rgb = []
            for band in range(3):
                values = np.asarray(array[..., band], dtype=np.float32)
                finite = np.isfinite(values)
                if not finite.any():
                    raise ValueError(f"Empty visible band in {source}")
                lo, hi = np.percentile(values[finite], (2, 98))
                scale = np.clip((values - lo) / max(float(hi - lo), 1e-8), 0, 1)
                rgb.append((np.nan_to_num(scale) * 255).astype(np.uint8))
            buffer = io.BytesIO()
            Image.fromarray(np.stack(rgb, axis=-1), "RGB").save(buffer, format="PNG")
            path, sha = put_bytes(folder, "image.png", buffer.getvalue())
            return path, sha, {"method": "per_tile_visible_rgb_percentile_v1", "bands_zero_based": [0, 1, 2],
                               "percentiles": [2, 98], "source_raster_sha256": digest(source.read_bytes())}
    path, sha = put_file(folder, "image" + source.suffix.lower(), source)
    return path, sha, None


def mask_values(path: Path) -> list[int]:
    return [int(x) for x in np.unique(np.asarray(Image.open(path)))]


def positive(rows: list[dict]) -> list[dict]:
    return [r for r in rows if any(mask_values(source_path(r.get("mask") or r["segmentation_mask"])))]


def stable(row: dict) -> str:
    return digest(str(row.get("source_image_id", row.get("sample_id"))).encode())


def take_spread(rows: list[dict], n: int, used_groups: set[str] | None = None) -> list[dict]:
    """Round-robin source groups, with deterministic order within each group."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        group = str(row.get("scene_group_id") or row.get("product_id") or row.get("source_image_id"))
        if used_groups and group in used_groups:
            continue
        groups.setdefault(group, []).append(row)
    for group in groups:
        groups[group].sort(key=stable)
    result: list[dict] = []
    while len(result) < n and any(groups.values()):
        for group in sorted(groups, key=lambda g: (len([r for r in result if str(r.get("scene_group_id") or r.get("product_id") or r.get("source_image_id")) == g]), g)):
            if groups[group] and len(result) < n:
                result.append(groups[group].pop(0))
    if len(result) != n:
        raise ValueError(f"Only {len(result)} available; requested {n}")
    return result


def add_mask(planet: str, rows: list[dict], start: int, out: list[dict], *, semantics: str) -> None:
    for row in rows:
        number = start + len(out)
        folder = OUT / planet / "segmentation" / f"{number:03d}"
        image_src = source_path(row["image"])
        mask_src = source_path(row.get("mask") or row["segmentation_mask"])
        with Image.open(mask_src) as mask:
            if image_size(image_src) != mask.size:
                raise ValueError(f"Image/mask size mismatch: {image_src}")
        values = mask_values(mask_src)
        if not any(values):
            raise ValueError(f"Empty mask: {mask_src}")
        image, image_hash, display_transform = put_image(folder, image_src)
        mask, mask_hash = put_file(folder, "mask" + mask_src.suffix.lower(), mask_src)
        split = row.get("source_split") or row.get("split") or row.get("sample_split")
        if split not in ("val", "test"):
            raise ValueError(f"Segmentation source is not val/test: {split}")
        classes = row.get("classes") or (["Background", "IMP"] if semantics == "imp" else ["Background", "Crater"])
        out.append({
            "sample_id": f"{planet}_seg_{number:03d}", "planet": planet, "track": "segmentation",
            "source_dataset": row.get("source_repo") or row.get("source") or row.get("dataset"),
            "source_revision": row.get("source_revision") or row.get("revision"),
            "source_split": split, "evaluation_eligible": True, "final_test_eligible": split == "test",
            "source_image_id": row.get("source_image_id") or row.get("sample_id"),
            "scene_group_id": row.get("scene_group_id") or row.get("product_id"),
            "mask_semantics": semantics, "classes": classes, "mask_values_present": values,
            "annotation_method": row.get("annotation_method") or row.get("mask_semantics"),
            "display_transform": display_transform,
            "license_claim": row.get("license_claim") or "CC-BY-4.0 (dataset card)",
            "image": image, "mask": mask,
            "image_sha256": image_hash, "mask_sha256": mask_hash,
        })


def add_multi(out: list[dict]) -> None:
    parquet = pq.ParquetFile(MULTI)
    choices = []
    for group_idx in range(parquet.num_row_groups):
        for row in parquet.read_row_group(group_idx).to_pylist():
            name = row["image"]["path"]
            image_bytes = row["image"]["bytes"]
            with Image.open(io.BytesIO(row["mask"]["bytes"])) as mask:
                counts = np.bincount(np.asarray(mask).ravel(), minlength=5)
                values = {int(x) for x in np.flatnonzero(counts)}
            labels = {MULTI_CLASSES[x] for x in values if x in range(1, 5)}
            if not labels:
                continue
            min_class_pixels = int(min(counts[x] for x in values if x in range(1, 5)))
            choices.append((name, image_bytes, row["mask"]["bytes"], labels, min_class_pixels))
    # Include all four crater types in multiple source regions whenever possible.
    chosen = []
    scene_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    remaining = choices[:]
    while len(chosen) < 12:
        best = min(remaining, key=lambda x: (
            -len(x[3]), scene_counts[x[0].rsplit("_", 1)[0]],
            -x[4], digest(x[0].encode()),
        ))
        remaining.remove(best)
        chosen.append(best)
        scene_counts[best[0].rsplit("_", 1)[0]] += 1
        label_counts.update(best[3])
    if any(label_counts[c] < 10 for c in MULTI_CLASSES[1:]):
        raise ValueError(f"Insufficient multi-crater class coverage: {label_counts}")
    for name, image_bytes, mask_bytes, _, _ in chosen:
        number = len(out) + 1
        folder = OUT / "mars" / "segmentation" / f"{number:03d}"
        with Image.open(io.BytesIO(image_bytes)) as image, Image.open(io.BytesIO(mask_bytes)) as mask:
            if image.size != mask.size or image.size != (512, 512):
                raise ValueError(f"Invalid multi-crater pair: {name}")
            values = [int(x) for x in np.unique(np.asarray(mask))]
        if any(x not in range(5) for x in values) or not any(values):
            raise ValueError(f"Invalid multi-crater codes: {name}: {values}")
        image, image_hash = put_bytes(folder, "image.png", image_bytes)
        mask, mask_hash = put_bytes(folder, "mask.png", mask_bytes)
        out.append({
            "sample_id": f"mars_seg_{number:03d}", "planet": "mars", "track": "segmentation",
            "source_dataset": "Mirali33/mb-crater_multi_seg", "source_revision": MULTI_REV,
            "source_split": "test", "evaluation_eligible": True, "final_test_eligible": True,
            "source_image_id": name, "scene_group_id": name.rsplit("_", 1)[0],
            "mask_semantics": "Mars-Bench crater type pixel mask", "classes": MULTI_CLASSES,
            "mask_values_present": values, "license_claim": "CC-BY-4.0 (dataset card)",
            "image": image, "mask": mask, "image_sha256": image_hash, "mask_sha256": mask_hash,
        })


def copy_old_track(planet: str, track: str) -> list[dict]:
    rows = read_rows(OLD / planet / f"{track}_manifest.jsonl")
    output = []
    expected = ("image", "height_map", "height_valid_mask") if track == "height" else ("image", "segmentation_mask", "height_map")
    for i, old in enumerate(rows, 1):
        folder = OUT / planet / track / f"{i:03d}"
        new = dict(old)
        new["sample_id"] = f"{planet}_{track}_{i:03d}"
        new["track"] = track
        if track == "train":
            new["evaluation_eligible"] = False
            new["model_inference_use"] = False
        for key in expected:
            src = source_path(old[key])
            if key == "image":
                new[key], new["image_sha256"], new["display_transform"] = put_image(folder, src)
            else:
                new[key], new[key + "_sha256"] = put_file(folder, key + src.suffix.lower(), src)
        output.append(new)
    if len(output) != 50:
        raise ValueError(f"Expected 50 {planet}/{track}, got {len(output)}")
    return output


def copy_mars_height() -> list[dict]:
    all_rows = read_rows(DATA / "working" / "mars" / "mmlsv2" / "manifest.jsonl")
    train_xy = [(int(r["grid_x"]), int(r["grid_y"])) for r in all_rows if r["sample_split"] == "train"]
    eligible = []
    for row in all_rows:
        if row["sample_split"] != "eval":
            continue
        x, y = int(row["grid_x"]), int(row["grid_y"])
        distance = min(abs(x - a) + abs(y - b) for a, b in train_xy)
        if distance >= 3:
            eligible.append((row, distance))
    if len(eligible) < 50:
        raise ValueError("Fewer than 50 Mars height tiles separated from train")
    # Farthest point sampling across the eligible Mars grid.
    first = max(range(len(eligible)), key=lambda i: (eligible[i][1], stable(eligible[i][0])))
    chosen = [first]
    while len(chosen) < 50:
        nxt = max((i for i in range(len(eligible)) if i not in chosen), key=lambda i: (
            min((int(eligible[i][0]["grid_x"]) - int(eligible[j][0]["grid_x"])) ** 2 +
                (int(eligible[i][0]["grid_y"]) - int(eligible[j][0]["grid_y"])) ** 2 for j in chosen),
            eligible[i][1], stable(eligible[i][0]),
        ))
        chosen.append(nxt)
    output = []
    for i, index in enumerate(chosen, 1):
        row, distance = eligible[index]
        folder = OUT / "mars" / "height" / f"{i:03d}"
        image_src = source_path(row["image"])
        image, image_sha, transform = put_image(folder, image_src)
        height, height_sha = put_file(folder, "height_map.npy", source_path(row["height_map"]))
        valid, valid_sha = put_file(folder, "height_valid_mask.npy", source_path(row["height_valid_mask"]))
        split = row["evaluation_stratum"]
        output.append({
            "sample_id": f"mars_height_{i:03d}", "planet": "mars", "track": "height",
            "source_dataset": row["dataset"], "source_revision": row["revision"],
            "source_split": split, "evaluation_eligible": True, "final_test_eligible": split == "test",
            "source_image_id": row["sample_id"], "scene_group_id": f"mmlsv2_grid_{row['grid_x']}_{row['grid_y']}",
            "grid_x": int(row["grid_x"]), "grid_y": int(row["grid_y"]),
            "nearest_train_grid_manhattan_tiles": distance,
            "height_units": row["height_units"], "license_claim": row.get("license_claim") or "CC-BY-4.0 (dataset card)",
            "image": image, "image_sha256": image_sha, "display_transform": transform,
            "height_map": height, "height_map_sha256": height_sha,
            "height_valid_mask": valid, "height_valid_mask_sha256": valid_sha,
        })
    return output


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    mars: list[dict] = []
    add_multi(mars)
    boulders = read_rows(DATA / "source_index" / "mars_boulder_seg.jsonl")
    add_mask("mars", boulders, 1, mars, semantics="boulder")
    craters = read_rows(DATA / "source_index" / "mars_crater_binary_seg.jsonl")
    crater_groups = {x["scene_group_id"] for x in mars}
    add_mask("mars", take_spread(positive([x for x in craters if x["split"] in ("val", "test")]), 14, crater_groups), 1, mars, semantics="crater_binary")
    cones = read_rows(DATA / "source_index" / "mars_conequest_seg.jsonl")
    add_mask("mars", take_spread(positive([x for x in cones if x["split"] in ("val", "test")]), 20), 1, mars, semantics="cone")
    moon: list[dict] = []
    debris = read_rows(DATA / "source_index" / "moon_debris_nontrain.jsonl")
    add_mask("moon", debris, 1, moon, semantics="pit_stone_crater")
    imp = read_rows(DATA / "source_index" / "moon_imp_seg.jsonl")
    add_mask("moon", take_spread(positive([x for x in imp if x["split"] in ("val", "test")]), 21), 1, moon, semantics="imp")
    moon_height = copy_old_track("moon", "height")
    wac = read_rows(DATA / "working" / "moon" / "wac_crater_height" / "manifest.jsonl")
    used = {str(x["scene_group_id"]) for x in moon_height}
    wac_eval = [x for x in wac if x["sample_split"] in ("val", "test")]
    add_mask("moon", take_spread(wac_eval, 20, used), 1, moon, semantics="catalog_crater_polygon")
    records = {
        ("mars", "segmentation"): mars, ("moon", "segmentation"): moon,
        ("mars", "height"): copy_mars_height(),
        ("moon", "height"): moon_height,
        ("mars", "train"): copy_old_track("mars", "train"),
        ("moon", "train"): copy_old_track("moon", "train"),
    }
    all_rows = []
    for (planet, track), rows in records.items():
        if len(rows) != 50:
            raise ValueError(f"Expected 50 {planet}/{track}, got {len(rows)}")
        write_rows(OUT / planet / f"{track}_manifest.jsonl", rows)
        all_rows += rows
    image_hashes = [r["image_sha256"] for r in all_rows]
    if len(image_hashes) != 300 or len(set(image_hashes)) != 300:
        raise ValueError("Images are missing or duplicated")
    if len({r["sample_id"] for r in all_rows}) != 300:
        raise ValueError("Duplicate sample IDs")
    moon_train_groups = {x["scene_group_id"] for x in records["moon", "train"]}
    moon_eval_groups = {x["scene_group_id"] for x in moon_height if x["source_dataset"] == "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection"}
    moon_eval_groups.update(x["scene_group_id"] for x in moon if x["mask_semantics"] == "catalog_crater_polygon")
    if moon_train_groups & moon_eval_groups:
        raise ValueError("Moon WAC train/eval product overlap")
    report = {"selection_version": "orbital-300-v3", "total_images": len(all_rows),
              "counts": {f"{p}_{t}": len(rs) for (p, t), rs in records.items()},
              "segmentation_sources": {p: dict(Counter(x["source_dataset"] for x in records[p, "segmentation"])) for p in ("mars", "moon")},
              "segmentation_splits": {p: dict(Counter(x["source_split"] for x in records[p, "segmentation"])) for p in ("mars", "moon")},
              "segmentation_values": {p: {k: sorted({v for x in records[p, "segmentation"] if x["mask_semantics"] == k for v in x["mask_values_present"]}) for k in sorted({x["mask_semantics"] for x in records[p, "segmentation"]})} for p in ("mars", "moon")},
              "moon_wac_train_eval_product_overlap": 0}
    (OUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
