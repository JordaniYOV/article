"""Create a bounded, provenance-preserving orbital benchmark subset.

This reads only data_orbital and never accesses the protected data_rover archive.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_orbital"
OUT = DATA / "working" / "selected_v2"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local(relative: str) -> Path:
    relpath = Path(relative)
    base = ROOT if relpath.parts and relpath.parts[0] == "data_orbital" else DATA
    p = (base / relpath).resolve()
    if DATA.resolve() not in p.parents:
        raise ValueError(f"Path escaped data_orbital: {relative}")
    return p


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sample_mask_fraction(r: dict) -> float:
    p = local(r["mask"])
    a = np.asarray(Image.open(p))
    return float((a > 0).mean())


def balanced_select(candidates: list[dict], count: int, *, allow_zero: bool = False) -> list[dict]:
    """Balance source groups and foreground-coverage quantiles deterministically."""
    enriched = []
    for r in candidates:
        image = local(r["image"])
        mask = local(r["mask"])
        if not image.is_file() or not mask.is_file():
            continue
        frac = sample_mask_fraction(r)
        if frac <= 0 and not allow_zero:
            continue
        enriched.append((r, frac))
    if len(enriched) < count:
        raise RuntimeError(f"Requested {count} masks, only {len(enriched)} locally available")
    vals = np.asarray([x[1] for x in enriched])
    edges = np.quantile(vals, [0, .25, .5, .75, 1])
    group_counts: Counter[str] = Counter()
    bin_counts: Counter[int] = Counter()
    remaining = list(enriched)
    selected = []
    while remaining and len(selected) < count:
        scored = []
        for r, frac in remaining:
            b = min(3, int(np.searchsorted(edges[1:-1], frac, side="right")))
            group = str(r.get("scene_group_id") or r.get("source_image_id"))
            key = hashlib.sha256((str(r.get("source_image_id")) + str(r.get("dataset"))).encode()).hexdigest()
            scored.append(((group_counts[group], bin_counts[b], key), r, frac, b, group))
        _, r, frac, b, group = min(scored, key=lambda x: x[0])
        selected.append((r, frac))
        group_counts[group] += 1
        bin_counts[b] += 1
        remaining = [(rr, ff) for rr, ff in remaining if rr is not r]
    return [r for r, _ in selected]


def copy_asset(source: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return rel(dest)


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")


def materialize_mask_tracks() -> dict[str, list[dict]]:
    mars_craters = rows(DATA / "source_index" / "mars_crater_binary_seg.jsonl")
    mars_cones = rows(DATA / "source_index" / "mars_conequest_seg.jsonl")
    mars_candidates = [r for r in mars_craters + mars_cones if r.get("split") in {"val", "test"}]
    boulder_index = DATA / "source_index" / "mars_boulder_seg.jsonl"
    boulders = rows(boulder_index) if boulder_index.is_file() else []
    if boulders and any(r.get("source_split") != "test" for r in boulders):
        raise ValueError("Boulder samples must retain the official test split")
    mars_sel = boulders + balanced_select(mars_candidates, 50 - len(boulders))

    moon_rows = rows(DATA / "source_index" / "moon_imp_seg.jsonl")
    moon_eval = [r for r in moon_rows if r.get("split") in {"val", "test"}]
    moon_train = [r for r in moon_rows if r.get("split") == "train"]
    debris_index = DATA / "source_index" / "moon_debris_nontrain.jsonl"
    debris = rows(debris_index) if debris_index.is_file() else []
    if debris and any(r.get("source_split") not in {"val", "test"} for r in debris):
        raise ValueError("Lunar debris supplement may contain only official val/test rows")
    moon_eval_selected = balanced_select(moon_eval, len(moon_eval))
    moon_eval_selected += balanced_select(debris, len(debris))
    # Fill the collection from IMP train only if independent orbital sources
    # provide fewer than 50 non-training samples; retain the official split.
    moon_train_count = max(0, 50 - len(moon_eval_selected))
    moon_sel = moon_eval_selected + balanced_select(moon_train, moon_train_count)

    output: dict[str, list[dict]] = {"mars": [], "moon": []}
    for planet, selected in (("mars", mars_sel), ("moon", moon_sel)):
        for i, r in enumerate(selected, start=1):
            source_split = r.get("source_split") or r.get("split")
            folder = OUT / planet / "segmentation" / f"{i:03d}"
            image = copy_asset(local(r["image"]), folder / ("image" + local(r["image"]).suffix.lower()))
            mask = copy_asset(local(r["mask"]), folder / ("segmentation_mask" + local(r["mask"]).suffix.lower()))
            output[planet].append({
                "sample_id": f"{planet}_detailed_mask_{i:03d}", "planet": planet,
                "task": "published_pixel_segmentation", "source_dataset": r.get("dataset"),
                "source_repo": r.get("source_repo"), "source_revision": r.get("source_revision"),
                "source_split": source_split, "evaluation_eligible": source_split in {"val", "test"},
                "final_test_eligible": source_split == "test",
                "scene_group_id": r.get("scene_group_id"), "source_image_id": r.get("source_image_id"),
                "classes": r.get("classes") or (["Background", "IMP"] if planet == "moon" else None),
                "license_claim": r.get("license_claim"),
                "foreground_fraction": sample_mask_fraction(r), "image": image, "mask": mask,
                "image_sha256": sha(local(r["image"])), "mask_sha256": sha(local(r["mask"])),
                "height_map": None,
            })
    return output


def farthest_rows(rs: list[dict], n: int, xkey: str, ykey: str, groupkey: str | None = None) -> list[dict]:
    available = [r for r in rs if r.get(xkey) is not None and r.get(ykey) is not None]
    if len(available) < n:
        raise RuntimeError(f"Only {len(available)} rows with {xkey}/{ykey}; need {n}")
    x = np.asarray([float(r[xkey]) for r in available]); y = np.asarray([float(r[ykey]) for r in available])
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-12)
    y = (y - y.min()) / max(float(y.max() - y.min()), 1e-12)
    points = np.column_stack((x, y))
    first = min(range(len(available)), key=lambda j: hashlib.sha256(str(available[j].get("sample_id")).encode()).hexdigest())
    chosen = [first]
    dist = np.sum((points - points[first]) ** 2, axis=1)
    while len(chosen) < n:
        idx = max((j for j in range(len(available)) if j not in chosen), key=lambda j: (dist[j], str(available[j].get("sample_id"))))
        chosen.append(idx)
        dist = np.minimum(dist, np.sum((points - points[idx]) ** 2, axis=1))
    if groupkey:
        # Keep deterministic farthest-point ordering; report group coverage for audit.
        pass
    return [available[j] for j in chosen]


def materialize_height_tracks() -> dict[str, list[dict]]:
    mars_all = rows(DATA / "working" / "mars" / "mmlsv2" / "manifest.jsonl")
    mars_eval = [r for r in mars_all if r.get("sample_split") == "eval"]
    mars_sel = farthest_rows(mars_eval, 50, "grid_x", "grid_y")
    moon_all = rows(DATA / "working" / "moon" / "wac_crater_height" / "manifest.jsonl")
    moon_eval = [r for r in moon_all if r.get("sample_split") in {"val", "test"}]
    # Prefer different EDR product IDs, then maximize spatial spread.
    moon_unique = {}
    for r in moon_eval:
        moon_unique.setdefault(r.get("product_id"), []).append(r)
    moon_pool = [sorted(v, key=lambda z: (z.get("row", 0), z.get("col", 0)))[len(v)//2] for v in moon_unique.values()]
    moon_sel = farthest_rows(moon_pool, 50, "center_longitude", "center_latitude")

    output: dict[str, list[dict]] = {"mars": [], "moon": []}
    for planet, selected in (("mars", mars_sel), ("moon", moon_sel)):
        for i, r in enumerate(selected, start=1):
            folder = OUT / planet / "height" / f"{i:03d}"
            imgsrc = local(r["image"]); hsrc = local(r["height_map"]); vsrc = local(r["height_valid_mask"])
            image = copy_asset(imgsrc, folder / ("image" + imgsrc.suffix.lower()))
            height = copy_asset(hsrc, folder / "height_map.npy")
            valid = copy_asset(vsrc, folder / "height_valid_mask.npy")
            output[planet].append({
                "sample_id": f"{planet}_height_{i:03d}", "planet": planet, "task": "elevation_map_reading",
                "source_dataset": r.get("dataset") or r.get("source"),
                "source_revision": r.get("revision") or r.get("source_revision"),
                "source_split": r.get("evaluation_stratum") or r.get("source_split") or r.get("sample_split"),
                "evaluation_eligible": True, "scene_group_id": r.get("product_id") or r.get("grid_x"),
                "final_test_eligible": (r.get("evaluation_stratum") or r.get("source_split")) == "test",
                "source_image_id": r.get("sample_id"), "image": image, "height_map": height,
                "height_valid_mask": valid, "height_units": r.get("height_units") or r.get("height_representation"),
                "height_sha256": sha(hsrc), "image_sha256": sha(imgsrc),
                "license_claim": r.get("license_claim") or "source dataset card CC-BY-4.0",
                "segmentation_mask": None,
            })
    return output


def materialize_train() -> dict[str, list[dict]]:
    mars = [r for r in rows(DATA / "working" / "mars" / "mmlsv2" / "manifest.jsonl") if r.get("sample_split") == "train"]
    moon = [r for r in rows(DATA / "working" / "moon" / "wac_crater_height" / "manifest.jsonl") if r.get("sample_split") == "train"]
    output: dict[str, list[dict]] = {"mars": [], "moon": []}
    for planet, selected in (("mars", mars[:50]), ("moon", moon[:50])):
        if len(selected) < 50:
            raise RuntimeError(f"Insufficient prepared {planet} train samples: {len(selected)}")
        for i, r in enumerate(selected, start=1):
            folder = OUT / planet / "train" / f"{i:03d}"
            imgsrc = local(r["image"]); msrc = local(r["segmentation_mask"]); hsrc = local(r["height_map"])
            output[planet].append({
                "sample_id": f"{planet}_train_{i:03d}", "planet": planet, "source_split": "train",
                "source_dataset": r.get("dataset") or r.get("source"),
                "source_revision": r.get("revision") or r.get("source_revision"),
                "source_image_id": r.get("sample_id"), "scene_group_id": r.get("product_id") or r.get("grid_x"),
                "image": copy_asset(imgsrc, folder / ("image" + imgsrc.suffix.lower())),
                "segmentation_mask": copy_asset(msrc, folder / ("segmentation_mask" + msrc.suffix.lower())),
                "height_map": copy_asset(hsrc, folder / "height_map.npy"),
                "height_units": r.get("height_units") or r.get("height_representation"),
                "image_sha256": sha(imgsrc), "mask_sha256": sha(msrc), "height_sha256": sha(hsrc),
                "license_claim": r.get("license_claim") or "source dataset card CC-BY-4.0",
                "model_inference_use": False,
            })
    return output


def main() -> None:
    if OUT.exists():
        marker = OUT / "selection_audit.json"
        if OUT.resolve().parent != (DATA / "working").resolve() or not marker.is_file():
            raise FileExistsError(f"Refusing to overwrite an unrecognized output: {OUT}")
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("selection_version") != "orbital-selected-v2":
            raise FileExistsError(f"Refusing to overwrite an unrecognized output: {OUT}")
        shutil.rmtree(OUT)
    masks = materialize_mask_tracks()
    heights = materialize_height_tracks()
    train = materialize_train()
    audit = {}
    for planet in ("moon", "mars"):
        write_jsonl(OUT / planet / "segmentation_manifest.jsonl", masks[planet])
        write_jsonl(OUT / planet / "height_manifest.jsonl", heights[planet])
        write_jsonl(OUT / planet / "train_manifest.jsonl", train[planet])
        audit[planet] = {
            "segmentation_selected": len(masks[planet]),
            "segmentation_evaluation_eligible": sum(r["evaluation_eligible"] for r in masks[planet]),
            "segmentation_final_test_eligible": sum(r["final_test_eligible"] for r in masks[planet]),
            "segmentation_source_splits": dict(Counter(r["source_split"] for r in masks[planet])),
            "height_evaluation_selected": len(heights[planet]),
            "height_source_splits": dict(Counter(r["source_split"] for r in heights[planet])),
            "height_final_test_eligible": sum(r["final_test_eligible"] for r in heights[planet]),
            "train_selected": len(train[planet]),
            "segmentation_sources": dict(Counter(r["source_dataset"] for r in masks[planet])),
        }
    (OUT / "selection_audit.json").write_text(json.dumps({
        "selection_version": "orbital-selected-v2", "selection_seed": "sha256-stable",
        "totals": audit, "selection_note": "Image/mask track and elevation track are separate; masks/maps are not passed to inference. Moon includes all IMP val/test samples and available LROC NAC debris/void val/test samples; IMP train is used only for any remaining rows needed to reach 50. Mars includes all four expert-polygon Boulder test examples.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
