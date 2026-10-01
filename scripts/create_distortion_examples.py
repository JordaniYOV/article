"""Create orbital_300_v3 transformation previews with provenance.

Default: five transformed images for each of two scenes (Mars and Moon).
--with-masks: five image/mask pairs per selected planet.
--planets moon: append the lunar examples without overwriting the Mars set.
--refresh: regenerate the complete existing set under the current protocol.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planetary_vlm.distortion import (
    DISTORTION_VERSION, add_radiation_artifacts, apply_hard_shadows,
    apply_lens_flare, apply_synchronous_shear, spoil_image,
)

DATA = ROOT / "data_orbital" / "orbital_300_v3"
OUTPUT = ROOT / "src" / "planetary_vlm" / "distortion" / "example"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(planet):
    manifest = DATA / planet / "segmentation_manifest.jsonl"
    with manifest.open(encoding="utf-8") as stream:
        row = json.loads(next(line for line in stream if line.strip()))
    arrays = {}
    for key in ("image", "mask"):
        path = (ROOT / row[key]).resolve()
        if DATA.resolve() not in path.parents:
            raise ValueError(f"Source escaped orbital_300_v3: {path}")
        if sha256(path) != row[f"{key}_sha256"]:
            raise ValueError(f"Source fingerprint mismatch: {path}")
        with Image.open(path) as opened:
            arrays[key] = np.array(opened)
    if arrays["mask"].shape != arrays["image"].shape[:2]:
        raise ValueError("Source mask is not aligned")
    return row, arrays["image"], arrays["mask"]


def variants(image, mask, seed):
    shear = {"max_shift_px": 4.0, "control_spacing_px": 32, "scan_axis": "both",
             "seed": seed, "seg_ignore_label": 255}
    radiation = {"num_hits": 80, "max_streak_length": 10, "strength": 1.0, "seed": seed}
    flare = {"strength": 0.55, "radius": 0.3, "num_ghosts": 3, "seed": seed}
    shadows = {"strength": 0.65, "num_shadows": 1, "seed": seed}
    combined = {"max_shift_px": 4.0, "control_spacing_px": 32, "scan_axis": "both",
                "num_hits": 80, "max_streak_length": 10,
                "flare_strength": 0.55, "shadow_strength": 0.65,
                "seg_ignore_label": 255, "seed": seed}
    result, _, target = apply_synchronous_shear(image, seg_mask=mask, **shear)
    yield "shear", result, target, shear
    yield "radiation", add_radiation_artifacts(image, **radiation), mask.copy(), radiation
    yield "lens_flare", apply_lens_flare(image, **flare), mask.copy(), flare
    yield "hard_shadows", apply_hard_shadows(image, **shadows), mask.copy(), shadows
    result, _, target = spoil_image(image, seg_mask=mask, **combined)
    yield "all", result, target, combined


def save_image(path, array, *, overwrite=False):
    with path.open("wb" if overwrite else "xb") as stream:
        Image.fromarray(array).save(stream, format="PNG")
    with Image.open(path) as opened:
        actual = np.array(opened)
    np.testing.assert_array_equal(actual, array)


def save_mask(path, mask, *, overwrite=False):
    # P mode preserves class IDs and supplies a palette for readable previews.
    image = Image.fromarray(mask).convert("P")
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[:5] = [(0, 0, 0), (255, 150, 0), (70, 150, 255), (80, 210, 100), (230, 80, 160)]
    palette[255] = (180, 180, 180)
    image.putpalette(palette.ravel().tolist())
    with path.open("wb" if overwrite else "xb") as stream:
        image.save(stream, format="PNG")
    with Image.open(path) as opened:
        np.testing.assert_array_equal(np.array(opened), mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-masks", action="store_true")
    parser.add_argument("--planets", nargs="+", choices=("mars", "moon"), default=["mars", "moon"])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if len(set(args.planets)) != len(args.planets):
        parser.error("Select each planet only once")

    metadata = {"demo_only": True, "distortion_version": DISTORTION_VERSION,
                "numpy_version": np.__version__, "opencv_version": cv2.__version__,
                "seed": args.seed, "layout": "image_mask_pairs" if args.with_masks else "images_only",
                "sources": [], "outputs": [],
                "combined_order": ["shear", "hard_shadows", "lens_flare", "radiation"]}
    manifest_path = OUTPUT / "manifest.json"
    recorded = set()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not manifest_path.is_file():
            raise ValueError("Existing examples have no provenance manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("demo_only", "distortion_version", "numpy_version", "opencv_version",
                    "seed", "layout", "combined_order"):
            if previous.get(key) != metadata[key] and not args.refresh:
                raise ValueError(f"Existing example protocol differs: {key}")
        for row in previous["outputs"]:
            name = row["file"]
            if Path(name).name != name or name in recorded:
                raise ValueError("Invalid or duplicate output filename in existing manifest")
            recorded.add(name)
            path = OUTPUT / name
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError(f"Existing output fingerprint mismatch: {path}")
        if recorded != {path.name for path in OUTPUT.glob("*.png")}:
            raise ValueError("Existing PNG files differ from the provenance manifest")
        if not args.refresh:
            metadata = previous

    previews = []
    sources = []
    for planet in args.planets:
        row, image, mask = source(planet)
        original_image, original_mask = image.copy(), mask.copy()
        sources.append(row)
        for index, (name, result, target, parameters) in enumerate(variants(image, mask, args.seed), 1):
            previews.append((f"{planet}_{index:02d}_{name}_image.png", result, False,
                             row["sample_id"], name, parameters))
            if args.with_masks:
                previews.append((f"{planet}_{index:02d}_{name}_segmentation.png", target, True,
                                 row["sample_id"], name, parameters))
        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(mask, original_mask)
    expected_count = 5 * len(args.planets) * (2 if args.with_masks else 1)
    if len(previews) != expected_count:
        raise ValueError(f"Expected {expected_count} new preview images")
    planned = {row[0] for row in previews}
    if args.refresh and recorded and planned != recorded:
        raise ValueError("Refresh must regenerate the complete existing PNG set")
    for filename, *_ in previews:
        if (OUTPUT / filename).exists() and not args.refresh:
            raise FileExistsError(f"Refusing to overwrite existing example: {filename}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for filename, array, is_mask, sample_id, transform, parameters in previews:
        path = OUTPUT / filename
        (save_mask if is_mask else save_image)(path, array, overwrite=args.refresh)
        rows.append({"file": filename, "sha256": sha256(path), "source_sample_id": sample_id,
                     "transform": transform, "parameters": parameters,
                     "kind": "segmentation" if is_mask else "image",
                     "shape": list(array.shape), "dtype": str(array.dtype)})
    metadata["sources"].extend(sources)
    metadata["outputs"].extend(rows)
    temporary_manifest = OUTPUT / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    count = len(list(OUTPUT.glob("*.png")))
    if count != len(metadata["outputs"]):
        raise ValueError(f"PNG count differs from manifest: {count}")
    print(json.dumps({"directory": str(OUTPUT), "png_count": count,
                      "sample_ids": [row["sample_id"] for row in metadata["sources"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
