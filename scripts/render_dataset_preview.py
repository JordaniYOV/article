"""Render image, published mask and aligned elevation examples for both planets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_orbital" / "working" / "dataset_preview.png"


def normalized_rgb(array: np.ndarray) -> Image.Image:
    output = []
    for band in range(3):
        values = np.asarray(array[..., band], dtype=np.float32)
        lo, hi = np.nanpercentile(values, (2, 98))
        output.append((np.clip((values - lo) / max(float(hi - lo), 1e-8), 0, 1) * 255).astype(np.uint8))
    return Image.fromarray(np.stack(output, axis=-1), "RGB")


def height_display(array: np.ndarray) -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(values)
    lo, hi = np.percentile(values[valid], (2, 98))
    z = np.clip((values - lo) / max(float(hi - lo), 1e-8), 0, 1)
    # Compact blue → ochre → pale relief ramp; nodata remains dark.
    stops = np.array([[30, 74, 120], [50, 145, 147], [218, 177, 91], [250, 239, 204]], dtype=np.float32)
    pos = z * (len(stops) - 1)
    lo_i = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    frac = (pos - lo_i)[..., None]
    rgb = stops[lo_i] * (1 - frac) + stops[lo_i + 1] * frac
    rgb[~valid] = (18, 23, 31)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def mask_display(path: Path) -> Image.Image:
    with Image.open(path) as im:
        mask = np.asarray(im, dtype=np.uint8)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[:] = (25, 31, 40)
    rgb[mask > 0] = (255, 180, 30)
    return Image.fromarray(rgb, "RGB")


def select_sample(manifest: Path, preferred_ratio: tuple[float, float]) -> dict:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    def foreground(row: dict) -> int:
        if "foreground_pixels" in row:
            return int(row["foreground_pixels"])
        with Image.open(ROOT / row["segmentation_mask"]) as im:
            return int((np.asarray(im) > 0).sum())
    scored = [(foreground(row), row) for row in rows]
    pixels = 512 * 512 if "foreground_pixels" in rows[0] else 128 * 128
    eligible = [(count, row) for count, row in scored if preferred_ratio[0] <= count / pixels <= preferred_ratio[1]]
    return max(eligible or scored, key=lambda item: item[0])[1]


def panel_image(row: dict, planet: str) -> tuple[Image.Image, Image.Image, Image.Image]:
    if planet == "moon":
        image = normalized_rgb(tifffile.imread(ROOT / row["image"]))
    else:
        with Image.open(ROOT / row["image"]) as tif:
            pages = []
            for i in range(tif.n_frames):
                tif.seek(i)
                pages.append(np.asarray(tif, dtype=np.float32))
        image = normalized_rgb(np.stack(pages, axis=0))
    mask = mask_display(ROOT / row["segmentation_mask"])
    height = height_display(np.load(ROOT / row["height_map"], allow_pickle=False))
    return image, mask, height


def main() -> None:
    mars = select_sample(ROOT / "data_orbital/working/mars/mmlsv2/manifest.jsonl", (0.03, 0.75))
    moon = select_sample(ROOT / "data_orbital/working/moon/wac_crater_height/manifest.jsonl", (0.03, 0.40))
    mars_panels = panel_image(mars, "mars")
    moon_panels = panel_image(moon, "moon")
    panels = [
        ("MARS · orbital RGB display", mars_panels[0]),
        ("MARS · published landslide mask", mars_panels[1]),
        ("MARS · DEM band 4 (source scale)", mars_panels[2]),
        ("MOON · WAC visible RGB display", moon_panels[0]),
        ("MOON · crater polygons rasterized", moon_panels[1]),
        ("MOON · DTM on WAC 100 m grid", moon_panels[2]),
    ]
    tile_w, tile_h, gap = 460, 340, 20
    canvas = Image.new("RGB", (3 * tile_w + 4 * gap, 2 * tile_h + 3 * gap), "#0b0e12")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for i, (label, image) in enumerate(panels):
        row, col = divmod(i, 3)
        x, y = gap + col * (tile_w + gap), gap + row * (tile_h + gap)
        draw.text((x, y), label, fill="#ecf1f8", font=font)
        image = image.convert("RGB")
        area = (tile_w, tile_h - 38)
        image.thumbnail(area, Image.Resampling.NEAREST)
        if image.width < area[0] and image.height < area[1]:
            scale = min(area[0] / image.width, area[1] / image.height)
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.NEAREST)
        canvas.paste(image, (x + (area[0] - image.width) // 2,
                             y + 32 + (area[1] - image.height) // 2))
    canvas.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
