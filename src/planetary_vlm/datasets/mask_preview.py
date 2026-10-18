"""Deterministic display-only previews of published orbital masks, not new labels."""

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re

from planetary_vlm.datasets.orbital_questions import orbital_path
from planetary_vlm.io import read_jsonl, write_json

PALETTE = ((26, 31, 40), (255, 160, 35), (65, 155, 255), (75, 210, 110), (235, 90, 170))


def render(image, mask, classes, sample_id):
    """Render discrete class IDs, keeping background unchanged in the overlay."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    rgb = np.asarray(image.convert("RGB"))
    labels = np.asarray(mask)
    if labels.ndim != 2 or labels.shape != rgb.shape[:2] or labels.dtype.kind not in "iu":
        raise ValueError("Image/mask dimensions or class-ID dtype differ")
    values = np.unique(labels).tolist()
    if min(values) < 0 or max(values) >= len(classes) or max(values) >= len(PALETTE):
        raise ValueError("Mask class IDs do not match the source-local taxonomy")
    colours = np.asarray(PALETTE, dtype=np.uint8)[labels]
    overlay = rgb.copy()
    foreground = labels != 0
    overlay[foreground] = np.rint(rgb[foreground] * 0.55 + colours[foreground] * 0.45).astype(np.uint8)
    width, height = image.size
    gap, header = 12, 56
    canvas = Image.new("RGB", (3 * width + 4 * gap, height + header + 48 + 24 * len(values)), PALETTE[0])
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arial.ttf", 19)
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        title_font = font = ImageFont.load_default()
    draw.text((gap, 8), sample_id + " | published labels | display only", fill="white", font=title_font)
    for index, (caption, pixels) in enumerate((("Original photograph", rgb), ("Colour-coded mask", colours), ("Overlay (45%)", overlay))):
        x = gap + index * (width + gap)
        draw.text((x, 34), caption, fill="white", font=font)
        canvas.paste(Image.fromarray(pixels), (x, header))
    y = header + height + 12
    draw.text((gap, y), "Class IDs are source-local; background is not verified absence.", fill="white", font=font)
    for index, value in enumerate(values):
        ly = y + 27 + index * 24
        draw.rectangle((gap, ly, gap + 15, ly + 15), fill=PALETTE[value], outline="white")
        draw.text((gap + 24, ly), f"{value}: {classes[value]}", fill="white", font=font)
    return canvas


def build(root, output):
    from PIL import Image
    root, output = Path(root).resolve(), Path(output).resolve()
    if (not output.is_relative_to(root / "data_orbital") or output == root / "data_orbital"
            or output.is_relative_to(root / "data_orbital" / "orbital_300_v3")
            or "data_rover" in {p.casefold() for p in root.parts + output.parts}):
        raise ValueError("Use a new preview folder under data_orbital, outside the source collection")
    # Validate symlink resolution and protected scope before reading manifests/assets.
    output = orbital_path(root, output.relative_to(root).as_posix())
    if output.exists():
        raise FileExistsError("Preview already exists; use a new version, never overwrite sources")
    rows, seen = [], set()
    for planet in ("moon", "mars"):
        manifest = orbital_path(root, f"data_orbital/orbital_300_v3/{planet}/segmentation_manifest.jsonl")
        for row in read_jsonl(manifest):
            sid = row["sample_id"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sid) or sid in seen:
                raise ValueError("Unsafe or duplicate sample ID")
            seen.add(sid)
            assets = {}
            for field in ("image", "mask"):
                path = orbital_path(root, row[field])
                data = path.read_bytes()
                if sha256(data).hexdigest() != row[field + "_sha256"]:
                    raise ValueError("Source asset changed: " + sid + "/" + field)
                assets[field] = path
            rows.append((row, assets))
    output.mkdir(parents=True, exist_ok=False)
    (output / "panels").mkdir()
    sections, provenance = [], []
    for row, assets in rows:
        name = "panels/" + row["sample_id"] + ".png"
        with Image.open(assets["image"]) as original, Image.open(assets["mask"]) as mask:
            render(original, mask, row["classes"], row["sample_id"]).save(output / name)
        # Preserve and verify original encoded PNGs, not merely their pixel arrays.
        for field, path in assets.items():
            if sha256(path.read_bytes()).hexdigest() != row[field + "_sha256"]:
                raise ValueError("Source changed during display rendering")
        caption = escape(f"{row['sample_id']} | {row['source_split']} | {row['source_dataset']}")
        sections.append(f'<section><h2>{caption}</h2><a href="{name}"><img loading="lazy" src="{name}" alt="{caption}"></a></section>')
        provenance.append({"sample_id": row["sample_id"], "image_sha256": row["image_sha256"],
            "mask_sha256": row["mask_sha256"], "classes": row["classes"], "preview": name})
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Orbital mask previews</title>'
        '<style>body{background:#141820;color:#eee;font:16px sans-serif;margin:24px}section{margin:36px 0}'
        'h2{font-size:18px}img{width:100%;height:auto;max-width:1800px}a{color:#ffc16b}</style>'
        '<h1>Исходные семантические маски: 100 снимков</h1>'
        '<p>Слева — фото, в центре — цветная маска, справа — наложение. Нажми на изображение для полного размера.</p>'
        '<p>Ctrl+F: поиск sample_id или датасета. Цвета обозначают локальные классы источника; легенда под каждой панелью.</p>'
        '<p>Это только визуализация существующей разметки. Фон не доказывает отсутствие кратеров; новые объекты не добавлены.</p>'
        + "".join(sections) + '</html>')
    (output / "index.html").write_text(page, encoding="utf-8")
    write_json(output / "provenance.json", {"display_only": True, "new_annotations": False, "sources": provenance})
    return {"images": len(rows), "gallery": str(output / "index.html"), "source_files_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.workspace, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
