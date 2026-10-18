"""Inventory source-labelled Buried regions. Components are NOT crater instances."""

import argparse
from collections import Counter
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
import re
import tomllib

from planetary_vlm.datasets.orbital_questions import orbital_path
from planetary_vlm.datasets.mask_preview import render
from planetary_vlm.io import read_jsonl, write_json

SOURCE = "Mirali33/mb-crater_multi_seg"
SOURCE_URL = "https://zenodo.org/records/15494713"


def components(mask, class_id, *, size_fraction=.04, margin=1):
    import cv2
    import numpy as np
    values = np.asarray(mask)
    if values.ndim != 2 or values.dtype.kind not in "iu":
        raise ValueError("Expected a two-dimensional class-ID mask")
    binary = (values == class_id).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    threshold = math.ceil(size_fraction * min(values.shape))
    other = (values > 0) & (values != class_id)
    adjacent = cv2.dilate(other.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rows = []
    for label in range(1, count):
        x, y, w, h, area = (int(n) for n in stats[label])
        rows.append({"mask_bbox_xywh": [x, y, w, h], "mask_pixels": area,
            "bbox_midpoint_xy_proxy": [x + w / 2, y + h / 2],
            "size_threshold_px": threshold, "mask_bbox_passes_size_proxy": min(w, h) >= threshold,
            "mask_bbox_within_frame_margin": x >= margin and y >= margin and
                x + w <= values.shape[1] - margin and y + h <= values.shape[0] - margin,
            "touches_different_foreground_class": bool(adjacent[labels == label].any()),
            "needs_instance_and_geometry_review": True, "reference_admitted": False})
    rows.sort(key=lambda row: (row["mask_bbox_xywh"][1], row["mask_bbox_xywh"][0], row["mask_pixels"]))
    return rows


def build(root, output):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    root, output = Path(root).resolve(), Path(output).resolve()
    if (not output.is_relative_to(root / "data_orbital") or output == root / "data_orbital"
            or output.is_relative_to(root / "data_orbital" / "orbital_300_v3")
            or "data_rover" in {p.casefold() for p in root.parts + output.parts}):
        raise ValueError("Use a new audit folder under data_orbital, outside the source collection")
    output = orbital_path(root, output.relative_to(root).as_posix())
    if output.exists():
        raise FileExistsError("Audit already exists; use a new version")
    config_path = root / "configs" / "crater_annotation_v1.toml"
    rules = tomllib.loads(config_path.read_text(encoding="utf-8"))
    source_rows, manifests = [], {}
    for planet in ("moon", "mars"):
        relative = f"data_orbital/orbital_300_v3/{planet}/segmentation_manifest.jsonl"
        path = orbital_path(root, relative)
        manifests[relative] = sha256(path.read_bytes()).hexdigest()
        source_rows.extend(read_jsonl(path))
    seen, inventory, candidates, panels = set(), [], [], []
    for row in source_rows:
        sid = row["sample_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sid) or sid in seen:
            raise ValueError("Unsafe or duplicate sample ID")
        seen.add(sid)
        mask_path = orbital_path(root, row["mask"])
        if sha256(mask_path.read_bytes()).hexdigest() != row["mask_sha256"]:
            raise ValueError("Changed source mask: " + sid)
        with Image.open(mask_path) as source_mask:
            mask = np.array(source_mask)
        classes = row["classes"]
        if mask.ndim != 2 or mask.dtype.kind not in "iu" or mask.max() >= len(classes):
            raise ValueError("Mask does not match source-local classes: " + sid)
        encoded = row["source_dataset"] == SOURCE and classes == ["Background", "Other", "Layered", "Buried", "Secondary"]
        inventory.append({"sample_id": sid, "source_split": row["source_split"],
            "source_dataset": row["source_dataset"], "mask_sha256": row["mask_sha256"],
            "buried_type_encoded": encoded, "buried_pixels": int((mask == 3).sum()) if encoded else None,
            "negative_buried_reference": False})
        if not encoded:
            continue  # No subtype means unknown, NOT absence of Buried craters.
        image_path = orbital_path(root, row["image"])
        if sha256(image_path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError("Changed source image: " + sid)
        proposals = components(mask, 3, size_fraction=rules["objects"]["minimum_short_side_fraction"],
                               margin=rules["visibility"]["image_border_margin_px"])
        with Image.open(image_path) as original, Image.open(mask_path) as source_mask:
            if original.size != source_mask.size:
                raise ValueError("Unaligned source image/mask")
            panel = render(original, source_mask, classes, sid + " | Buried component proposals")
            width = original.width
        draw = ImageDraw.Draw(panel)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        overlay_x, overlay_y = 12 + 2 * (width + 12), 56
        for number, proposal in enumerate(proposals, 1):
            key = f"{sid}__buried_component_{number:03d}"
            candidate = {"candidate_id": key, "candidate_kind": "8_connected_Buried_mask_component_NOT_instance",
                "source_sample_id": sid, "source_class_id": 3, "source_class": "Buried",
                **{k: row[k] for k in ("planet", "source_dataset", "source_revision", "source_split",
                    "source_image_id", "scene_group_id", "license_claim", "image", "mask", "image_sha256", "mask_sha256")},
                **proposal, "preview": f"panels/{sid}.png"}
            candidates.append(candidate)
            x, y, w, h = proposal["mask_bbox_xywh"]
            draw.rectangle((overlay_x + x, overlay_y + y, overlay_x + x + w - 1, overlay_y + y + h - 1),
                           outline=(255, 240, 90), width=2)
            label_y = max(overlay_y, overlay_y + y - 19)
            draw.text((overlay_x + x, label_y), f"B{number:02d}", fill=(255, 240, 90), font=font,
                      stroke_width=1, stroke_fill="black")
        panels.append((row, panel, proposals))
    output.mkdir(parents=True, exist_ok=False)
    (output / "panels").mkdir()
    sections = []
    for row, panel, proposals in panels:
        sid = row["sample_id"]
        path = "panels/" + sid + ".png"
        panel.save(output / path)
        table_rows = "".join(f"<tr><td>B{i:02d}</td><td>{p['mask_bbox_xywh']}</td><td>{p['mask_pixels']}</td>"
            f"<td>{p['mask_bbox_passes_size_proxy']}</td><td>{p['mask_bbox_within_frame_margin']}</td>"
            f"<td>{p['touches_different_foreground_class']}</td></tr>" for i, p in enumerate(proposals, 1))
        sections.append(f"<section><h2>{escape(sid)} — {len(proposals)} компонентов (не подтверждённых кратеров)</h2>"
            f'<a href="{path}"><img src="{path}" loading="lazy" alt="{escape(sid)}"></a>'
            "<table><tr><th>ID</th><th>bbox [x,y,w,h]</th><th>Пикселей</th><th>Размерный proxy</th>"
            f"<th>Внутри кадра</th><th>Контакт с другим классом</th></tr>{table_rows}</table></section>")
    summary = {"schema": "buried-mask-audit-v1", "source_url": SOURCE_URL,
        "examined_semantic_images": len(inventory), "subtype_encoded_images": len(panels),
        "images_with_buried_pixels": sum(bool(p["buried_pixels"]) for p in inventory if p["buried_type_encoded"]),
        "buried_component_candidates": len(candidates), "confirmed_instances": None,
        "size_proxy_passing_candidates": sum(c["mask_bbox_passes_size_proxy"] for c in candidates),
        "size_and_frame_proxy_passing_candidates": sum(c["mask_bbox_passes_size_proxy"] and c["mask_bbox_within_frame_margin"] for c in candidates),
        "touching_different_class_candidates": sum(c["touches_different_foreground_class"] for c in candidates),
        "source_manifests_sha256": manifests, "annotation_rules_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "converter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": {"numpy": np.__version__, "opencv": cv2.__version__, "Pillow": Image.__version__},
        "scientific_result": False, "new_annotations": False, "source_files_unchanged": True,
        "limitations": ["8-connected components may merge multiple instances or split one crater.",
            "Other masks have no Buried subtype; they are not negative subtype references.",
            "Mask bbox is a proposal, not necessarily the full geological crater extent.",
            "Contact with another class is not evidence of superposition or which object lies above.",
            "No inferred hidden rims, no QA targets, no changes to annotation thresholds or source splits."]}
    # Recheck source bytes before claiming preservation, including images used in panels.
    for row in source_rows:
        for field in ("mask", "image"):
            if sha256(orbital_path(root, row[field]).read_bytes()).hexdigest() != row[field + "_sha256"]:
                raise ValueError("Source changed during audit")
    write_json(output / "candidates.json", candidates)
    write_json(output / "samples.json", inventory)
    write_json(output / "summary.json", summary)
    page = ("<!doctype html><html lang='ru'><meta charset='utf-8'><title>Buried mask candidates</title>"
        "<style>body{background:#171c25;color:#eee;font:16px sans-serif;margin:24px}img{width:100%;max-width:1800px}"
        "td,th{padding:8px;border:1px solid #555}table{border-collapse:collapse}section{margin:40px 0}</style>"
        "<h1>Buried: области исходных масок для ревью</h1>"
        f"<p>{len(candidates)} компонентов на {len(panels)} снимках с явной типовой разметкой.</p>"
        "<p>Зелёный — source class Buried. B01… — кандидаты, не подтверждённые экземпляры. "
        "Рамки вычислены по пикселям класса 3, не достроены по фотографии.</p>"
        "<p>Buried не равно обрезанный кадром кратер или автоматически найденное наложение. "
        "Остальные источники без типа Buried не являются отрицательными примерами.</p>" + "".join(sections) + "</html>")
    (output / "index.html").write_text(page, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.workspace, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
