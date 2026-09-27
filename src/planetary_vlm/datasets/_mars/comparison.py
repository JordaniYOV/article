"""Render existing stereo candidates; never modify or admit benchmark data."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


from planetary_vlm.datasets.base import PROJECT_ROOT as ROOT
SCALE = (2.6, 3.3)


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def palette(values):
    encoded = (np.clip((values - SCALE[0]) / (SCALE[1] - SCALE[0]), 0, 1) * 255)
    encoded = np.nan_to_num(encoded, nan=0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(encoded, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(data_root=None, output=None):
    data_root = Path(data_root) if data_root is not None else ROOT / "data"
    OUT = Path(output) if output is not None else ROOT / "outputs/mastcam_depth_comparison_v1"
    OUT.mkdir(parents=True, exist_ok=False)
    results = []
    alignment = Image.new("RGB", (1712, 1150), "white")
    ad = ImageDraw.Draw(alignment)
    for row, (sol, version) in enumerate(((703, "v4"), (788, "v2"))):
        folder = data_root / f"interim/mastcam_stereo_sol{sol}_{version}"
        report = json.loads((folder / "report.json").read_text())
        product = report["right_product"]
        source_path = data_root / f"source_audit_v1/mastcam_stereo_pilot/{product.lower()}.png"
        source = Image.open(source_path).convert("RGB")
        range_path = folder / "right_source_range_m.npy"
        ranges = np.load(range_path)
        valid = np.load(folder / "right_source_validity.npy")
        rgb = np.asarray(source)
        assert ranges.shape == valid.shape == rgb.shape[:2]
        assert np.array_equal(valid, np.isfinite(ranges))
        colors = palette(ranges)
        colors[~valid] = (55, 55, 55)
        overlay = rgb.copy()
        overlay[valid] = (0.55 * rgb[valid] + 0.45 * colors[valid]).astype(np.uint8)
        overlay[~valid] = (0.6 * rgb[~valid] + 0.4 * np.array([220, 0, 180])).astype(np.uint8)
        canvas = Image.new("RGB", (1584, 674), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"Sol {sol} | Right Mastcam | valid: {valid.mean():.1%}", fill="black", font=font(27))
        for col, (label, panel) in enumerate((
            ("Original RGB", source),
            ("Stereo range (m)", Image.fromarray(colors)),
            ("RGB + range overlay", Image.fromarray(overlay)),
        )):
            x = 12 + col * 524
            draw.text((x, 47), label, fill="black", font=font(24))
            canvas.paste(panel.resize((512, 512), Image.Resampling.NEAREST if col else Image.Resampling.LANCZOS), (x, 80))
        ramp = np.linspace(*SCALE, 1000)[None, :].repeat(18, axis=0)
        canvas.paste(Image.fromarray(palette(ramp)), (12, 610))
        for value in np.linspace(*SCALE, 8):
            x = 12 + int((value - SCALE[0]) / (SCALE[1] - SCALE[0]) * 1000)
            draw.text((min(x, 975), 631), f"{value:.1f}", fill="black", font=font(19))
        draw.text((1050, 607), "Gray / magenta: invalid", fill="black", font=font(20))
        draw.text((1050, 638), "Candidate, not depth GT", fill="black", font=font(20))
        canvas.save(OUT / f"sol{sol}_rgb_range_overlay.png")

        # A previously identified hypothesis only: no geometric certification.
        bench_path = data_root / f"interim/mars_bench_msl_v1/test/images/cr_{sol}_{product}.jpg"
        bench = Image.open(bench_path).convert("RGB")
        hypothesis = source.resize((560, 560), Image.Resampling.BILINEAR).crop((0, 30, 560, 530))
        delta = np.abs(np.asarray(bench, dtype=np.float32) - np.asarray(hypothesis, dtype=np.float32))
        amplified = Image.fromarray(np.clip(delta * 10, 0, 255).astype(np.uint8))
        y = row * 565
        ad.text((12, y + 4), f"Sol {sol}: Mars-Bench registration hypothesis | RGB MAE = {delta.mean():.2f}/255", fill="black", font=font(23))
        for col, (label, panel) in enumerate((("Mars-Bench RGB", bench), ("Source resized + center crop", hypothesis), ("Absolute difference x10", amplified))):
            x = 8 + col * 568
            ad.text((x, y + 34), label, fill="black", font=font(21))
            alignment.paste(panel, (x, y + 64))
        results.append({"sol": sol, "range_definition": "Euclidean camera-to-surface distance in metres", "valid_fraction_native_right": float(valid.mean()), "native_valid_range_percentiles_5_50_95_m": np.percentile(ranges[valid], [5, 50, 95]).tolist(), "display_range_m": list(SCALE), "display_clipped_fraction_valid": float(((ranges[valid] < SCALE[0]) | (ranges[valid] > SCALE[1])).mean()), "benchmark_hypothesis": {"resize": [560, 560], "crop": [0, 30, 560, 530], "rgb_mae_0_255": float(delta.mean()), "status": "not_certified"}, "source_sha256": sha(source_path), "range_sha256": sha(range_path), "benchmark_sha256": sha(bench_path)})
    alignment.save(OUT / "benchmark_alignment_comparison.png")
    (OUT / "comparison.json").write_text(json.dumps({"status": "visual_qa_only_not_metric_validation", "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return OUT / "comparison.json"
