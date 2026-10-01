import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from linear_head_approach.encoder_probe import SOURCE, build_wac_crater_probe


class EncoderProbeManifestTests(unittest.TestCase):
    def make_collection(self, root):
        base = root / "data_orbital" / "orbital_300_v3" / "moon"
        train, evaluation, height = [], [], []
        coco = {split: {"images": [], "annotations": []} for split in ("val", "test")}
        transform = {"method": "per_tile_visible_rgb_percentile_v1",
                     "bands_zero_based": [0, 1, 2],
                     "source_raster_sha256": "a" * 64}
        for index in range(120):
            split = ("train" if index < 50 else "val" if index < 65 else "test" if index < 70
                     else "val" if index < 94 else "test")
            folder = base / split / f"{index:03d}"
            folder.mkdir(parents=True)
            image_buffer = io.BytesIO()
            Image.new("RGB", (8, 8), (index, 0, 0)).save(image_buffer, format="PNG")
            image_bytes = image_buffer.getvalue()
            mask_bytes = f"mask-{index}".encode()
            image = folder / "image.png"
            mask = folder / "mask.png"
            image.write_bytes(image_bytes)
            mask.write_bytes(mask_bytes)
            relative_image = image.relative_to(root).as_posix()
            relative_mask = mask.relative_to(root).as_posix()
            row = {"sample_id": f"wac_{index}", "source_dataset": SOURCE,
                   "source_revision": "b" * 40, "source_split": split,
                   "source_image_id": index, "scene_group_id": f"scene_{index}",
                   "image": relative_image, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                   "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
                   "display_transform": transform}
            if index < 50:
                row.update(track="train", evaluation_eligible=False,
                           segmentation_mask=relative_mask,
                           segmentation_mask_sha256=row["mask_sha256"])
                train.append(row)
            elif index < 70:
                row.update(evaluation_eligible=True, mask=relative_mask,
                           mask_semantics="catalog_crater_polygon")
                evaluation.append(row)
                coco[split]["images"].append({"id": index, "file_name": f"seg_tile_{index}.tif",
                                                "height": 8, "width": 8})
            else:
                stem = f"tile_{index}"
                row.update(track="height", evaluation_eligible=True,
                           segmentation_mask=None,
                           source_image_id=f"moon_wac_{split}_{stem}")
                height.append(row)
                coco[split]["images"].append({"id": index, "file_name": f"{stem}.tif",
                                                "height": 8, "width": 8})
                coco[split]["annotations"].append({"image_id": index,
                                                     "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]]})
        (base / "train_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
        (base / "segmentation_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in evaluation))
        (base / "height_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in height))
        native = root / "data_orbital" / "encoder_probe_wac_v1" / "native_vis"
        native.mkdir(parents=True)
        for split, content in coco.items():
            (native / f"{split}_coco.json").write_text(json.dumps(content))
        return base, evaluation

    def test_targets_never_appear_in_encoder_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_collection(root)
            inputs, targets, audit = build_wac_crater_probe(root)
            self.assertEqual(audit["split_counts"], {"train": 50, "val": 39, "test": 31})
            self.assertEqual(len(inputs), len(targets))
            self.assertTrue(all("mask" not in json.dumps(row) for row in inputs))
            self.assertTrue(all("mask" in row for row in targets))

    def test_scene_group_may_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, evaluation = self.make_collection(root)
            evaluation[0]["scene_group_id"] = "scene_0"
            (base / "segmentation_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in evaluation))
            with self.assertRaisesRegex(ValueError, "Scene group crosses splits"):
                build_wac_crater_probe(root)


class SpatialProbeTests(unittest.TestCase):
    def test_spatial_signal_is_learned_without_test_data(self):
        import numpy as np
        from linear_head_approach.linear_probe import fit, image_iou, choose_threshold

        features, masks = [], []
        for offset in range(4):
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, offset:offset + 3] = 1
            signal = mask.reshape(4, 2, 4, 2).mean(axis=(1, 3))
            features.append(np.stack((signal, np.ones_like(signal) * offset), axis=-1))
            masks.append(mask)
        probe = fit(features[:3], masks[:3], grid=(4, 4), regularization=0.01)
        threshold, val_score = choose_threshold(probe, features[3:], masks[3:])
        self.assertGreater(val_score, 0.5)
        self.assertGreater(image_iou(probe.predict_grid(features[3]), masks[3], threshold), 0.5)

    def test_uncertainty_resamples_scenes_not_pixels(self):
        from linear_head_approach.linear_probe import grouped_mean_interval

        first = grouped_mean_interval([0.1, 0.1, 0.9], ["scene_a", "scene_a", "scene_b"], seed=7)
        second = grouped_mean_interval([0.1, 0.1, 0.9], ["scene_a", "scene_a", "scene_b"], seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 0.1)
        self.assertGreaterEqual(first[1], 0.9)

    def test_paired_comparison_uses_matching_samples(self):
        from linear_head_approach.linear_probe import paired_group_difference

        a = {"one": {"iou": 0.7}, "two": {"iou": 0.5}, "three": {"iou": 0.9}}
        b = {"one": {"iou": 0.6}, "two": {"iou": 0.3}, "three": {"iou": 0.8}}
        groups = {"one": "A", "two": "A", "three": "B"}
        result = paired_group_difference(a, b, groups)
        self.assertAlmostEqual(result["mean_iou_difference_a_minus_b"], 0.13333333333333333)
        self.assertEqual(result["samples"], 3)
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_group_difference(a, {"one": b["one"]}, groups)


if __name__ == "__main__":
    unittest.main()
