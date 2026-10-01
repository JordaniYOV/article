"""Synthetic fixtures for geometry alignment, nodata and seeded corruptions."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import cv2
    import numpy as np
    from planetary_vlm.distortion import (
        add_radiation_artifacts, apply_hard_shadows, apply_lens_flare,
        apply_synchronous_shear, spoil_image,
    )
except ImportError:
    cv2 = np = None


@unittest.skipIf(np is None, "NumPy/OpenCV optional dependencies are not installed")
class DistortionTests(unittest.TestCase):
    def test_line_jitter_preserves_image_segmentation_alignment(self):
        h, w = 64, 96
        y, x = np.mgrid[:h, :w]
        # Each channel records its source coordinate; each label records pixel ID.
        image = np.stack((x / (w - 1), y / (h - 1), np.full((h, w), 0.5)), axis=2)
        seg = (y * w + x).astype(np.uint16)
        original_image, original_seg = image.copy(), seg.copy()
        output, dem, target = apply_synchronous_shear(
            image, 3, seg_mask=seg, seg_ignore_label=65535, control_spacing_px=16, seed=42)
        self.assertIsNone(dem)
        source_x, source_y = target.astype(int) % w, target.astype(int) // w
        inside = ((target != 65535) & (source_x > 0) & (source_x < w - 1)
                  & (source_y > 0) & (source_y < h - 1))
        self.assertLessEqual(np.max(np.abs(output[..., 0][inside] * (w - 1) - source_x[inside])), 0.52)
        self.assertLessEqual(np.max(np.abs(output[..., 1][inside] * (h - 1) - source_y[inside])), 0.52)
        self.assertIn(65535, np.unique(target))
        self.assertFalse(np.array_equal(output, image))
        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(seg, original_seg)

    def test_jitter_field_is_smooth_bounded_and_has_no_folds(self):
        from planetary_vlm.distortion.geometry import _jitter_maps
        h, w = 256, 192
        map_x, map_y = _jitter_maps(h, w, 6, 32, 42, "both")
        dx = 80 - map_x[:, 80]
        dy = 80 - map_y[80, :]
        for profile in (dx, dy):
            self.assertLessEqual(np.max(np.abs(profile)), 6.0001)
            self.assertLessEqual(np.max(np.abs(np.diff(profile))), 0.3501)
            self.assertLess(np.max(np.abs(np.diff(profile, n=2))), 0.15)
            self.assertAlmostEqual(float(profile.mean()), 0, places=4)
            self.assertGreater(np.std(np.diff(profile)), 0.01)
        determinant = 1 - np.gradient(dx)[:, None] * np.gradient(dy)[None, :]
        self.assertGreater(determinant.min(), 0.85)

    def test_geometry_seed_repeats_field_and_scan_modes(self):
        from planetary_vlm.distortion.geometry import _jitter_maps
        h, w = 64, 96
        y, x = np.mgrid[:h, :w]
        for axis in ("both", "rows", "columns"):
            with self.subTest(axis=axis):
                first = _jitter_maps(h, w, 3, 16, 42, axis)
                repeated = _jitter_maps(h, w, 3, 16, 42, axis)
                changed = _jitter_maps(h, w, 3, 16, 43, axis)
                for a, b, c in zip(first, repeated, changed):
                    np.testing.assert_array_equal(a, b)
                    self.assertFalse(np.array_equal(a, c))
                dx, dy = x - first[0], y - first[1]
                if axis == "rows":
                    np.testing.assert_allclose(dx, np.broadcast_to(dx[:, :1], dx.shape), atol=1e-5)
                    np.testing.assert_allclose(dy, np.broadcast_to(dy[:, :1], dy.shape), atol=1e-5)
                    self.assertTrue((np.diff(first[1][:, 0]) > 0).all())
                elif axis == "columns":
                    np.testing.assert_allclose(dx, np.broadcast_to(dx[:1], dx.shape), atol=1e-5)
                    np.testing.assert_allclose(dy, np.broadcast_to(dy[:1], dy.shape), atol=1e-5)
                    self.assertTrue((np.diff(first[0][0]) > 0).all())

    @staticmethod
    def half_pixel_maps(h, w, *_):
        x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return x + 0.5, y

    def test_nearest_segmentation_preserves_large_integer_ids(self):
        image = np.zeros((5, 9), dtype=np.uint8)
        seg = np.full((5, 9), 2**60 + 3, dtype=np.int64)
        _, _, target = apply_synchronous_shear(image, 3, seg_mask=seg, seg_ignore_label=-1)
        self.assertEqual(target.dtype, seg.dtype)
        self.assertTrue(set(np.unique(target)) <= {-1, 2**60 + 3})

    def test_dem_interpolates_elevations_and_marks_new_borders(self):
        image = np.zeros((5, 9), dtype=np.uint8)
        y, x = np.mgrid[:5, :9]
        dem = (x + 10 * y).astype(np.float32)
        with patch("planetary_vlm.distortion.geometry._jitter_maps", side_effect=self.half_pixel_maps):
            _, target, seg = apply_synchronous_shear(image, 0.5, dem_mask=dem)
        self.assertIsNone(seg)
        self.assertAlmostEqual(float(target[1, 4]), 14.5)
        self.assertTrue(np.isnan(target[0, -1]))
        self.assertTrue(np.isnan(target[:, -1]).all())
        self.assertEqual(target.dtype, dem.dtype)

    def test_dem_does_not_blend_nodata_into_valid_elevations(self):
        image = np.zeros((5, 9), dtype=np.uint8)
        for nodata in (np.nan, -9999.0):
            with self.subTest(nodata=nodata):
                dem = np.full((5, 9), 50, dtype=np.float64)
                dem[1, 4] = nodata
                with patch("planetary_vlm.distortion.geometry._jitter_maps", side_effect=self.half_pixel_maps):
                    _, target, _ = apply_synchronous_shear(image, 0.5, dem_mask=dem, dem_nodata=nodata)
                if np.isnan(nodata):
                    self.assertTrue(np.isnan(target[1, 3]))
                    self.assertTrue(np.isnan(target[1, 4]))
                else:
                    self.assertEqual(target[1, 3], nodata)
                    self.assertEqual(target[1, 4], nodata)
                self.assertEqual(target[1, 6], 50)

    def test_zero_shear_returns_independent_arrays(self):
        image = np.full((4, 4, 3), 70, dtype=np.uint8)
        seg = np.zeros((4, 4), dtype=np.uint8)
        output, dem, target = apply_synchronous_shear(image, 0, seg_mask=seg)
        self.assertIsNone(dem)
        self.assertFalse(np.shares_memory(output, image))
        self.assertFalse(np.shares_memory(target, seg))
        np.testing.assert_array_equal(output, image)
        np.testing.assert_array_equal(target, seg)

    def test_geometry_rejects_invalid_map_contract(self):
        image = np.zeros((4, 4), dtype=np.uint8)
        seg = image.copy()
        dem = image.astype(np.float32)
        cases = [dict(), dict(dem_mask=dem, seg_mask=seg), dict(seg_mask=seg[:2]),
                 dict(seg_mask=seg.astype(float)), dict(dem_mask=seg),
                 dict(seg_mask=seg, seg_ignore_label=256), dict(dem_mask=dem, dem_nodata="NaN")]
        for kwargs in cases:
            with self.subTest(kwargs=tuple(kwargs)):
                with self.assertRaises(ValueError):
                    apply_synchronous_shear(image, 0, **kwargs)

    def test_geometry_rejects_invalid_jitter_parameters(self):
        image = np.zeros((8, 8), dtype=np.uint8)
        cases = [dict(max_shift_px=-1), dict(max_shift_px=np.inf),
                 dict(control_spacing_px=1), dict(control_spacing_px=3.5),
                 dict(seed=-1), dict(scan_axis="unknown")]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                apply_synchronous_shear(image, seg_mask=image.copy(), **kwargs)

    def test_geometry_handles_small_shapes_and_supported_dtypes(self):
        for shape in ((1, 1), (1, 8), (8, 1), (2, 2), (8, 12, 3)):
            for dtype, value in ((np.uint8, 70), (np.uint16, 40000),
                                 (np.float32, 0.4), (np.float64, 0.4)):
                with self.subTest(shape=shape, dtype=dtype):
                    image = np.full(shape, value, dtype=dtype)
                    seg = np.zeros(shape[:2], dtype=np.uint8)
                    result, _, target = apply_synchronous_shear(image, seg_mask=seg)
                    self.assertEqual(result.shape, image.shape)
                    self.assertEqual(result.dtype, image.dtype)
                    self.assertEqual(target.shape, seg.shape)
                    self.assertTrue(np.isfinite(result).all())

    def test_random_effects_are_reproducible_without_mutation(self):
        image = np.full((48, 64, 3), 80, dtype=np.uint8)
        original = image.copy()
        for function in (add_radiation_artifacts, apply_lens_flare, apply_hard_shadows):
            with self.subTest(function=function.__name__):
                first = function(image, seed=42)
                np.testing.assert_array_equal(first, function(image, seed=42))
                self.assertFalse(np.array_equal(first, function(image, seed=43)))
                self.assertFalse(np.shares_memory(first, image))
                np.testing.assert_array_equal(image, original)

    def test_particle_points_are_bright_and_streaks_cover_multiple_pixels(self):
        image = np.zeros((64, 64), dtype=np.uint8)
        point = add_radiation_artifacts(image, num_hits=1, max_streak_length=1, seed=12)
        self.assertEqual(np.count_nonzero(point), 1)
        self.assertEqual(point.max(), 255)
        streaks = add_radiation_artifacts(image, num_hits=10, max_streak_length=12, seed=12)
        self.assertGreater(np.count_nonzero(streaks), 10)

    def test_flare_brightens_and_converges_towards_white(self):
        image = np.full((21, 21), 100, dtype=np.uint8)
        output = apply_lens_flare(image, strength=1, center=(0.5, 0.5), radius=0.1, num_ghosts=0)
        self.assertEqual(output[10, 10], 255)
        self.assertGreater(output[10, 10], output[0, 0])
        self.assertTrue((output >= image).all())

    def test_explicit_shadow_changes_only_polygon_with_hard_edge(self):
        image = np.full((10, 10, 3), 100, dtype=np.uint8)
        vertices = [(0, 0), (0.4, 0), (0.4, 1), (0, 1)]
        output = apply_hard_shadows(image, strength=0.5, vertices=vertices)
        np.testing.assert_array_equal(output[:, :5], np.full((10, 5, 3), 50))
        np.testing.assert_array_equal(output[:, 5:], image[:, 5:])

    def test_disabled_effects_are_exact_copies(self):
        image = np.arange(100, dtype=np.uint16).reshape(10, 10)
        calls = [(add_radiation_artifacts, dict(num_hits=0)),
                 (add_radiation_artifacts, dict(strength=0)),
                 (apply_lens_flare, dict(strength=0)),
                 (apply_hard_shadows, dict(strength=0)),
                 (apply_hard_shadows, dict(num_shadows=0))]
        for function, kwargs in calls:
            output = function(image, **kwargs)
            np.testing.assert_array_equal(output, image)
            self.assertFalse(np.shares_memory(output, image))

    def test_effects_preserve_image_shape_dtype_and_range(self):
        for dtype, intensity in ((np.uint8, 80), (np.uint16, 40000),
                                 (np.float32, 0.4), (np.float64, 0.4)):
            for shape in ((8, 12), (8, 12, 3), (1, 1)):
                for function in (add_radiation_artifacts, apply_lens_flare, apply_hard_shadows):
                    with self.subTest(dtype=dtype, shape=shape, function=function.__name__):
                        image = np.full(shape, intensity, dtype=dtype)
                        output = function(image)
                        self.assertEqual(output.dtype, image.dtype)
                        self.assertEqual(output.shape, image.shape)
                        self.assertTrue(np.isfinite(output).all())
                        self.assertGreaterEqual(output.min(), 0)
                        maximum = 1 if image.dtype.kind == "f" else np.iinfo(dtype).max
                        self.assertLessEqual(output.max(), maximum)

    def test_invalid_parameters_fail_explicitly(self):
        image = np.zeros((8, 8), dtype=np.uint8)
        cases = [(add_radiation_artifacts, dict(num_hits=-1)),
                 (add_radiation_artifacts, dict(num_hits=1.5)),
                 (add_radiation_artifacts, dict(max_streak_length=0)),
                 (apply_lens_flare, dict(radius=0)),
                 (apply_lens_flare, dict(center=(2, 0))),
                 (apply_lens_flare, dict(strength=np.nan)),
                 (apply_hard_shadows, dict(strength=1.1)),
                 (apply_hard_shadows, dict(vertices=[(0, 0), (1, 1)])),
                 (apply_hard_shadows, dict(vertices=[(0, 0), (0.5, 0.5), (1, 1)])),
                 (apply_hard_shadows, dict(seed=-1))]
        for function, kwargs in cases:
            with self.subTest(function=function.__name__, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    function(image, **kwargs)
        with self.assertRaises(ValueError):
            apply_lens_flare(np.full((2, 2), 255.0, dtype=np.float32))

    def test_composition_changes_image_and_only_shears_segmentation(self):
        image = np.full((48, 64, 3), 100, dtype=np.uint8)
        seg = np.zeros((48, 64), dtype=np.uint8)
        seg[12:24, 20:40] = 3
        first, dem, target = spoil_image(image, seg_mask=seg, seed=12)
        repeated, _, repeated_target = spoil_image(image, seg_mask=seg, seed=12)
        self.assertIsNone(dem)
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(target, repeated_target)
        geometric_image, _, geometric_target = spoil_image(
            image, seg_mask=seg, seed=12, num_hits=0, flare_strength=0, shadow_strength=0)
        np.testing.assert_array_equal(target, geometric_target)
        self.assertFalse(np.array_equal(first, geometric_image))
        self.assertEqual(image.min(), 100)
        self.assertEqual(seg[12, 20], 3)

    def test_composition_dem_and_disabled_stages(self):
        image = np.full((16, 16), 80, dtype=np.uint8)
        dem = np.full((16, 16), 120, dtype=np.float32)
        dem[5, 5] = np.nan
        first, target, seg = spoil_image(image, dem_mask=dem, seed=9)
        self.assertIsNone(seg)
        _, expected, _ = spoil_image(image, dem_mask=dem, seed=9,
                                    num_hits=0, flare_strength=0, shadow_strength=0)
        np.testing.assert_array_equal(target, expected)
        output, unchanged, _ = spoil_image(image, dem_mask=dem, max_shift_px=0,
                                           num_hits=0, flare_strength=0, shadow_strength=0)
        np.testing.assert_array_equal(output, image)
        np.testing.assert_array_equal(unchanged, dem)
        self.assertFalse(np.shares_memory(unchanged, dem))


if __name__ == "__main__":
    unittest.main()
