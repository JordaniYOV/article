"""Small offline contracts; never download sources or import Torch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from planetary_vlm.datasets.crater_segmentation import (
    Exclusions, atomic_json, collection_path, image_signature, mars_group,
    moon_group, rasterize_craters, render_wac, source_stem, spread, validate_pair,
)


class SelectionContracts(unittest.TestCase):
    def test_source_aliases_are_canonical(self):
        self.assertEqual(source_stem(r'C:\data\THM_DIR_N-30_300_121.PNG'), 'thm_dir_n-30_300_121')
        self.assertEqual(source_stem('moon_wac_test_M123CE_r2_c3'), 'm123ce_r2_c3')

    def test_groups_are_source_product_not_sample_id(self):
        self.assertEqual(moon_group('M123CE_r12_c240'), 'M123CE')
        self.assertEqual(mars_group('thm_dir_N-30_300_121'), 'thm_dir_N-30_300')
        for value in ('unknown', '../fake', 'M123CE'):
            with self.assertRaises(ValueError):
                moon_group(value)

    def test_same_mars_image_excluded_across_different_repositories(self):
        index = Exclusions()
        index.add('mars', 'thm_dir_N00_270_224.png', 'thm_dir_N00_270', 'file', ('pixels', '0000000000000000'))
        self.assertEqual(index.reason('mars', 'THM_DIR_N00_270_224.PNG', 'other'), 'source_id_overlap')
        self.assertEqual(index.reason('mars', 'thm_dir_N00_270_225.png', 'thm_dir_N00_270'), 'scene_group_overlap')

    def test_decoded_and_near_duplicate_guards(self):
        index = Exclusions()
        index.add('moon', 'old', 'group1', 'file', ('pixels', '0000000000000000'))
        self.assertEqual(index.reason('moon', 'new', 'group2', signature=('pixels', 'ffffffffffffffff')), 'decoded_pixels_overlap')
        self.assertEqual(index.reason('moon', 'new', 'group2', signature=('new', '0000000000000003')), 'near_duplicate_dhash')
        self.assertIsNone(index.reason('moon', 'new', 'group2', signature=('new', '0000000000000007')))

    def test_shared_mars_mosaic_is_not_falsely_called_a_new_region(self):
        index = Exclusions()
        index.add('mars', 'thm_dir_N00_270_1.png', 'thm_dir_N00_270', 'old_file', ('old_pixels', '0000000000000000'))
        self.assertEqual(index.reason('mars', 'thm_dir_N00_270_2.png', 'thm_dir_N00_270'), 'scene_group_overlap')
        self.assertIsNone(index.reason('mars', 'thm_dir_N00_270_2.png', 'thm_dir_N00_270', check_groups=False))
        self.assertEqual(index.reason('mars', 'thm_dir_N00_270_1.png', 'thm_dir_N00_270', check_groups=False), 'source_id_overlap')

    def test_spread_is_deterministic_round_robin(self):
        source = [{'source_image_id': f'{group}_{i}', 'scene_group_id': group} for group in ('a', 'b', 'c') for i in range(3)]
        a, b = list(spread(source)), list(spread(list(reversed(source))))
        self.assertEqual(a, b)
        self.assertEqual(len({r['scene_group_id'] for r in a[:3]}), 3)

    def test_collection_never_overwrites_originals_or_training(self):
        root = Path(tempfile.gettempdir()) / 'crater-unit-workspace'
        for value in ('data_orbital', 'data_orbital/orbital_300_v3/new', 'data_orbital/head_training_v2', 'data_rover/new', '../outside'):
            with self.assertRaises(ValueError):
                collection_path(root, value)
        self.assertEqual(collection_path(root, 'data_orbital/new_v1'), root.resolve() / 'data_orbital/new_v1')

    def test_metadata_write_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity.json'
            atomic_json(path, {'count': 50})
            atomic_json(path, {'count': 50})
            with self.assertRaisesRegex(ValueError, 'Immutable'):
                atomic_json(path, {'count': 51})
            self.assertEqual(json.loads(path.read_text()), {'count': 50})


@unittest.skipUnless(importlib.util.find_spec('PIL') and importlib.util.find_spec('numpy'), 'Pillow/NumPy optional')
class ImageContracts(unittest.TestCase):
    def test_mask_union_uses_publisher_polygons_not_components(self):
        from PIL import Image
        coco = {'categories': [{'id': 1, 'name': 'Crater'}], 'annotations': [
            {'id': 1, 'category_id': 1, 'image_id': 1, 'segmentation': [[2, 2, 8, 2, 8, 8, 2, 8]]},
            {'id': 2, 'category_id': 1, 'image_id': 1, 'segmentation': [[7, 7, 12, 7, 12, 12, 7, 12]]},
        ]}
        mask, annotations = rasterize_craters(coco, {'id': 1, 'width': 32, 'height': 32})
        self.assertEqual(len(annotations), 2)
        self.assertEqual(set(mask.tobytes()), {0, 1})
        self.assertEqual(mask.getpixel((7, 7)), 1)
        validate_pair(Image.new('RGB', (32, 32)), mask)

    def test_non_crater_category_is_not_silently_remapped(self):
        coco = {'categories': [{'id': 1, 'name': 'IMP'}], 'annotations': [
            {'id': 1, 'category_id': 1, 'image_id': 1, 'segmentation': [[2, 2, 8, 2, 8, 8]]}]}
        with self.assertRaises(ValueError):
            rasterize_craters(coco, {'id': 1, 'width': 32, 'height': 32})

    def test_invalid_mask_labels_or_size_are_rejected(self):
        from PIL import Image
        image = Image.new('RGB', (32, 32))
        for mask in (Image.new('L', (32, 32), 2), Image.new('L', (31, 32), 1)):
            with self.assertRaises(ValueError):
                validate_pair(image, mask)

    def test_exact_pixel_signature_ignores_png_serialization(self):
        from io import BytesIO
        from PIL import Image
        source = Image.new('RGB', (32, 32), '#6384ab')
        buffer = BytesIO()
        source.save(buffer, format='PNG', compress_level=0)
        with Image.open(BytesIO(buffer.getvalue())) as decoded:
            self.assertEqual(image_signature(source), image_signature(decoded))

    def test_wac_renderer_matches_declared_original_percentile_rule(self):
        import numpy as np
        cube = np.arange(512 * 512 * 5, dtype=np.float32).reshape(512, 512, 5)
        rgb, info = render_wac(cube)
        self.assertEqual(rgb.shape, (512, 512, 3))
        for band in range(3):
            low, high = np.percentile(cube[..., band], (2, 98))
            expected = (np.clip((cube[..., band] - low) / (high - low), 0, 1) * 255).astype('uint8')
            np.testing.assert_array_equal(rgb[..., band], expected)
        self.assertEqual(info['bands_zero_based'], [0, 1, 2])
        with self.assertRaises(ValueError):
            render_wac(np.zeros((512, 512, 3)))


if __name__ == '__main__':
    unittest.main()
