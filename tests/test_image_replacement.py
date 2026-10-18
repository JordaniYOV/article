"""CPU-only image replacement integrity fixtures, not scientific/GPU evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planetary_vlm.datasets import image_replacement as module
from planetary_vlm.datasets.orbital_questions import digest, file_hash
from planetary_vlm.io import read_jsonl, write_json, write_jsonl
try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:
    cv2 = np = Image = None


@unittest.skipIf(Image is None, 'Pillow/NumPy/OpenCV required')
class ImageReplacementTests(unittest.TestCase):
    def test_real_source_support_and_inverse_map(self):
        source = np.arange(100, dtype=np.uint8).reshape(10, 10)
        result = module.resample_to_reference(source, [[1, 0, 2], [0, 1, 2]], (4, 4))
        np.testing.assert_array_equal(result, source[2:6, 2:6])

    def test_invalid_transform_and_fabricated_border_are_rejected(self):
        source = np.zeros((10, 10), dtype=np.uint8)
        for matrix in ([[1, 0, 0], [0, 1, 0]], [[1, 0, 8], [0, 1, 8]],
                       [[1, 0, float('nan')], [0, 1, 2]], [[1, 0], [0, 1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                module.resample_to_reference(source, matrix, (4, 4))

    def fixture(self, root):
        source = root / 'data_orbital/orbital_300_v3'
        assets = source / 'mars/segmentation/018'
        assets.mkdir(parents=True)
        image, mask = assets / 'image.png', assets / 'mask.png'
        Image.new('L', (4, 4), 1).save(image)
        Image.new('L', (4, 4), 255).save(mask)
        (assets / 'height.npy').write_bytes(b'immutable synthetic height fixture')
        row = dict(sample_id='mars_seg_018', image=image.relative_to(root).as_posix(),
                   mask=mask.relative_to(root).as_posix(), image_sha256=file_hash(image),
                   mask_sha256=file_hash(mask), source_split='val', scene_group_id='fixture-scene')
        for planet in ('moon', 'mars'):
            for track in ('segmentation', 'height', 'train'):
                write_jsonl(source / planet / (track + '_manifest.jsonl'),
                            [row] if (planet, track) == ('mars', 'segmentation') else [])
        write_json(source / 'audit.json', dict(total_images=1))
        comparison = root / 'data_orbital/source_comparison_v1'
        comparison.mkdir()
        pixels = np.arange(100, dtype=np.uint8).reshape(10, 10)
        for name in ('themis_v11_geographic_crop.png', 'themis_v11_geographic_crop_padded.png'):
            Image.fromarray(pixels).save(comparison / name)
        return source, pixels

    def test_new_collection_changes_one_image_and_leaves_all_old_files_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, pixels = self.fixture(root)
            before = {p.relative_to(source): file_hash(p) for p in source.rglob('*') if p.is_file()}
            registration = dict(reference_to_padded_native=[[1, 0, 2], [0, 1, 2]], output_size=[4, 4])
            with patch.object(module, 'register_themis', return_value=(pixels[2:6, 2:6], registration)):
                result = module.replace_mars_image(root)
            self.assertEqual(result['changed_data_assets'], ['mars/segmentation/018/image.png'])
            self.assertEqual(result['unchanged_data_assets'], 2)
            self.assertEqual(before, {p: file_hash(source / p) for p in before})
            output = root / 'data_orbital/orbital_300_v4'
            for name in ('mask.png', 'height.npy'):
                self.assertEqual(file_hash(output / 'mars/segmentation/018' / name),
                                 file_hash(source / 'mars/segmentation/018' / name))
            row = read_jsonl(output / 'mars/segmentation_manifest.jsonl')[0]
            proof = module.verify_replacement(root, row)
            sealed = json.loads((output / 'image_replacement.json').read_text(encoding='utf-8'))
            self.assertEqual(digest(proof), sealed['record_sha256'])
            with self.assertRaises(ValueError):
                module.replace_mars_image(root)
            Image.new('L', (4, 4), 99).save(output / 'mars/segmentation/018/image.png')
            with self.assertRaisesRegex(ValueError, 'input/output changed'):
                module.verify_replacement(root, row)


if __name__ == '__main__':
    unittest.main()
