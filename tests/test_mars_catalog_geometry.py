"""Small synthetic CPU fixtures; no network, real data, or model imports."""
from pathlib import Path
import csv
import io
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from planetary_vlm.datasets.mars_catalog_geometry import ellipse_parameters, replay_catalog
try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:
    cv2 = None


def record(iid='fixture', x=250, y=230, major='10', minor='6', angle='60', kind=1):
    return {'CRATER_ID': iid, 'LONGITUDE_IMAGE_PIXEL_Y': str(x), 'LATITUDE_IMAGE_PIXEL_X': str(y),
            'DIAM_ELLIPSE_MAJOR_IMAGE': major, 'DIAM_ELLIPSE_MINOR_IMAGE': minor,
            'DIAM_ELLIPSE_ANGLE_IMAGE': angle, 'TYPE': str(kind),
            'STATUS': {1: 'Other', 2: 'Layered', 3: 'Buried', 4: 'Secondary'}[kind]}


class EllipseParameterTests(unittest.TestCase):
    def test_frozen_field_order_scale_integer_rounding_and_rotation(self):
        self.assertEqual(ellipse_parameters(record()), ((250, 230), (21, 12), 30, 1))

    def test_malformed_catalog_is_not_silently_repaired(self):
        for changes in ({'DIAM_ELLIPSE_MAJOR_IMAGE': 'nan'}, {'DIAM_ELLIPSE_MINOR_IMAGE': '11'},
                        {'LONGITUDE_IMAGE_PIXEL_Y': '-1'}, {'STATUS': 'Other crater'},
                        {'DIAM_ELLIPSE_MINOR_IMAGE': '0.01'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ellipse_parameters({**record(), **changes})


@unittest.skipIf(cv2 is None, 'OpenCV/numpy/Pillow required for optional CPU raster fixtures')
class CatalogRasterTests(unittest.TestCase):
    def replay(self, records, expected, multi=False, size=(512, 512)):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mask.png'; Image.fromarray(expected).save(path)
            return replay_catalog(records, size, path, multiclass=multi)

    def test_exact_binary_replay_retains_individual_native_bbox(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.ellipse(mask, (250, 230), (21, 12), 30, 0, 360, 1, -1, cv2.LINE_8)
        objects, audit = self.replay([record()], mask)
        self.assertTrue(audit['source_mask_reproduced_exactly'])
        ys, xs = np.nonzero(mask)
        self.assertEqual(objects[0]['bbox_xyxy'], [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1])
        self.assertFalse(objects[0]['geometry_verified_against_image'])
        self.assertFalse(audit['recipe']['physical_map_scale_verified'])

    def test_one_different_class_pixel_fails_closed(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.ellipse(mask, (250, 230), (21, 12), 30, 0, 360, 1, -1, cv2.LINE_8)
        mask[0, 0] = 1
        objects, audit = self.replay([record()], mask)
        self.assertEqual(objects, [])
        self.assertEqual(audit['differing_class_pixels'], 1)

    def test_multiclass_overdraw_keeps_distinct_catalog_IDs_even_if_hidden(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.ellipse(mask, (250, 230), (21, 12), 30, 0, 360, 3, -1, cv2.LINE_8)
        objects, audit = self.replay([record('a', kind=2), record('b', kind=3)], mask, multi=True)
        self.assertEqual([o['source_annotation_id'] for o in objects], ['a', 'b'])
        self.assertTrue(objects[0]['fully_overpainted_in_source_mask'])
        self.assertEqual(audit['fully_overpainted_catalog_instances'], 1)

    def test_clipped_catalog_center_keeps_full_extent_and_truncated(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.ellipse(mask, (250, 512), (21, 12), 30, 0, 360, 1, -1, cv2.LINE_8)
        objects, audit = self.replay([record(y=512)], mask)
        self.assertTrue(audit['source_mask_reproduced_exactly'])
        self.assertTrue(objects[0]['truncated'])
        self.assertGreater(objects[0]['bbox_xyxy'][3], 512)

    def test_ids_and_native_dimensions_are_strict(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.replay([record(), record()], mask)
        with self.assertRaisesRegex(ValueError, '512x512'):
            self.replay([record()], mask, size=(256, 256))

    def source_fixture(self, folder):
        from planetary_vlm.datasets.crater_sources import SourceAnnotations
        source = SourceAnnotations.__new__(SourceAnnotations)
        source.cache = folder
        row = {'sample_id': 'fixture', 'source_dataset': 'Mirali33/mb-crater_binary_seg',
               'source_revision': '127ebfb30f2ec68781534a3071093a234ec90257',
               'source_split': 'val', 'source_image_id': 'fixture.png'}
        value = record(); csv_text = io.StringIO()
        writer = csv.DictWriter(csv_text, fieldnames=list(value)); writer.writeheader(); writer.writerow(value)
        with zipfile.ZipFile(folder / 'mars_crater_metadata.zip', 'w') as bundle:
            bundle.writestr('metadata/val/fixture.csv', csv_text.getvalue())
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.ellipse(mask, (250, 230), (21, 12), 30, 0, 360, 1, -1, cv2.LINE_8)
        path = folder / 'mask.png'; Image.fromarray(mask).save(path)
        return source, row, path

    def test_source_reader_admits_geometry_only_after_exact_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            source, row, path = self.source_fixture(Path(directory))
            objects, pending, audit = source.recover(row, (512, 512), path)
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0]['instance_id'], 'fixture__source_catalog_fixture')
            self.assertEqual(pending, [])
            self.assertTrue(audit['source_instances_complete'])
            with Image.open(path) as image:
                image.putpixel((0, 0), 1); image.save(path)
            objects, pending, audit = source.recover(row, (512, 512), path)
            self.assertEqual(objects, [])
            self.assertEqual(len(pending), 1)
            self.assertIsNone(pending[0]['bbox_xyxy'])
            self.assertFalse(audit['source_instances_complete'])

    def test_source_reader_missing_renderer_keeps_catalog_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            source, row, path = self.source_fixture(Path(directory))
            with patch('planetary_vlm.datasets.mars_catalog_geometry.replay_catalog',
                       side_effect=ModuleNotFoundError('CPU dependency absent', name='cv2')):
                objects, pending, audit = source.recover(row, (512, 512), path)
            self.assertEqual(objects, [])
            self.assertEqual(len(pending), 1)
            self.assertEqual(audit['missing_dependency'], 'cv2')

    def test_source_reader_rejects_unpinned_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            source, row, path = self.source_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, 'revision'):
                source.recover({**row, 'source_revision': 'changed'}, (512, 512), path)


if __name__ == '__main__':
    unittest.main()
