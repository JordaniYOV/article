"""Local pilot parsing checks, without network or GPU dependencies."""
import importlib.util
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'),
                     'Pilot inspection uses optional NumPy and Pillow')
class MastcamAuditTests(unittest.TestCase):
    def test_namespaced_three_component_camera_vector(self):
        from planetary_vlm.datasets._mars.labels import vector
        model = ET.fromstring('<m xmlns="fixture"><Vector_Center><x>0.1</x><y>0.2</y><z>0.3</z></Vector_Center></m>')
        self.assertEqual(vector(model, 'Vector_Center'), [0.1, 0.2, 0.3])

    def test_missing_or_ambiguous_geometry_rejected(self):
        from planetary_vlm.datasets._mars.labels import one
        with self.assertRaises(ValueError):
            one(ET.fromstring('<m/>'), 'Camera_Model_Parameters')
        with self.assertRaises(ValueError):
            one(ET.fromstring('<m><A/><A/></m>'), 'A')

    def test_nonfinite_or_wrong_length_vector_rejected(self):
        from planetary_vlm.datasets._mars.labels import vector
        for xml in ('<m><A><x>nan</x><y>1</y><z>2</z></A></m>',
                    '<m><A><x>1</x><y>2</y></A></m>'):
            with self.assertRaises(ValueError):
                vector(ET.fromstring(xml), 'A')
