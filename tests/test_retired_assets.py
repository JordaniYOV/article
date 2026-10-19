import hashlib
from pathlib import Path
import tempfile
import unittest

from planetary_vlm.datasets.retired_assets import retained_orbital_path


class RetainedAssetsTests(unittest.TestCase):
    def test_image_can_use_hash_identical_v4_but_not_changed_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            old='data_orbital/orbital_300_v3/moon/height/001/image.png'
            current=root/old.replace('orbital_300_v3','orbital_300_v4')
            current.parent.mkdir(parents=True); current.write_bytes(b'original')
            checksum=hashlib.sha256(b'original').hexdigest()
            self.assertEqual(retained_orbital_path(root,old,checksum),current)
            current.write_bytes(b'changed')
            with self.assertRaises(FileNotFoundError):
                retained_orbital_path(root,old,checksum)
            archived=root/old.replace('data_orbital/','data_orbital/provenance/',1)
            archived.parent.mkdir(parents=True); archived.write_bytes(b'original')
            self.assertEqual(retained_orbital_path(root,old,checksum),archived)

    def test_manifest_must_use_original_snapshot_not_v4(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            relative='data_orbital/orbital_300_v3/moon/height_manifest.jsonl'
            current=root/relative.replace('orbital_300_v3','orbital_300_v4')
            current.parent.mkdir(parents=True); current.write_text('new manifest')
            with self.assertRaises(FileNotFoundError):
                retained_orbital_path(root,relative)

    def test_changed_existing_original_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); relative='data_orbital/orbital_300_v3/file.png'
            path=root/relative; path.parent.mkdir(parents=True); path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash changed'):
                retained_orbital_path(root,relative,hashlib.sha256(b'original').hexdigest())
