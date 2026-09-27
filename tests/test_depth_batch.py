"""Offline fixtures, not scientific stereo validation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io

from planetary_vlm.datasets._mars.batch import checkpoint, digest, pair_key


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_is_replaceable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'progress.json'
            checkpoint(path, {'count':1})
            checkpoint(path, {'count':2})
            self.assertEqual(json.loads(path.read_text()), {'count':2})
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_pair_identity_not_just_sol(self):
        self.assertNotEqual(pair_key({'left':'703MLa','right':'703MRa'}), pair_key({'left':'703MLb','right':'703MRa'}))


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('numpy','PIL','cv2')), 'optional image dependencies')
class DepthBatchTests(unittest.TestCase):
    def test_prepared_subset_crops_depth_and_keeps_GT_out_of_requests(self):
        import numpy as np
        from PIL import Image
        from planetary_vlm.datasets.preparation import build
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            Image.new('RGB',(8,8),(80,90,100)).save(root/'image.png')
            Image.new('L',(8,8),3).save(root/'mask.png')
            sample = {'sample_id':'fixture','image_path':'image.png','mask_path':'mask.png','group_id':'fixture-group',
                      'split':'test','license':'fixture-only','source_url':'https://example.invalid/fixture'}
            checkpoint(root/'samples.jsonl',[sample],jsonl=True)
            (root/'spec.toml').write_text('ignore_labels=[]\nroi=[0.25,0.25,0.75,0.75]\nmin_valid_fraction=1.0\ndominant_min_fraction=0.6\nrock_min_pixels=1\nrock_classes=["ROCK"]\ntasks=["rock_presence"]\nquestion_policy="one_per_image"\n[label_to_class]\n3="ROCK"\n')
            np.save(root/'depth.npy',np.ones((8,8),dtype='float32'))
            np.save(root/'valid.npy',np.ones((8,8),dtype=bool))
            entry = {'sample_id':'fixture','status':'technical_depth_qa_pass','image_path':str(root/'image.png'),
                     'registration':{'benchmark_sha256':digest(root/'image.png')},'depth_path':str(root/'depth.npy'),
                     'validity_path':str(root/'valid.npy'),'depth_sha256':digest(root/'depth.npy'),'validity_sha256':digest(root/'valid.npy')}
            checkpoint(root/'depth_index.jsonl',[entry],jsonl=True)
            build(root/'samples.jsonl',root/'spec.toml',root/'built',depth_index=root/'depth_index.jsonl')
            manifest = json.loads((root/'built/manifest.json').read_text())
            self.assertEqual(manifest['mars']['depth_maps'],1)
            record = json.loads((root/'built/mars/test/depth_index.jsonl').read_text())
            self.assertEqual(np.load(record['depth_path']).shape,(4,4))
            request = json.loads((root/'built/mars/test/requests.jsonl').read_text())
            self.assertNotIn('mask_path',request)
            self.assertNotIn('answer',request)
            np.save(root/'valid.npy',np.zeros((8,8),dtype=bool))
            entry['validity_sha256'] = digest(root/'valid.npy')
            checkpoint(root/'depth_index.jsonl',[entry],jsonl=True)
            with self.assertRaisesRegex(ValueError,'No depth-complete'):
                build(root/'samples.jsonl',root/'spec.toml',root/'rejected',depth_index=root/'depth_index.jsonl')
    def test_mapping_preserves_missing_values_and_camera_grid(self):
        import numpy as np
        from planetary_vlm.datasets._mars.batch import aligned_depth
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root/'depth.npy', np.array([[1.,np.nan],[3.,4.]],dtype='float32'))
            np.save(root/'valid.npy', np.array([[True,False],[True,True]]))
            depth, valid = aligned_depth(root/'depth.npy',root/'valid.npy', {'resize_size':[4,4],'crop_box':[0,0,4,4]})
            self.assertEqual(depth.shape, (4,4))
            self.assertTrue(np.isnan(depth[:2,2:]).all())
            self.assertFalse(valid[:2,2:].any())
            self.assertEqual(float(depth[3,3]),4.)

    def test_registration_accepts_identical_textured_image_rejects_wrong_scene(self):
        import numpy as np
        from PIL import Image
        from planetary_vlm.datasets._mars.batch import registration
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(42)
            array = rng.integers(0,256,(512,512,3),dtype=np.uint8)
            Image.fromarray(array).save(root/'a.png')
            Image.fromarray(255-array).save(root/'b.png')
            self.assertTrue(registration(root/'a.png',root/'a.png',8.)['pass'])
            self.assertFalse(registration(root/'a.png',root/'b.png',8.)['pass'])

    def test_missing_sources_resume_and_threshold_lock(self):
        from planetary_vlm.datasets._mars.batch import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'samples.jsonl').write_text('{"sample_id":"a"}\n')
            (root/'source.xml').write_text('fixture')
            plan = {'samples':str(root/'samples.jsonl'), 'source_samples_sha256':digest(root/'samples.jsonl'),
                    'planned_photographs':1,'planned_pairs':1,
                    'images':[{'product_id':'left','label':str(root/'source.xml'), 'label_sha256':digest(root/'source.xml'),'image_name':'missing.IMG'}],
                    'jobs':[{'sample_id':'a','eye':'left','pair':{'left':'left','right':'right'}}]}
            checkpoint(root/'plan.json',plan)
            run(root/'plan.json', root/'cache', root/'output')
            run(root/'plan.json', root/'cache', root/'output',resume=True)
            summary = json.loads((root/'output/summary.json').read_text())
            self.assertEqual(summary['qualified_photographs'],0)
            self.assertEqual(summary['attempted_jobs'],1)
            self.assertEqual(summary['unprocessed_jobs'],1)
            with self.assertRaises(ValueError):
                run(root/'plan.json',root/'cache',root/'output',resume=True,min_coverage=.7)
