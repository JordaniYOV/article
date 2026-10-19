"""Synthetic CVAT merge fixtures; never touch real source data or network."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from planetary_vlm.datasets import crater_annotations as module
from planetary_vlm.datasets.orbital_questions import file_hash
from planetary_vlm.io import read_jsonl, write_jsonl
try:
    from PIL import Image
except ImportError:
    Image=None


class FakeSources:
    provenance={"synthetic_fixture":True}
    def __init__(self,*args): pass
    def recover(self,*args):
        return [],[],{"source_instances_complete":True,"source_geometry_status":"synthetic_complete","source_mask_reproduced_exactly":None}


@unittest.skipIf(Image is None,"Pillow required for CPU fixtures")
class CraterAnnotationTests(unittest.TestCase):
    def fixture(self,root):
        rules=root/'configs/crater_annotation_v2.toml'
        rules.parent.mkdir(parents=True)
        rules.write_bytes((ROOT/'configs/crater_annotation_v2.toml').read_bytes())
        exports={}
        for planet in ('moon','mars'):
            rows=[]
            for index,split in enumerate(('val','test'),1):
                sid=f'{planet}_seg_{index:03d}'
                folder=root/f'data_orbital/orbital_300_v3/{planet}/segmentation/{index:03d}'
                folder.mkdir(parents=True)
                image,mask=folder/'image.png',folder/'mask.png'
                Image.new('RGB',(100,100)).save(image)
                Image.new('L',(100,100)).save(mask)
                stage=root/f'data_orbital/cvat_images/{planet}/{split}/{sid}.png'
                stage.parent.mkdir(parents=True)
                stage.write_bytes(image.read_bytes())
                rows.append({"sample_id":sid,"planet":planet,"source_split":split,"evaluation_eligible":True,
                    "source_dataset":"synthetic_fixture","scene_group_id":sid,"source_image_id":sid,"source_revision":"fixture",
                    "license_claim":"synthetic test fixture, no scientific claim","final_test_eligible":split=='test',
                    "image":image.relative_to(root).as_posix(),"mask":mask.relative_to(root).as_posix(),
                    "image_sha256":file_hash(image),"mask_sha256":file_hash(mask)})
                document=ET.Element('annotations'); ET.SubElement(document,'version').text='1.1'
                frame=ET.SubElement(document,'image',id='0',name=sid+'.png',width='100',height='100')
                box=ET.SubElement(frame,'box',label='crater',xtl='40',ytl='40',xbr='60',ybr='60')
                for key in ('uncertain','truncated'): ET.SubElement(box,'attribute',name=key).text='false'
                tag=ET.SubElement(frame,'tag',label='image_review')
                ET.SubElement(tag,'attribute',name='result').text='supplemented'
                ET.SubElement(tag,'attribute',name='complete_for_size_rule').text='true'
                archive=root/f'{planet}_{split}.zip'
                with zipfile.ZipFile(archive,'w') as bundle: bundle.writestr('annotations.xml',ET.tostring(document))
                exports[planet,split]=archive
            manifest=root/f'data_orbital/orbital_300_v3/{planet}/segmentation_manifest.jsonl'
            write_jsonl(manifest,rows)
            write_jsonl(manifest.with_name('train_manifest.jsonl'),[])
        return exports

    def build(self,root,exports,output='data_orbital/annotation_v1',**kwargs):
        with patch.object(module,'SourceAnnotations',FakeSources):
            return module.build_crater_questions(root,output,exports,'data_orbital/source_cache',previews=False,**kwargs)

    def test_build_keeps_splits_and_separates_unreviewed_candidates_from_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); exports=self.fixture(root)
            result=self.build(root,exports)
            self.assertEqual(result['images'],4)
            self.assertEqual(result['candidate_questions'],16)
            self.assertEqual(result['invariant_candidate_answers'],16)
            self.assertEqual(result['reference_ready_questions'],0)
            folder=root/'data_orbital/annotation_v1'
            self.assertEqual(read_jsonl(folder/'targets.jsonl'),[])
            self.assertEqual(len(read_jsonl(folder/'candidate_requests.jsonl')),16)
            self.assertEqual({s['source_split'] for s in read_jsonl(folder/'samples.jsonl')},{'val','test'})
            verified=module.verify_crater_build(root,folder)
            self.assertFalse(verified['ready_for_scored_inference'])
            inputs, metadata = module.load_crater_requests(root, 'data_orbital/annotation_v1')
            self.assertEqual(len(inputs), 16)
            self.assertEqual(metadata['question_type'], 'candidate_Q1_Q4_clean_unscored')
            self.assertEqual(metadata['reference_ready_questions'], 0)
            for request in read_jsonl(folder/'candidate_requests.jsonl'):
                self.assertEqual(set(request),{'request_id','image_paths','prompt','allowed_answers','max_new_tokens','do_sample','modality_paths'})
                self.assertEqual(request['modality_paths'],[])
            with self.assertRaises(FileExistsError): self.build(root,exports)

    def test_only_explicit_hash_bound_second_review_emits_real_reference_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); exports=self.fixture(root); self.build(root,exports)
            reviews=read_jsonl(root/'data_orbital/annotation_v1/review_queue.jsonl')
            for record in reviews:
                record.update(review_status='accepted',annotator='fixture annotator',reviewer='fixture independent reviewer',
                    reviewed_at='2026-10-15T12:00:00Z',review_basis='synthetic known geometry',source_instances_complete=True,
                    duplicates_reviewed=True,uncertainty_accounted_for=True)
            review_path=root/'data_orbital/accepted.jsonl'; write_jsonl(review_path,reviews)
            result=self.build(root,exports,'data_orbital/annotation_v2',reviews=review_path)
            self.assertEqual(result['reference_ready_questions'],16)
            targets=read_jsonl(root/'data_orbital/annotation_v2/targets.jsonl')
            self.assertEqual({t['answer'] for t in targets},{'YES','5','RIGHT'})
            self.assertTrue(all(t['derivation_version']=='crater-reference-v1' for t in targets))
            self.assertTrue(module.verify_crater_build(root,root/'data_orbital/annotation_v2')['ready_for_scored_inference'])
            from planetary_vlm.datasets.manifest import load_requests, check_assets
            check_assets(load_requests(root/'data_orbital/annotation_v2/requests.jsonl'))

    def test_changed_asset_fails_before_creating_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); exports=self.fixture(root)
            (root/'data_orbital/orbital_300_v3/moon/segmentation/001/mask.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed source asset'): self.build(root,exports)
            self.assertFalse((root/'data_orbital/annotation_v1').exists())

    def test_unknown_candidates_cannot_be_exported_as_complete_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); exports=self.fixture(root)
            class Incomplete(FakeSources):
                def recover(self, *args):
                    return [], [], {'source_instances_complete':False,
                                    'source_geometry_status':'synthetic_unknown', 'source_mask_reproduced_exactly':None}
            with patch.object(module,'SourceAnnotations',Incomplete):
                result=module.build_crater_questions(root,'data_orbital/incomplete',exports,
                                                     'data_orbital/source_cache',previews=False)
            self.assertGreater(result['unknown_references'], 0)
            with self.assertRaisesRegex(ValueError, 'unresolved references'):
                module.load_crater_requests(root,'data_orbital/incomplete')

    def test_stale_review_cannot_be_reused(self):
        obj={'instance_id':'id','bbox_xyxy':[1,1,10,10],'uncertain':False,'truncated':False}
        review={'review_status':'accepted','annotator':'fixture','reviewer':'other fixture','reviewed_at':'2026-10-15T00:00:00Z',
                'review_basis':'fixture','annotation_fingerprint':'old'}
        with self.assertRaisesRegex(ValueError,'immutable annotation version'): module.apply_review([obj],[],review,'new')

    def test_snapshot_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); exports=self.fixture(root); self.build(root,exports)
            folder=root/'data_orbital/annotation_v1'
            (folder/'exports/moon_val.zip').write_bytes(b'changed synthetic snapshot')
            with self.assertRaisesRegex(ValueError,'Snapshotted CVAT export changed'):
                module.verify_crater_build(root,folder)

    def test_unpacked_cvat_xml_keeps_the_sealed_content_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); exports=self.fixture(root); self.build(root,exports)
            folder=root/'data_orbital/annotation_v1'
            import zipfile
            for archive in (folder/'exports').glob('*.zip'):
                with zipfile.ZipFile(archive) as bundle:
                    raw=bundle.read('annotations.xml')
                xml=archive.with_suffix('')/'annotations.xml'
                xml.parent.mkdir(); xml.write_bytes(raw); archive.unlink()
            result=module.verify_crater_build(root,folder)
            self.assertEqual(set(result['export_storage'].values()),
                             {'original_xml_verified_zip_bytes_not_available'})
            xml.write_bytes(b'changed XML')
            with self.assertRaisesRegex(ValueError,'Snapshotted CVAT XML changed'):
                module.verify_crater_build(root,folder)

    def test_unresolved_catalog_instances_need_geometry_or_reasoned_exclusion(self):
        review={'review_status':'accepted','annotator':'fixture','reviewer':'other fixture','reviewed_at':'2026-10-15T00:00:00Z',
                'review_basis':'fixture','annotation_fingerprint':'same','complete_for_size_rule':True,'source_instances_complete':True,
                'duplicates_reviewed':True,'uncertainty_accounted_for':True,'unknown_regions':[]}
        with self.assertRaisesRegex(ValueError,'Every unresolved'): module.apply_review([],[{'instance_id':'id'}],review,'same')

    def test_overlap_is_a_review_flag_not_deduplication(self):
        objects=[{'instance_id':str(i),'origin':'publisher_coco','bbox_xyxy':[10+i,10+i,30+i,30+i],
                  'uncertain':False,'truncated':False} for i in range(2)]
        self.assertEqual(len(module.overlap_pairs(objects,100,100)),1)
        self.assertEqual(len(objects),2)

    def visibility_fixture(self):
        iid='moon_seg_018__cvat_1d6deba4ef902a52'
        obj={'instance_id':iid,'bbox_xyxy':[176.71,224.58,208.64,254.03],'uncertain':False,'truncated':False}
        imported={'moon_seg_018':{'supplemental_instances':[obj]}}
        exports={f'{p}_{s}.zip':{'archive_sha256':f'fixture-{p}-{s}'} for p in ('moon','mars') for s in ('val','test')}
        context={'export_sha256':{k:v['archive_sha256'] for k,v in exports.items()},'visibility_confirmations':[
            {'sample_id':'moon_seg_018','instance_id':iid,'bbox_xyxy':obj['bbox_xyxy'][:],'truncated':False,
             'decision':'original_bbox_and_truncated_flag_confirmed','reviewed_by':'synthetic human reviewer',
             'review_basis':'synthetic explicit rim confirmation'}]}
        return context,imported,exports

    def test_partial_visibility_confirmation_is_exact_and_not_scoring_acceptance(self):
        context,imported,exports=self.visibility_fixture()
        result=module.confirmed_visibility(context,imported,exports)
        self.assertEqual(set(result),{'moon_seg_018__cvat_1d6deba4ef902a52'})
        self.assertNotIn('review_status',next(iter(result.values())))
        self.assertFalse(context.get('whole_dataset_ground_truth_accepted',False))
        self.assertEqual(module.confirmed_visibility(None,imported,exports),{})

    def test_visibility_confirmation_rejects_changed_exports_geometry_or_arbitrary_regions(self):
        for change in ('hash','box','flag','unknown_region','attribution','duplicate'):
            context,imported,exports=self.visibility_fixture(); record=context['visibility_confirmations'][0]
            if change=='hash':context['export_sha256']['moon_test.zip']='changed'
            if change=='box':record['bbox_xyxy'][3]=254.1
            if change=='flag':record['truncated']=True
            if change=='unknown_region':record['instance_id']='mars_seg_018__arbitrary_source_object'
            if change=='attribution':record['review_basis']=''
            if change=='duplicate':context['visibility_confirmations'].append(dict(record))
            with self.subTest(change=change),self.assertRaises(ValueError):
                module.confirmed_visibility(context,imported,exports)

    def test_review_context_snapshot_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);exports=self.fixture(root)
            context=root/'data_orbital/context.json';context.write_text('{}',encoding='utf-8')
            self.build(root,exports,context=context)
            folder=root/'data_orbital/annotation_v1'
            (folder/'review_context.json').write_text('{"changed":true}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'review_context.json'):
                module.verify_crater_build(root,folder)

    def test_protected_and_source_outputs_rejected_before_creating_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); exports=self.fixture(root)
            for output in ('data_rover/new','data_orbital/orbital_300_v3/new','data_orbital/cvat_images/new'):
                with self.subTest(output=output),self.assertRaises(ValueError): self.build(root,exports,output)

    def test_cli_calls_offline_builder_without_implicit_download(self):
        from planetary_vlm.cli import main
        with patch.object(module,'build_crater_questions',return_value={'synthetic_fixture':True}) as call, \
             patch('planetary_vlm.datasets.crater_sources.fetch_annotations') as fetch,redirect_stdout(StringIO()):
            args=['crater-questions','--output','data_orbital/example']
            for planet in ('moon','mars'):
                for split in ('val','test'): args.extend([f'--{planet}-{split}',f'{planet}_{split}.zip'])
            self.assertEqual(main(args),0)
            call.assert_called_once(); fetch.assert_not_called()


if __name__=='__main__': unittest.main()
