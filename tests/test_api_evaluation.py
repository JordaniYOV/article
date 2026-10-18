"""Direct SQLite evaluation tests; no HTTP worker, network, model or GPU."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
try:
    import numpy as np
    from PIL import Image
    from fastapi import HTTPException
    from sqlalchemy.pool import StaticPool
    from sqlmodel import SQLModel,Session,create_engine,select
    from api.evaluation import calculate,report,compare,METHOD
    from benchmark_services.common.store import DatasetConfig,ModelConfig,ModelResult,Run
    from benchmark_services.common.files import digest,file_hash
    from benchmark_services.common.preparation import prepare_bundle,renderer_provenance
except ImportError:
    Session = None


@unittest.skipIf(Session is None,'Install API/image dependencies')
class APIEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        patcher = patch('benchmark_services.common.settings.WORKSPACE_ROOT',self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Keep fixture paths below Windows MAX_PATH; the production contract
        # is still created and verified by the actual preparation functions.
        bundle_patch = patch('benchmark_services.common.preparation.bundle_directory',
            side_effect=lambda run:self.root/'bundles'/run.protocol_sha256[:12])
        bundle_patch.start()
        self.addCleanup(bundle_patch.stop)
        self.engine = create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool)
        self.addCleanup(self.engine.dispose)
        SQLModel.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.addCleanup(self.session.close)
        self.sources = []
        for i,answer in enumerate(('YES','NO')):
            path = self.root/'data_orbital'/f'image{i}.png'
            path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.zeros((256,256),dtype=np.uint8)).save(path)
            self.sources.append(dict(sample_id=f's{i}',scene_group_id=f'scene{i}',source_split='val',
                image=str(path),image_sha256=file_hash(path),answer=answer,reference_status='REFERENCE_READY'))
        self.count = 0

    def make_run(self,*,kind='vlm',outputs=('YES','NO'),statuses=('ok','ok'),sources=None,distortions=None,task='crater_presence_v1'):
        self.count += 1
        sources = deepcopy(self.sources if sources is None else sources)
        manifest = self.root/'data_orbital'/f'manifest{self.count}.jsonl'
        # Reuse the same declared dataset/manifest for matched architecture tests.
        content = ''.join(json.dumps(s)+'\n' for s in sources)
        matching = self.session.exec(select(DatasetConfig)).all()
        dataset = next((d for d in matching if Path(d.manifest_path).read_text()==content and d.task_id==task),None)
        if dataset is None:
            manifest.write_text(content)
            dataset = DatasetConfig(name=f'fixture{self.count}',planet='moon',task_id=task,
                manifest_path=str(manifest),config_sha256=digest(content))
            self.session.add(dataset)
            self.session.commit()
        model = ModelConfig(name=f'model{self.count}',model_id='synthetic-fixture',
            backend='mock' if kind=='vlm' else 'ibm_imp',config_sha256=f'fixture{self.count}')
        self.session.add(model)
        self.session.commit()
        protocol = dict(include_clean=True,distortions=distortions or [],parameters={},seed=42,
            prompt='Choose one label.',allowed_answers=['YES','NO'],max_new_tokens=8,task_id=task,
            planet='moon',split='val',renderer=renderer_provenance(),selection_sha256=digest(sources),
            manifest_sha256=file_hash(dataset.manifest_path),model_config_sha256=model.config_sha256)
        checksum = digest({k:v for k,v in protocol.items() if k!='model_config_sha256'})
        run = Run(id=f'fixture_run_{self.count}',model_config_id=model.id,dataset_config_id=dataset.id,kind=kind,
            status='completed',phase='finished',total=len(sources)*(1+len(protocol['distortions'])),
            completed=len(sources)*(1+len(protocol['distortions'])),protocol=protocol,protocol_sha256=checksum,
            samples=sources,output_dir=str(self.root/'outputs'/f'run{self.count}'),loading={'mock':True})
        self.session.add(run)
        self.session.commit()
        requests,records,_ = prepare_bundle(run)
        for request in requests:
            meta = records[request.request_id]
            i = int(meta['source_sample_id'][1:])
            row = ModelResult(run_id=run.id,model_config_id=model.id,dataset_config_id=dataset.id,
                request_id=request.request_id,source_sample_id=meta['source_sample_id'],
                scene_group_id=meta['scene_group_id'],source_split=meta['source_split'],condition_id=meta['condition_id'],
                image_paths=list(request.image_paths),prompt=request.prompt,output_type='text',
                status=statuses[i],raw_response=outputs[i],elapsed_seconds=.01,
                error_message=None if statuses[i]=='ok' else 'synthetic failure',run_metadata=meta,mock=True)
            self.session.add(row)
        self.session.commit()
        return run

    def test_canonical_encoder_labels_and_vlm_use_identical_metrics(self):
        left = self.make_run(kind='vlm')
        right = self.make_run(kind='vit',outputs=('NO','NO'))
        calculate(self.session,left.id)
        calculate(self.session,right.id)
        result = compare(self.session,[left.id,right.id])
        self.assertTrue(result['quality_comparable'])
        self.assertEqual(result['contrasts'][0]['metric'],'accuracy')
        self.assertEqual(result['contrasts'][0]['right_minus_left'],-.5)
        self.assertEqual(result['contrasts'][0]['paired_interval']['n_groups'],2)
        self.assertFalse(result['scientific_result'])

    def test_known_false_positive_and_cached_metrics(self):
        run = self.make_run(outputs=('YES','YES'))
        first = calculate(self.session,run.id)
        quality = first.metrics['conditions']['clean']['quality']
        self.assertEqual(quality['accuracy'],.5)
        self.assertEqual(quality['binary']['fp'],1)
        self.assertEqual(quality['binary']['fpr'],1)
        self.assertTrue(all(row['mock'] is True for row in first.metrics['per_request']))
        self.assertEqual(first.method_version,METHOD)
        self.assertEqual(calculate(self.session,run.id).id,first.id)
        self.assertEqual(first.metrics['experiments']['E2']['status'],'not_selected')
        self.assertEqual(first.metrics['experiments']['E3']['status'],'derived_proxy')

    def test_unknown_and_candidate_references_never_become_negative(self):
        sources = deepcopy(self.sources)
        sources[0].update(reference_status='CANDIDATE_REFERENCE')
        sources[1].update(answer=None,reference_status='UNKNOWN_REFERENCE')
        run = self.make_run(sources=sources,outputs=('YES','YES'))
        result = calculate(self.session,run.id).metrics
        condition = result['conditions']['clean']
        self.assertIsNone(condition['quality'])
        self.assertEqual(condition['reference']['n_unscored_reference'],2)
        self.assertEqual(result['experiments']['E1']['status'],'unavailable')
        self.assertEqual(result['experiments']['E3']['status'],'unavailable')
        self.assertEqual(result['scored_rows'],[])
        self.assertTrue(all(r['correct'] is None for r in result['per_request']))

    def test_unreviewed_crater_answer_requires_explicit_admission(self):
        sources = deepcopy(self.sources)
        for row in sources:
            row.pop('reference_status')
        run = self.make_run(sources=sources)
        self.assertIsNone(calculate(self.session,run.id).metrics['conditions']['clean']['quality'])

    def test_shear_needs_aligned_crater_answers_and_same_reference_cohort(self):
        sources = deepcopy(self.sources)
        sources[0]['condition_answers'] = {'shear':'YES'}
        run = self.make_run(sources=sources,distortions=[['shear']])
        result = calculate(self.session,run.id).metrics
        self.assertEqual(result['conditions']['clean']['quality']['n_scored'],2)
        self.assertEqual(result['conditions']['shear']['quality']['n_scored'],1)
        self.assertEqual(result['conditions']['shear']['quality_status'],'partial')
        self.assertIsNone(result['conditions']['shear']['change_from_clean'])
        self.assertEqual(result['experiments']['E2']['status'],'partial')

    def test_failure_and_invalid_responses_stay_in_accuracy_denominator(self):
        run = self.make_run(outputs=('YES but prose','NO'),statuses=('ok','error'))
        result = calculate(self.session,run.id).metrics
        quality = result['conditions']['clean']['quality']
        self.assertEqual(quality['n_scored'],2)
        self.assertEqual(quality['accuracy'],0)
        self.assertEqual(quality['n_invalid'],1)
        self.assertEqual(quality['n_failure'],1)
        self.assertEqual(result['experiments']['E3']['status'],'unavailable')

    def test_changed_scene_group_fails_closed(self):
        run = self.make_run()
        row = self.session.exec(select(ModelResult).where(ModelResult.run_id==run.id)).first()
        row.scene_group_id = 'wrong-scene'
        self.session.add(row)
        self.session.commit()
        with self.assertRaises(HTTPException) as error:
            calculate(self.session,run.id)
        self.assertEqual(error.exception.status_code,422)

    def test_changed_target_selection_fails_closed(self):
        run = self.make_run()
        calculate(self.session,run.id)
        sources = deepcopy(run.samples)
        sources[0]['answer'] = 'NO'
        run.samples = sources
        self.session.add(run)
        self.session.commit()
        self.assertIsNone(report(self.session,run.id)['metrics'])
        with self.assertRaises(HTTPException) as error:
            calculate(self.session,run.id)
        self.assertEqual(error.exception.status_code,422)

    def test_duplicate_request_ids_and_changed_prepared_pixels_fail_closed(self):
        run = self.make_run()
        rows = self.session.exec(select(ModelResult).where(ModelResult.run_id==run.id)).all()
        with patch('api.evaluation.run_results',return_value=[rows[0],rows[0]]):
            with self.assertRaises(HTTPException) as error:
                calculate(self.session,run.id)
            self.assertEqual(error.exception.status_code,422)
        Path(rows[0].image_paths[0]).write_bytes(b'synthetic changed prepared pixels')
        with self.assertRaises(HTTPException) as error:
            calculate(self.session,run.id)
        self.assertEqual(error.exception.status_code,422)

    def test_report_does_not_serve_stale_predictions(self):
        run = self.make_run()
        calculate(self.session,run.id)
        row = self.session.exec(select(ModelResult).where(ModelResult.run_id==run.id)).first()
        row.raw_response = 'changed'
        self.session.add(row)
        self.session.commit()
        self.assertEqual(report(self.session,run.id)['metrics_status'],'not_calculated')

    def test_no_blanket_hallucination_claim_for_other_binary_tasks(self):
        run = self.make_run(task='crater_center_v1')
        result = calculate(self.session,run.id).metrics
        self.assertEqual(result['experiments']['E3']['status'],'unavailable')
        self.assertEqual(result['experiments']['E3']['unsupported_claims_status'],'not_evaluated')

    def test_compare_requires_two_distinct_runs(self):
        run = self.make_run()
        for ids in ([run.id],[run.id,run.id]):
            with self.subTest(ids=ids),self.assertRaises(HTTPException) as error:
                compare(self.session,ids)
            self.assertEqual(error.exception.status_code,422)

    def native_run(self,*,statuses=('ok','ok')):
        from benchmark_services.common.segmentation import save_maps
        sources = deepcopy(self.sources)
        for i,source in enumerate(sources):
            mask = self.root/'data_orbital'/f'mask{i}.png'
            if not mask.exists():
                Image.fromarray(np.zeros((256,256),dtype=np.uint8)).save(mask)
            source.update(mask=str(mask),mask_sha256=file_hash(mask),classes=['Background','IMP'])
        run = self.make_run(kind='vit',task='imp_segmentation',sources=sources,statuses=statuses)
        for row in self.session.exec(select(ModelResult).where(ModelResult.run_id==run.id)).all():
            row.output_type = 'imp_segmentation'
            if row.status=='ok':
                logits = np.zeros((2,256,256),dtype=np.float32)
                maps = save_maps(Path(run.output_dir)/'maps',row.request_id,logits)
                row.raw_response = json.dumps(maps)
                row.artifacts = maps
            self.session.add(row)
        self.session.commit()
        return run

    def test_native_imp_segmentation_remains_separate_and_hash_verified(self):
        run = self.native_run()
        result = calculate(self.session,run.id).metrics
        condition = result['conditions']['clean']
        self.assertEqual(condition['metric_family'],'imp_segmentation')
        self.assertEqual(condition['quality']['pixel_accuracy'],1)
        self.assertEqual(condition['quality']['mean_iou'],1)
        self.assertEqual(result['chart'][0]['metric'],'mean_iou')
        mask = Path(run.samples[0]['mask'])
        mask.write_bytes(b'synthetic corrupted mask')
        with self.assertRaises(HTTPException) as error:
            calculate(self.session,run.id)
        self.assertEqual(error.exception.status_code,422)

    def test_incomplete_map_coverage_is_not_an_equal_quality_comparison(self):
        left = self.native_run()
        right = self.native_run(statuses=('ok','error'))
        calculate(self.session,left.id)
        score = calculate(self.session,right.id).metrics
        self.assertEqual(score['conditions']['clean']['quality_status'],'partial')
        self.assertEqual(score['conditions']['clean']['quality']['prediction_coverage'],.5)
        result = compare(self.session,[left.id,right.id])
        self.assertFalse(result['quality_comparable'])
        self.assertIsNone(result['contrasts'][0]['right_minus_left'])

    def test_masks_and_text_are_not_compared_even_for_the_same_run_kind(self):
        native = self.native_run()
        text = self.make_run(kind='vit',task='imp_segmentation',sources=native.samples)
        calculate(self.session,native.id)
        calculate(self.session,text.id)
        self.assertFalse(compare(self.session,[native.id,text.id])['quality_comparable'])


if __name__=='__main__':
    unittest.main()
