"""Language pilot CPU fixtures; not evidence of real GPU/model performance."""

import ast
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from planetary_vlm.colab_language import prepare_requests, summarize, SAMPLE_IDS
from planetary_vlm.contracts import ModelRequest, ModelResponse
from benchmark_services.spacellava.language_diagnostics import token_trace


class LanguagePilotTests(unittest.TestCase):
    def test_diagnostic_repeat_controls_are_explicit_validated_and_do_not_enable_sampling(self):
        from benchmark_services.spacellava.language_diagnostics import generation_arguments
        request = ModelRequest('fixture', ('image.png',), 'Question', (), 96)
        base = generation_arguments(request, {})
        self.assertFalse(base['do_sample'])
        self.assertNotIn('repetition_penalty', base)
        args = generation_arguments(request, {'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0})
        self.assertEqual(args['repetition_penalty'], 1.0)
        self.assertEqual(args['no_repeat_ngram_size'], 0)
        self.assertEqual(args['max_new_tokens'], 96)
        for settings in ({'repetition_penalty': True}, {'repetition_penalty': float('nan')},
                         {'repetition_penalty': 0}, {'no_repeat_ngram_size': True}, {'no_repeat_ngram_size': -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                generation_arguments(request, settings)

    def test_initial_past_argument_distinguishes_missing_none_and_provided(self):
        from benchmark_services.spacellava.language_diagnostics import initial_past_state
        self.assertEqual(initial_past_state({}), 'not_passed')
        self.assertEqual(initial_past_state({'past_key_values':None}), 'none')
        # An empty cache object/sequence is only "provided", not assumed stale.
        self.assertEqual(initial_past_state({'past_key_values':()}), 'provided')
        self.assertEqual(initial_past_state({'past_key_values':object()}), 'provided')

    def test_generation_trace_records_real_kwargs_and_native_output_without_slicing_prompt(self):
        import numpy as np
        from benchmark_services.spacellava.language_diagnostics import generate_with_trace
        class Tensor:
            def __init__(self,value):self.value=np.asarray(value)
            def __getitem__(self,key):return Tensor(self.value[key])
            def detach(self):return self
            def cpu(self):return self
            def float(self):return self
            def tolist(self):return self.value.tolist()
            def numpy(self):return self.value
        calls=[]
        class Model:
            generation_config=SimpleNamespace(eos_token_id=2,to_dict=lambda:{'eos_token_id':2})
            hook=None
            fail=False
            def register_forward_pre_hook(self,hook,*,with_kwargs):
                self.hook=hook
                assert with_kwargs
                return SimpleNamespace(remove=lambda:setattr(self,'hook',None))
            def __call__(self,**kwargs):
                if self.hook is not None:self.hook(self,(),kwargs)
                return SimpleNamespace(logits=Tensor([[[.1,.2,3.]]]))
            def generate(self,*args,**kwargs):
                calls.append(kwargs)
                # Mimic the pinned inputs_embeds API overriding the request.
                self(use_cache=True)
                if self.fail:raise RuntimeError('fixture generation failure')
                self(use_cache=True)
                return SimpleNamespace(sequences=Tensor([[1,42,2]]),scores=(Tensor([[.1,.2,3.]]),Tensor([[.1,.2,3.]])))
        with tempfile.TemporaryDirectory() as temporary:
            request=ModelRequest('fixture',(str(Path(temporary)/'image.png'),),'Describe.',(),128)
            options={'runtime_repo':temporary,'diagnostic_output_dir':temporary,
                     'diagnostic_request_settings':{'fixture':{'conversation_template':'llava_v0','use_cache':False,'compare_first_forward':True,
                        'repetition_penalty':1.0,'no_repeat_ngram_size':0}}}
            prepared=[]
            def prepare(r,**kwargs):
                prepared.append(kwargs)
                return Tensor([[1,-200,5]]),Tensor([1]),(256,256),{'image_token_count':1,'rendered_prompt':'Human: <image> Describe.'}
            adapter=SimpleNamespace(options=options,model=Model(),torch=SimpleNamespace(inference_mode=nullcontext),prepare_inputs=prepare,
                tokenizer=SimpleNamespace(batch_decode=lambda ids,**kwargs:['fixture answer'],decode=lambda ids:str(ids)))
            self.assertEqual(generate_with_trace(adapter,request),'fixture answer')
            report=json.loads((Path(temporary)/'generation_diagnostics.json').read_text())['fixture']
            self.assertTrue(report['success'])
            self.assertTrue(report['same_first_argmax'])
            self.assertEqual(report['generated_token_ids'],[42,2])
            self.assertEqual(prepared,[{'conversation_template':'llava_v0','return_metadata':True}])
            self.assertFalse(calls[0]['use_cache'])
            self.assertTrue(calls[0]['return_dict_in_generate'])
            self.assertTrue(calls[0]['output_scores'])
            self.assertEqual(calls[0]['max_new_tokens'],128)
            self.assertEqual(calls[0]['repetition_penalty'],1.0)
            self.assertEqual(calls[0]['no_repeat_ngram_size'],0)
            self.assertEqual(report['generation_arguments'],
                {'do_sample':False,'num_beams':1,'max_new_tokens':128,'use_cache':False,
                 'return_dict_in_generate':True,'output_scores':True,'repetition_penalty':1.0,'no_repeat_ngram_size':0})
            self.assertEqual(report['cache_trace'],{'requested':False,'observed_forward_calls':2,
                'effective_modes':[True],'unobserved_cache_calls':0,'request_overridden':True,
                'initial_past_key_values':'not_passed'})
            self.assertIsNone(adapter.model.hook)
            with self.assertRaisesRegex(ValueError,'Duplicate'):generate_with_trace(adapter,request)
            adapter.model.fail=True
            failed=ModelRequest('fixture_failure',request.image_paths,request.prompt,(),128)
            with self.assertRaisesRegex(RuntimeError,'fixture generation failure'):
                generate_with_trace(adapter,failed)
            self.assertIsNone(adapter.model.hook)
            failed_report=json.loads((Path(temporary)/'generation_diagnostics.json').read_text())['fixture_failure']
            self.assertFalse(failed_report['success'])
            self.assertEqual(failed_report['cache_trace']['observed_forward_calls'],1)

    def fixture(self,root):
        requests,origins=[],[]
        for sid in SAMPLE_IDS:
            request_id=sid+"__crater_presence_v1__clean"
            requests.append(ModelRequest(request_id,("data_orbital/"+sid+".png",),"ORIGINAL FROZEN QUESTION",("YES","NO"),16))
            origins.append(dict(request_id=request_id,sample_id=sid,source_split="val",condition="clean",
                                evaluation_eligible=True,scene_group_id=sid))
        return requests,origins

    def test_fixed_val_only_nineteen_freeform_requests_preserve_benchmark(self):
        originals,origins=self.fixture(ROOT)
        requests,settings,sources=prepare_requests(ROOT,originals,origins)
        self.assertEqual(len(requests),19)
        self.assertEqual(len({r.request_id for r in requests}),19)
        self.assertEqual(len(sources),5)
        self.assertTrue(all(not r.allowed_answers and not r.modality_paths for r in requests))
        self.assertTrue(all(r.max_new_tokens>=96 for r in requests))
        self.assertEqual(sum(not s['use_cache'] for s in settings.values()),2)
        self.assertEqual(sum(s['conversation_template']=='llava_v0' for s in settings.values()),2)
        self.assertEqual(sum(s['compare_first_forward'] for s in settings.values()),2)
        self.assertTrue(all(r.prompt=='ORIGINAL FROZEN QUESTION' and r.max_new_tokens==16 for r in originals))
        self.assertNotIn('expected_presence',json.dumps([asdict(r) for r in requests]))

    def test_test_split_and_missing_source_refused(self):
        originals,origins=self.fixture(ROOT)
        origins[0]['source_split']='test'
        with self.assertRaisesRegex(ValueError,'must not use test'):
            prepare_requests(ROOT,originals,origins)
        with self.assertRaisesRegex(ValueError,'Missing fixed'):
            prepare_requests(ROOT,[],[])

    def test_native_dummy_bos_not_confused_with_generated_tokens(self):
        report=token_trace([1,15043,2],2,128,2)
        self.assertEqual(report['returned_prefix_ids'],[1])
        self.assertEqual(report['generated_token_ids'],[15043,2])
        self.assertTrue(report['ended_with_eos'])
        self.assertFalse(report['hit_token_budget'])
        self.assertTrue(token_trace([1,3,4],2,2,2)['hit_token_budget'])
        self.assertTrue(token_trace([1,3,2],2,2,[2,5])['ended_with_eos'])
        with self.assertRaises(ValueError): token_trace([1],2,2,2)

    def test_summary_does_not_claim_accuracy_or_admit_benchmark(self):
        requests,_,_=prepare_requests(ROOT,*self.fixture(ROOT))
        rows=[dict(asdict(ModelResponse(r.request_id,'fixture text','ok',.1)),mock=False) for r in requests]
        summary=summarize(rows,{})
        self.assertFalse(summary['benchmark_admitted'])
        self.assertFalse(summary['scientific_score_available'])
        self.assertEqual(len(summary['answers']),19)
        self.assertEqual(len(summary['comparisons']),2)
        self.assertFalse(summary['comparisons'][0]['cache_off_verified'])
        key=SAMPLE_IDS[0]+'__language_describe_nocache__v1'
        for effective,verified in ((True,False),(False,True)):
            traces={key:{'cache_trace':{'requested':False,'observed_forward_calls':2,
                    'effective_modes':[effective],'unobserved_cache_calls':0}}}
            self.assertEqual(summarize(rows,traces)['comparisons'][0]['cache_off_verified'],verified)

    def test_complete_cpu_runner_never_writes_or_changes_smoke_proof(self):
        from planetary_vlm import colab
        from planetary_vlm.colab_language import run_pilot
        from planetary_vlm.io import write_jsonl
        from benchmark_services.common.files import atomic_json
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            requests,origins=self.fixture(root)
            for r in requests:
                path=root/r.image_paths[0];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic-fixture-not-image')
            inputs=root/'data_orbital/colab_inputs';inputs.mkdir(parents=True)
            write_jsonl(inputs/'requests.jsonl',map(asdict,requests));write_jsonl(inputs/'provenance.jsonl',origins)
            proof=root/'colab_smoke_passed.json';proof.write_text('{"success":false}',encoding='utf-8')
            before=proof.read_bytes()
            class Adapter:
                def __init__(self,options):self.options=options
                def initialize(self):return {'device_map':"{'':0}"}
                def predict(self,request):
                    path=Path(self.options['diagnostic_output_dir'])/'generation_diagnostics.json'
                    traces=json.loads(path.read_text()) if path.exists() else {}
                    traces[request.request_id]={'hit_token_budget':False,'fixture':True}
                    atomic_json(path,traces)
                    return ModelResponse(request.request_id,'CPU FIXTURE','ok',.1)
                def close(self):pass
            bundle={'bundle_sha256':'fixture','protocol':{},'files_sha256':{}}
            with patch.object(colab,'verify_bundle',return_value=bundle),patch.object(colab,'environment',return_value={'fixture':True}), \
                 patch('planetary_vlm.models.registry.create_adapter',side_effect=lambda _,options:Adapter(options)), \
                 patch('benchmark_services.spacellava.diagnostics.diagnose',return_value={'success':True,'fixture':True}):
                result=run_pilot(root,root/'drive','pilot','e'*32)
            self.assertTrue(result['success'])
            self.assertFalse(result['benchmark_admitted'])
            self.assertEqual(proof.read_bytes(),before)
            self.assertTrue((Path(result['persistent_dir'])/'language_report.json').is_file())
            self.assertEqual(len(json.loads((Path(result['output_dir'])/'language_report.json').read_text())['answers']),19)

    def test_notebook_schema_and_no_automatic_full_run(self):
        notebook=json.loads((ROOT/'notebooks/spacellava_language_colab.ipynb').read_text(encoding='utf-8'))
        code='\n'.join(''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in notebook['cells'] if c['cell_type']=='code')
        self.assertIn('RUN_PILOT = False',code)
        self.assertIn('DOWNLOAD_WEIGHTS = False',code)
        self.assertIn("'precision-pilot'",code)
        self.assertNotIn("step('run'",code)
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']) if isinstance(cell['source'],list) else cell['source'])
                self.assertFalse(cell['outputs'])


if __name__=='__main__':unittest.main()
