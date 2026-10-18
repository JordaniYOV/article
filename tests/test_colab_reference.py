"""Synthetic CPU fixtures only. No model weights, GPU, or scientific scores."""

import ast
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planetary_vlm import colab, colab_evidence, colab_reference as coordinator, reference_spacellava as worker
from planetary_vlm.contracts import ModelRequest
from planetary_vlm.datasets.crater_reference import TASKS, ANSWERS, prompt_for
from planetary_vlm.io import read_jsonl, write_jsonl


class Tensor:
    def __init__(self, value):
        import numpy as np
        self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def float(self): return self
    def to(self, *args, **kwargs): return self
    def numpy(self): return self.value
    def tolist(self): return self.value.tolist()
    def unsqueeze(self, axis):
        import numpy as np
        return Tensor(np.expand_dims(self.value, axis))
    def sum(self): return Tensor(self.value.sum())
    def item(self): return self.value.item()
    def __eq__(self, other): return Tensor(self.value == other)
    def __getitem__(self, key): return Tensor(self.value[key])


def fake_runtime():
    class Conversation:
        roles = ('USER', 'ASSISTANT')
        def __init__(self): self.messages = []
        def copy(self): return Conversation()
        def append_message(self, role, message): self.messages.append((role, message))
        def get_prompt(self): return str(self.messages)
    class Model:
        config = SimpleNamespace(mm_use_im_start_end=False)
        generation_config = SimpleNamespace(to_dict=lambda: {'fixture': True})
        device = 'cuda:0'
        hooks = []; calls = 0
        def register_forward_pre_hook(self, fn, with_kwargs):
            self.hooks.append(fn)
            return SimpleNamespace(remove=lambda: self.hooks.remove(fn))
        def generate(self, ids, **kwargs):
            self.calls += 1
            assert kwargs['do_sample'] is False and kwargs['max_new_tokens'] == 96
            for hook in self.hooks: hook(self, (), {'past_key_values': None, 'use_cache': True})
            # LLaVA inputs_embeds returns a dummy BOS, not the input prompt IDs.
            return SimpleNamespace(sequences=Tensor([[1, 5, 2]]), scores=(Tensor([[0, 1, 2, 3, 4, 5]]), Tensor([[0] * 6])))
    tokenizer = SimpleNamespace(eos_token_id=2,
        batch_decode=lambda ids, skip_special_tokens: ['CPU FIXTURE' if skip_special_tokens else '<s>CPU FIXTURE</s>'],
        decode=lambda ids: str(ids[0]))
    return SimpleNamespace(model=Model(), tokenizer=tokenizer, processor=object(),
        torch=SimpleNamespace(float16='float16', is_tensor=lambda t: isinstance(t, Tensor), inference_mode=nullcontext),
        constants=SimpleNamespace(DEFAULT_IMAGE_TOKEN='<image>', IMAGE_TOKEN_INDEX=-200),
        conversation=SimpleNamespace(conv_templates={'llava_v1': Conversation()}),
        mm_utils=SimpleNamespace(process_images=lambda *args: Tensor([[[[1., 2.]]]]),
            tokenizer_image_token=lambda *args, **kwargs: Tensor([1, -200, 42])))


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'workspace'
        inputs = self.root / 'data_orbital/colab_inputs'
        inputs.mkdir(parents=True)
        originals, origins, hashes = [], [], {}
        for index, sid in enumerate(colab_evidence.SOURCES):
            image = inputs / (sid + '.png')
            Image.new('RGB', (256, 256), (index * 40, 0, 0)).save(image)
            hashes[sid] = worker.file_hash(image)
            for task, answers in zip(TASKS, ANSWERS):
                key = sid + '__' + task + '__clean'
                originals.append(ModelRequest(key, (image.relative_to(self.root).as_posix(),), prompt_for(task, 256, 256), answers, 16))
                origins.append(dict(request_id=key, sample_id=sid, source_split='val', condition='clean',
                    evaluation_eligible=True, image_sha256=hashes[sid], scene_group_id=sid))
        self.hash_patch = patch.object(colab_evidence, 'SOURCES', hashes)
        self.hash_patch.start(); self.addCleanup(self.hash_patch.stop)
        write_jsonl(inputs / 'requests.jsonl', map(asdict, originals))
        write_jsonl(inputs / 'provenance.jsonl', origins)
        requests, _, pairs, self.sources = colab_evidence.prepare_requests(self.root, originals, origins)
        self.rows = json.loads(json.dumps([asdict(r) for r in requests]))
        traces = {}; self.expectations = {}
        runtime = fake_runtime()
        for row in self.rows:
            _, _, _, trace = worker.prepare(runtime, row['image_paths'][0], row['prompt'])
            traces[row['request_id']] = {**trace, 'generation_arguments': worker.GENERATION,
                'generated_token_ids': [5, 2], 'first_token_top5': []}
            self.expectations[row['request_id']] = {'rendered_prompt': trace['rendered_prompt'],
                'input_token_ids': trace['input_token_ids'], 'pixels_sha256_float32': trace['pixels']['sha256_float32']}
        self.environment = {'dependencies': {k: 'FIXTURE' for k in coordinator.CORE_DEPENDENCIES},
            **{k: 'FIXTURE' for k in ('gpu', 'gpu_memory_bytes', 'compute_capability', 'cuda', 'driver')}}
        self.probes = [{'sample_id': s['sample_id'], **{k: {'sha256_float32': trace['pixels']['sha256_float32']}
            for k in ('pixels', 'vision', 'projector', 'next_token_logits')}} for s in self.sources]
        self.baseline = {'execution': {'pairs': pairs, 'sources': self.sources, 'environment': self.environment},
            'requests': self.rows, 'traces': traces, 'predictions': [{'request_id': r['request_id'],
                'raw_response': 'CPU FIXTURE', 'status': 'ok'} for r in self.rows],
            'vision': {'traces': self.probes}, 'loading': {'provenance': {'checkpoint_sha256': {
                '/old/weights/SpaceLLaVA/file' + str(i): 'a' * 64 for i in range(15)}}},
            'raw': {s['archived_image']: Path(s['image_path']).read_bytes() for s in self.sources}}

    def job(self, name='fixture'):
        folder = self.root / 'outputs' / name
        folder.mkdir(parents=True)
        job = dict(schema=worker.SCHEMA, workspace=str(self.root), output_dir=str(folder), requests=self.rows,
            input_expectations=self.expectations, sources=[{'sample_id': s['sample_id'], 'image_path': s['image_path']} for s in self.sources],
            checkpoint_sha256={}, worker_sha256=worker.file_hash(worker.__file__))
        path = folder / 'reference_job.json'
        worker.write_json(path, job)
        return path, job, folder

    def test_standalone_cli_has_no_project_imports_and_runs_help_without_torch(self):
        tree = ast.parse(Path(worker.__file__).read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(n and n.startswith(('planetary_vlm', 'benchmark_services')) for n in imports))
        result = subprocess.run([sys.executable, '-B', worker.__file__, '--help'], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--job', result.stdout)

    def test_native_generation_uses_full_returned_ids_and_fresh_cache(self):
        runtime = fake_runtime()
        row = self.rows[0]
        text, trace = worker.predict(runtime, row, self.expectations[row['request_id']])
        self.assertEqual(text, 'CPU FIXTURE')
        self.assertEqual(trace['returned_ids'], [1, 5, 2])
        self.assertEqual(trace['generated_token_ids'], [5, 2])
        self.assertTrue(trace['ended_with_eos']); self.assertFalse(trace['hit_token_budget'])
        self.assertEqual(trace['cache_trace']['initial_past_key_values'], 'none')
        self.assertEqual(runtime.model.hooks, [])

    def test_input_drift_stops_before_generate(self):
        for field in self.expectations[self.rows[0]['request_id']]:
            bad = deepcopy(self.expectations[self.rows[0]['request_id']]); bad[field] = 'changed'
            runtime = fake_runtime()
            with self.assertRaisesRegex(ValueError, 'inputs differ'): worker.predict(runtime, self.rows[0], bad)
            self.assertEqual(runtime.model.calls, 0)

    def test_preparer_same_sixteen_prompts_order_inputs_and_no_answers(self):
        rows, expected, sources = coordinator.make_inputs(self.root, self.baseline)
        self.assertEqual(rows, self.rows); self.assertEqual(expected, self.expectations)
        self.assertEqual(len(sources), 2)
        self.assertNotIn('raw_response', json.dumps(expected))
        self.baseline['requests'][0]['prompt'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'contract changed'): coordinator.make_inputs(self.root, self.baseline)

    def test_environment_missing_or_different_versions_fail_closed(self):
        self.assertEqual(coordinator.environment_differences(self.environment, deepcopy(self.environment)), {})
        changed = deepcopy(self.environment); changed['dependencies']['torch'] = None; changed['driver'] = 'changed'
        result = coordinator.environment_differences(self.environment, changed)
        self.assertEqual(set(result), {'dependencies/torch', 'driver'})

    def test_exact_baseline_archive_hash_required(self):
        archive = self.root / 'different.zip'; archive.write_bytes(b'not the audited result')
        with self.assertRaisesRegex(ValueError, 'exact audited'): coordinator.read_baseline(archive)

    def test_checkpoint_identity_rejects_missing_and_ambiguous_files(self):
        self.assertEqual(len(coordinator.checkpoint_identity(self.baseline['loading']['provenance'])), 15)
        with self.assertRaisesRegex(ValueError, 'Incomplete'): coordinator.checkpoint_identity({'checkpoint_sha256': {}})
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            coordinator.checkpoint_identity({'checkpoint_sha256': {'outside/file': 'hash'}})

    def test_worker_input_only_job_guard_before_any_load(self):
        for target in ('job', 'request', 'sidecar'):
            path, job, _ = self.job(target)
            if target == 'job': job['baseline_predictions'] = []
            if target == 'request': job['requests'][0]['ground_truth'] = 'NO'
            if target == 'sidecar': job['input_expectations'][job['requests'][0]['request_id']]['logits'] = [1.]
            worker.write_json(path, job)
            with patch.object(worker, 'load_runtime') as load, self.assertRaises(ValueError): worker.execute(path)
            load.assert_not_called()
            # The fixture lists are deliberately shared; restore mutations.
            job['requests'][0].pop('ground_truth', None)
            job['input_expectations'][job['requests'][0]['request_id']].pop('logits', None)

    def runtime_fixture(self):
        return SimpleNamespace(provenance={'fixture_not_inference': True}, model=object(),
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))

    def fixture_predict(self, runtime, row, expected):
        self.assertNotIn('raw_response', json.dumps(expected))
        return 'CPU FIXTURE', {'generated_token_ids': [5, 2], 'first_token_top5': []}

    def test_worker_one_load_sixteen_calls_no_overwrite(self):
        path, _, folder = self.job()
        with patch.object(worker, 'load_runtime', return_value=self.runtime_fixture()) as load, \
             patch.object(worker, 'neutral_probes', return_value=self.probes), \
             patch.object(worker, 'predict', side_effect=self.fixture_predict) as generate, redirect_stdout(StringIO()):
            worker.execute(path)
            with self.assertRaisesRegex(ValueError, 'exists'): worker.execute(path)
        self.assertEqual(load.call_count, 1); self.assertEqual(generate.call_count, 16)
        self.assertEqual(len(read_jsonl(folder / 'predictions.jsonl')), 16)
        self.assertFalse(json.loads((folder / 'worker_started.json').read_text())['adapter_used'])

    def test_worker_error_retained_and_no_fallback_or_following_queries(self):
        path, _, folder = self.job()
        with patch.object(worker, 'load_runtime', return_value=self.runtime_fixture()), \
             patch.object(worker, 'neutral_probes', return_value=self.probes), \
             patch.object(worker, 'predict', side_effect=RuntimeError('synthetic failure')) as generate:
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'): worker.execute(path)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(read_jsonl(folder / 'predictions.jsonl')[0]['status'], 'error')
        self.assertTrue((folder / 'reference_error.json').is_file())

    def test_comparison_missing_errors_duplicates_and_no_accuracy(self):
        _, _, folder = self.job()
        write_jsonl(folder / 'predictions.jsonl', [{'request_id': self.rows[0]['request_id'], 'status': 'error', 'raw_response': ''}])
        report = coordinator.compare_results(folder, self.baseline)
        self.assertEqual((report['completed'], report['runtime_errors'], len(report['missing_request_ids'])), (1, 1, 15))
        self.assertIsNone(report['comparisons'][0]['same_generated_tokens'])
        self.assertFalse(report['benchmark_admitted'] or report['scientific_score_available'])
        self.assertNotIn('accuracy', report)
        with (folder / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(self.baseline['predictions'][0]) + '\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'): coordinator.compare_results(folder, self.baseline)

    def test_coordinator_direct_script_no_answers_until_exit_and_drive_checkpoint(self):
        destination = self.root / 'src/planetary_vlm/reference_spacellava.py'
        destination.parent.mkdir(parents=True)
        shutil.copyfile(worker.__file__, destination)
        owner = self
        class Process:
            pid = os.getpid()
            stdout = StringIO('')
            def __init__(self, command, **kwargs):
                owner.assertEqual(command[:3], [sys.executable, '-u', str(destination)])
                path = Path(command[-1]); folder = path.parent
                owner.assertFalse((folder / 'baseline_predictions.jsonl').exists())
                owner.assertNotIn('raw_response', path.read_text())
                with patch.object(worker, 'load_runtime', return_value=owner.runtime_fixture()), \
                     patch.object(worker, 'neutral_probes', return_value=owner.probes), \
                     patch.object(worker, 'predict', side_effect=owner.fixture_predict), redirect_stdout(StringIO()):
                    worker.execute(path)
                owner.assertFalse((folder / 'baseline_predictions.jsonl').exists())
            def wait(self, **kwargs): return 0
            def poll(self): return 0
        with patch.object(colab, 'verify_bundle', return_value={'bundle_sha256': 'fixture'}), \
             patch.object(colab, 'environment', return_value=self.environment), \
             patch.object(coordinator, 'read_baseline', return_value=self.baseline), \
             patch.object(coordinator.subprocess, 'Popen', Process):
            result = coordinator.run_pilot(self.root, self.root / 'drive', 'cpu_fixture', 'a' * 32, 'not_read.zip')
        self.assertTrue(result['success']); self.assertFalse(result['benchmark_admitted'])
        durable = Path(result['persistent_dir'])
        self.assertEqual(len(read_jsonl(durable / 'baseline_predictions.jsonl')), 16)
        self.assertEqual(len(list(durable.glob('*.png'))), 2)
        report = json.loads((durable / 'reference_comparison.json').read_text())
        self.assertEqual(report['same_raw_responses'], 16)
        self.assertTrue((durable / 'worker_attestation.json').is_file())
        self.assertFalse((self.root / 'colab_smoke_passed.json').exists())
        with self.assertRaisesRegex(ValueError, 'exists'):
            coordinator.run_pilot(self.root, self.root / 'drive', 'cpu_fixture', 'a' * 32, 'not_read.zip')

    def test_cli_dispatch_baseline_required(self):
        with patch.object(coordinator, 'run_pilot', return_value={'success': True}) as launch, redirect_stdout(StringIO()):
            code = colab.main(['reference-pilot', '--workspace', str(self.root), '--persist', str(self.root / 'drive'),
                '--run-id', 'pilot', '--session-id', 'c' * 32, '--baseline', 'audited.zip'])
        self.assertEqual(code, 0); self.assertEqual(launch.call_args.args[-1], 'audited.zip')

    def native_builder_fixture(self, *, precision='nf4', paired=False):
        repo = self.root / 'external/LLaVA'
        model_dir = self.root / 'weights/SpaceLLaVA'; model_dir.mkdir(parents=True)
        (self.root / 'weights/clip-vit-large-patch14-336').mkdir()
        (model_dir / 'config.json').write_text(json.dumps(dict(model_type='llava', vocab_size=32000,
            mm_vision_tower='openai/clip-vit-large-patch14-336')))
        for name in ('llava/model/builder.py', 'llava/eval/run_llava.py', 'llava/model/language_model/llava_llama.py', 'llava/__init__.py'):
            path = repo / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# CPU fixture')
        modules = {name: ModuleType(name) for name in ('torch', 'bitsandbytes', 'transformers', 'llava', 'llava.model.builder',
            'llava.model.language_model.llava_llama', 'llava.utils', 'llava.constants', 'llava.conversation', 'llava.mm_utils')}
        modules['llava'].__file__ = str(repo / 'llava/__init__.py')
        modules['torch'].float16 = 'float16'; modules['torch'].manual_seed = lambda seed: None
        modules['torch'].cuda = SimpleNamespace(is_available=lambda: True)
        class Linear4bit:
            weight = SimpleNamespace(quant_state=SimpleNamespace(quant_type='nf4'))
        modules['bitsandbytes'].nn = SimpleNamespace(Linear4bit=Linear4bit)
        modules['transformers'].BitsAndBytesConfig = lambda **kwargs: kwargs
        modules['transformers'].CLIPImageProcessor = SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace())
        config = SimpleNamespace()
        modules['llava.model.language_model.llava_llama'].LlavaConfig = SimpleNamespace(from_pretrained=lambda *args, **kwargs: config)
        modules['llava.utils'].disable_torch_init = lambda: None
        class Tokenizer:
            is_fast = False
            def __len__(self): return 32000
        class Model:
            is_loaded_in_4bit = precision == 'nf4'; device = 'cuda:0'; hf_device_map = {'': 0}
            def eval(self): return self
            def requires_grad_(self, value): return self
            def named_modules(self): return [('model.layers.' + str(i), Linear4bit()) for i in range(280 if precision == 'nf4' else 0)]
            def parameters(self): return [SimpleNamespace(is_floating_point=lambda: True, dtype='float16')]
        calls = []
        def builder(*args, **kwargs):
            calls.append((args, kwargs)); return Tokenizer(), Model(), SimpleNamespace(), 2048
        modules['llava.model.builder'].load_pretrained_model = builder
        original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        with patch.dict(sys.modules, modules), patch.dict(os.environ), \
             patch.object(worker.subprocess, 'check_output', side_effect=[worker.RUNTIME_COMMIT, '']), \
             patch.object(worker, 'precision_admission', return_value={'gpu_identity': {'uuid': 'fixture'}}), \
             patch.object(worker, 'verify_tensors', return_value={'fixture': True}) as check, redirect_stdout(StringIO()):
            runtime = worker.load_runtime({'workspace': str(self.root), 'checkpoint_sha256': {},
                'schema': worker.PRECISION_SCHEMA if paired else worker.SCHEMA, 'precision': precision,
                'gpu_identity': {'uuid': 'fixture'}}, self.root)
        if precision == 'fp16': self.assertEqual(check.call_args.kwargs, {'all_weights': True})
        return calls, runtime, config, model_dir

    def test_native_builder_explicit_quantization_and_visual_overrides(self):
        calls, runtime, config, model_dir = self.native_builder_fixture()
        args, kwargs = calls[0]
        self.assertEqual(args, (str(model_dir), None, 'llava-v1.5-13b'))
        self.assertFalse(kwargs['load_4bit'])
        self.assertTrue(kwargs['quantization_config']['load_in_4bit'])
        self.assertEqual(kwargs['quantization_config']['llm_int8_skip_modules'], ['vision_tower', 'mm_projector', 'lm_head'])
        self.assertTrue(config.unfreeze_mm_vision_tower)
        self.assertEqual(config.mm_vision_tower, str(self.root / 'weights/clip-vit-large-patch14-336'))
        self.assertEqual(runtime.provenance['quantized_linear_modules'], 280)


if __name__ == '__main__': unittest.main()
