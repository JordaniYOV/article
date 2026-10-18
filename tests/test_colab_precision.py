"""CPU fixtures only: not FP16 loading, VRAM or scientific results."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import test_colab_reference as shared
from planetary_vlm import colab, colab_precision as pair, reference_spacellava as worker
from planetary_vlm.io import read_jsonl


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = shared.ReferenceTests('runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.identity = {'uuid': 'GPU-CPU-FIXTURE', 'name': 'SYNTHETIC',
            'total_memory_bytes': 40 * 1024**3, 'compute_capability': [8, 0]}
        self.loads = []; self.processes = []; self.runtimes = []

    def fake_torch(self, free_gib):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
            get_device_properties=lambda _: SimpleNamespace(name='SYNTHETIC', total_memory=40 * 1024**3),
            get_device_capability=lambda _: (8, 0), mem_get_info=lambda _: (int(free_gib * 1024**3), 40 * 1024**3)))

    def test_free_memory_gate_rejects_16gb_and_accepts_40gb_fixture(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
             patch.object(worker.subprocess, 'check_output', return_value='GPU-CPU-FIXTURE\n'):
            with self.assertRaisesRegex(ValueError, 'rerun BOTH arms'): worker.precision_admission(self.fake_torch(15), self.root)
            value = worker.precision_admission(self.fake_torch(38), self.root)
        self.assertTrue(value['admitted']); self.assertEqual(value['required_free_bytes'], 32 * 1024**3)
        self.assertEqual(value['gpu_identity'], self.identity)

    def test_checkpoint_size_can_raise_memory_gate_and_bad_metadata_rejected(self):
        path = self.root / 'weights/SpaceLLaVA/model.safetensors.index.json'; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'metadata': {'total_size': 38_000_000_000}}))
        with patch.object(worker, 'gpu_identity', return_value=self.identity):
            with self.assertRaisesRegex(ValueError, 'FREE GPU memory'): worker.precision_admission(self.fake_torch(38), self.root)
            path.write_text(json.dumps({'metadata': {'total_size': 1}}))
            with self.assertRaisesRegex(ValueError, 'checkpoint size'): worker.precision_admission(self.fake_torch(38), self.root)

    def test_native_fp16_has_no_quantization_config_and_checks_all_weights(self):
        calls, runtime, _, _ = self.fixture.native_builder_fixture(precision='fp16', paired=True)
        kwargs = calls[0][1]
        self.assertNotIn('quantization_config', kwargs)
        self.assertFalse(kwargs['load_4bit'] or kwargs['load_8bit'])
        self.assertEqual(kwargs['device_map'], 'cuda:0')
        self.assertEqual(runtime.provenance['quantization'], 'none')
        self.assertEqual(runtime.provenance['quantized_linear_modules'], 0)

    def test_native_nf4_uses_same_gpu_only_placement(self):
        calls, runtime, _, _ = self.fixture.native_builder_fixture(precision='nf4', paired=True)
        self.assertEqual(calls[0][1]['device_map'], 'cuda:0')
        self.assertEqual(calls[0][1]['quantization_config']['bnb_4bit_quant_type'], 'nf4')
        self.assertEqual(runtime.provenance['quantized_linear_modules'], 280)

    def fake_load(self, job, folder):
        arm = job['precision']; self.loads.append(arm)
        self.assertNotIn('raw_response', json.dumps(job))
        if self.runtimes: self.assertIsNone(self.runtimes[-1].model)  # Closed before next load.
        provenance = {'fixture_not_inference': True, 'precision': arm, 'quantization': 'none' if arm == 'fp16' else 'nf4',
            'quantized_linear_modules': 0 if arm == 'fp16' else 280,
            'tensor_checks': {'all_language_weight_values_compared': arm == 'fp16'},
            'checkpoint_sha256': job['checkpoint_sha256'], 'code_revision': 'fixture', 'source_sha256': {},
            'processor_semantics': {}, 'memory_admission': {'gpu_identity': self.identity}}
        runtime = SimpleNamespace(provenance=provenance, model=object(),
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
        self.runtimes.append(runtime)
        return runtime

    def fake_predict(self, runtime, row, expected):
        return runtime.provenance['precision'] + ' CPU FIXTURE', {'input_token_ids': expected['input_token_ids'],
            'rendered_prompt': expected['rendered_prompt'], 'pixels': {'sha256_float32': expected['pixels_sha256_float32']},
            'generation_arguments': worker.GENERATION, 'generated_token_ids': [5, 2]}

    def patches(self, *, failure=False, changed_environment=False):
        destination = self.root / 'src/planetary_vlm/reference_spacellava.py'; destination.parent.mkdir(parents=True)
        shutil.copyfile(worker.__file__, destination)
        owner = self
        class Process:
            pid = os.getpid()
            def __init__(self, command, **kwargs):
                path = Path(command[-1]); job = json.loads(path.read_text())
                owner.processes.append(job['precision'])
                owner.assertEqual(command[:3], [sys.executable, '-u', str(destination)])
                self.stdout = StringIO('')
                worker.execute(path)
            def wait(self, **kwargs): return 0
            def poll(self): return 0
        environment = self.fixture.environment
        changed = deepcopy(environment); changed['dependencies']['torch'] = 'CHANGED'
        patches = [patch.object(colab, 'verify_bundle', return_value={'bundle_sha256': 'fixture'}),
            patch.object(colab, 'environment', side_effect=[environment, environment, changed] if changed_environment else None,
                return_value=environment),
            patch.object(pair, 'preflight', return_value={'gpu_identity': self.identity}),
            patch.object(pair, 'read_baseline', return_value=self.fixture.baseline),
            patch.object(pair.subprocess, 'Popen', Process),
            patch.object(worker, 'load_runtime', side_effect=self.fake_load),
            patch.object(worker, 'neutral_probes', return_value=self.fixture.probes),
            patch.object(worker, 'predict', side_effect=RuntimeError('CPU OOM FIXTURE') if failure else self.fake_predict)]
        for mocked in patches: mocked.start(); self.addCleanup(mocked.stop)

    def run_pair(self):
        with redirect_stdout(StringIO()):
            return pair.run_pair(self.root, self.root / 'drive', 'precision', 'a' * 32, 'not_read.zip')

    def test_two_sequential_loads_32_queries_durable_nested_results_and_frozen_inputs(self):
        self.patches()
        result = self.run_pair()
        self.assertTrue(result['success']); self.assertFalse(result['benchmark_admitted'])
        self.assertEqual(self.loads, ['fp16', 'nf4']); self.assertEqual(self.processes, self.loads)
        durable = Path(result['persistent_dir'])
        for arm in pair.ARMS:
            self.assertEqual(len(read_jsonl(durable / arm / 'predictions.jsonl')), 16)
            self.assertTrue((durable / arm / 'worker_attestation.json').is_file())
        report = json.loads((durable / 'precision_comparison.json').read_text())
        self.assertEqual(report['different_raw_responses'], 16)  # A difference is not treated as a failure/correctness claim.
        self.assertTrue(report['matched_control']); self.assertFalse(report['scientific_score_available'])
        self.assertFalse((durable / 'baseline_predictions.jsonl').exists())
        self.assertFalse((self.root / 'colab_smoke_passed.json').exists())
        with self.assertRaisesRegex(ValueError, 'exists'): self.run_pair()

    def test_fp16_error_retains_partial_results_and_does_not_launch_nf4(self):
        self.patches(failure=True)
        with self.assertRaisesRegex(RuntimeError, 'CPU OOM FIXTURE'): self.run_pair()
        self.assertEqual(self.processes, ['fp16'])
        report_path = self.root / 'drive/runs/SpaceLLaVA/precision_precision_aaaaaaaa/precision_comparison.json'
        report = json.loads(report_path.read_text())
        self.assertEqual(report['errors_per_arm']['fp16'], 1)
        self.assertEqual(len(report['missing_per_arm']['nf4']), 16)
        self.assertFalse(report['comparison_available'] or report['matched_control'])

    def test_environment_drift_stops_second_arm(self):
        self.patches(changed_environment=True)
        with self.assertRaisesRegex(ValueError, 'environment changed'): self.run_pair()
        self.assertEqual(self.processes, ['fp16'])

    def test_gpu_uuid_drift_stops_second_arm(self):
        self.patches()
        admitted = {'gpu_identity': self.identity}
        changed = {'gpu_identity': {**self.identity, 'uuid': 'GPU-OTHER-FIXTURE'}}
        pair.preflight.side_effect = [admitted, admitted, changed]
        with self.assertRaisesRegex(ValueError, 'GPU or core environment changed'): self.run_pair()
        self.assertEqual(self.processes, ['fp16'])

    def test_changed_visual_hash_withholds_matched_status(self):
        self.patches()
        result = self.run_pair(); folder = Path(result['output_dir'])
        probes = json.loads((folder / 'nf4/neutral_probes.json').read_text())
        probes[0]['vision']['sha256_float32'] = 'different'
        worker.write_json(folder / 'nf4/neutral_probes.json', probes)
        report = pair.summarize(folder, [r['request_id'] for r in self.fixture.rows])
        self.assertTrue(report['comparison_available']); self.assertFalse(report['matched_control'])

    def test_full_fp16_tensor_audit_catches_language_weight_not_in_nf4_subset(self):
        import numpy as np
        names = {'model.vision_tower.fixture' + str(i) + '.weight' for i in range(395)} | {
            'model.embed_tokens.weight', 'lm_head.weight', 'model.norm.weight',
            'model.layers.0.input_layernorm.weight', 'model.layers.39.post_attention_layernorm.weight'}
        extra = 'model.layers.0.self_attn.q_proj.weight'
        names.add(extra)
        class Tensor:
            is_meta = False
            def __init__(self, value): self.value = np.asarray(value, dtype=np.float16)
            @property
            def dtype(self): return self.value.dtype
            @property
            def shape(self): return self.value.shape
            def __getitem__(self, key): return Tensor(self.value[key])
            def detach(self): return self
            def cpu(self): return self
            def to(self, dtype): return Tensor(self.value.astype(dtype))
        state = {key: Tensor([1., 2.]) for key in names}
        class Slice:
            def get_shape(self): return [2]
            def __getitem__(self, key): return Tensor(np.asarray([1., 2.])[key])
        class Source:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def get_slice(self, name): return Slice()
        module = ModuleType('safetensors'); module.safe_open = lambda *args, **kwargs: Source()
        model = SimpleNamespace(state_dict=lambda: state, named_parameters=lambda: list(state.items()), named_buffers=lambda: [])
        torch = SimpleNamespace(float16=np.dtype('float16'), isfinite=lambda a: np.isfinite(a.value),
            equal=lambda a, b: bool(np.array_equal(a.value, b.value)))
        folder = self.root / 'weights/SpaceLLaVA'; folder.mkdir(parents=True)
        (folder / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {key: 'fixture.safetensors' for key in names}}))
        with patch.dict(sys.modules, {'safetensors': module}):
            result = worker.verify_tensors(model, folder, torch, all_weights=True)
            self.assertEqual(result['checkpoint_tensors_exact'], 401)
            self.assertTrue(result['all_language_weight_values_compared'])
            state[extra] = Tensor([5., 6.])
            self.assertFalse(worker.verify_tensors(model, folder, torch)['all_language_weight_values_compared'])
            with self.assertRaisesRegex(ValueError, 'differs from checkpoint'):
                worker.verify_tensors(model, folder, torch, all_weights=True)

    def test_cli_and_notebook_preflight_before_weights_with_safe_defaults(self):
        with patch.object(pair, 'run_pair', return_value={'success': True}) as run, redirect_stdout(StringIO()):
            self.assertEqual(colab.main(['precision-pilot', '--workspace', str(self.root), '--persist', 'drive',
                '--run-id', 'pair', '--session-id', 'b' * 32, '--baseline', 'baseline.zip']), 0)
        run.assert_called_once()
        notebook = json.loads((ROOT / 'notebooks/spacellava_language_colab.ipynb').read_text(encoding='utf-8'))
        cells = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        self.assertIn("step('precision-preflight')", cells['install'])
        self.assertIn('RUN_PILOT = False', cells['parameters']); self.assertIn('DOWNLOAD_WEIGHTS = False', cells['parameters'])
        self.assertIn("PRECISION_RUN['session_id']", cells['checks'])
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code': ast.parse(''.join(cell['source'])); self.assertFalse(cell['outputs'])


if __name__ == '__main__': unittest.main()
