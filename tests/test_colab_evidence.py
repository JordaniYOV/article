"""CPU-only format pilot fixtures, never scientific model results."""

import ast
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_services.common.files import atomic_json
from planetary_vlm import colab, colab_evidence as pilot
from planetary_vlm.contracts import ModelRequest, ModelResponse
from planetary_vlm.datasets.crater_reference import TASKS, ANSWERS, prompt_for
from planetary_vlm.evaluation.structured import parse_label_evidence
from planetary_vlm.evaluation.parser import parse_answer
from planetary_vlm.io import write_jsonl, read_jsonl


class LabelEvidenceParserTests(unittest.TestCase):
    def test_two_explicit_fields_normalize_only_label_case_whitespace(self):
        result = parse_label_evidence('ANSWER: yes\r\nEVIDENCE: A rim is visible.', ['YES', 'NO'])
        self.assertEqual(result['parsed_answer'], 'YES')
        self.assertTrue(result['format_valid'])
        self.assertEqual(result['evidence'], 'A rim is visible.')
        self.assertEqual(result['evidence_review_status'], 'NOT_REVIEWED')
        self.assertIsNone(parse_answer('ANSWER: YES\nEVIDENCE: A rim is visible.', ['YES', 'NO']))

    def test_missing_evidence_has_label_but_invalid_format(self):
        for text in ('ANSWER: NO', 'ANSWER: NO\nEVIDENCE:   '):
            with self.subTest(text=text):
                result = parse_label_evidence(text, ['YES', 'NO'])
                self.assertEqual(result['parsed_answer'], 'NO')
                self.assertFalse(result['format_valid'])
                self.assertEqual(result['evidence_review_status'], 'NOT_PROVIDED')

    def test_never_extract_favorable_label_from_prose_or_duplicate_fields(self):
        for text in ('Yes, there is a crater.', 'YES', 'ANSWER: YES or NO\nEVIDENCE: Unclear.',
                     'ANSWER: YES\nANSWER: NO\nEVIDENCE: A pit.',
                     'ANSWER: YES\nEVIDENCE: A rim.\nMore evidence.',
                     '```\nANSWER: YES\nEVIDENCE: A rim.\n```', ''):
            with self.subTest(text=text):
                self.assertFalse(parse_label_evidence(text, ['YES', 'NO'])['format_valid'])
                self.assertIsNone(parse_label_evidence(text, ['YES', 'NO'])['parsed_answer'])

    def test_abstention_is_not_negative_and_contradiction_is_not_auto_judged(self):
        result = parse_label_evidence('ANSWER: UNSURE\nEVIDENCE: Unclear.', ['YES', 'NO'])
        self.assertTrue(result['unlisted_abstention'])
        self.assertFalse(result['label_valid'] or result['format_valid'])
        self.assertIsNone(result['parsed_answer'])
        result = parse_label_evidence('ANSWER: YES\nEVIDENCE: No crater is visible.', ['YES', 'NO'])
        self.assertTrue(result['format_valid'])
        self.assertEqual(result['evidence_review_status'], 'NOT_REVIEWED')
        with self.assertRaises(ValueError): parse_label_evidence(None, ['YES', 'NO'])


class EvidencePilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'workspace'
        self.inputs = self.root / 'data_orbital/colab_inputs'
        self.inputs.mkdir(parents=True)
        self.originals, self.origins, hashes = [], [], {}
        for sid in pilot.SOURCES:
            image = self.inputs / (sid + '.png')
            image.write_bytes(('SYNTHETIC CPU FIXTURE ' + sid).encode())
            hashes[sid] = hashlib.sha256(image.read_bytes()).hexdigest()
            for task, answers in zip(TASKS, ANSWERS):
                key = sid + '__' + task + '__clean'
                self.originals.append(ModelRequest(key, (image.relative_to(self.root).as_posix(),),
                    prompt_for(task, 256, 256), answers, 16))
                self.origins.append(dict(request_id=key, sample_id=sid, source_split='val', condition='clean',
                    evaluation_eligible=True, image_sha256=hashes[sid], scene_group_id=sid))
        write_jsonl(self.inputs / 'requests.jsonl', map(asdict, self.originals))
        write_jsonl(self.inputs / 'provenance.jsonl', self.origins)
        self.hash_patch = patch.object(pilot, 'SOURCES', hashes)
        self.hash_patch.start(); self.addCleanup(self.hash_patch.stop)

    def prepare(self):
        return pilot.prepare_requests(self.root, self.originals, self.origins)

    def test_sixteen_fixed_requests_preserve_task_rules_and_source_questions(self):
        before = [asdict(r) for r in self.originals]
        requests, settings, pairs, sources = self.prepare()
        self.assertEqual((len(requests), len(pairs), len(sources)), (16, 8, 2))
        self.assertEqual(len({r.request_id for r in requests}), 16)
        self.assertEqual(len({r.image_paths for r in requests}), 2)
        by_id = {r.request_id: r for r in requests}
        originals = {r.request_id: r for r in self.originals}
        for pair in pairs:
            original = originals[pair['source_request_id']]
            closed, evidence = (by_id[pair[m + '_request_id']] for m in ('closed', 'evidence'))
            self.assertEqual(closed.prompt, original.prompt)
            suffix = 'Reply with exactly one label: ' + ', '.join(original.allowed_answers) + '. No explanation.'
            self.assertIn(original.prompt.removesuffix(suffix), evidence.prompt)
            self.assertNotIn(suffix, evidence.prompt)
            self.assertIn('ANSWER:', evidence.prompt); self.assertIn('EVIDENCE:', evidence.prompt)
            self.assertEqual(closed.image_paths, evidence.image_paths)
        self.assertEqual(sum(p['execution_order'][0] == 'closed' for p in pairs), 4)
        self.assertTrue(all(r.max_new_tokens == 96 and not r.do_sample for r in requests))
        self.assertTrue(all(not r.allowed_answers and not r.modality_paths for r in requests))
        self.assertNotIn('candidate_answer', json.dumps([asdict(r) for r in requests]))
        self.assertTrue(all(s['repetition_penalty'] == 1 and s['no_repeat_ngram_size'] == 0
            and not s['compare_first_forward'] for s in settings.values()))
        self.assertEqual([asdict(r) for r in self.originals], before)

    def test_test_split_missing_source_hash_and_changed_suffix_rejected(self):
        self.origins[0]['source_split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'val'): self.prepare()
        self.origins[0]['source_split'] = 'val'
        with self.assertRaisesRegex(ValueError, 'Missing'): pilot.prepare_requests(self.root, [], [])
        self.originals[0] = replace(self.originals[0], prompt='Unexpected source instruction')
        with self.assertRaisesRegex(ValueError, 'instruction changed'): self.prepare()
        self.originals[0] = replace(self.originals[0], prompt=prompt_for(TASKS[0], 256, 256))
        (self.root / self.originals[0].image_paths[0]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'): self.prepare()

    def test_summary_reports_format_not_accuracy_and_retains_failures(self):
        requests, _, pairs, _ = self.prepare()
        labels = {p[m + '_request_id']: p['allowed_answers'][0] for p in pairs for m in ('closed', 'evidence')}
        rows = [asdict(ModelResponse(r.request_id, ('ANSWER: ' + labels[r.request_id] + '\nEVIDENCE: CPU fixture.')
            if '__evidence_' in r.request_id else labels[r.request_id], 'ok', .1)) for r in requests]
        report = pilot.summarize(rows, {}, pairs)
        self.assertEqual(report['comparable_label_pairs'], 8)
        self.assertEqual(report['changed_label_pairs'], 0)
        self.assertFalse(report['scientific_score_available'] or report['benchmark_admitted']
                         or report['evidence_truth_automatically_scored'])
        self.assertNotIn('accuracy', report)
        self.assertEqual(report['by_format']['evidence']['valid_formats'], 8)
        for pair in report['comparisons']:
            self.assertEqual(pair['evidence']['evidence_review_status'], 'NOT_REVIEWED')
        partial = pilot.summarize([{**rows[0], 'status': 'error', 'raw_response': 'YES'}], {}, pairs)
        self.assertEqual(partial['runtime_errors'], 1)
        self.assertEqual(len(partial['missing_request_ids']), 15)
        self.assertTrue(all(p['same_label'] is None for p in partial['comparisons']))
        with self.assertRaisesRegex(ValueError, 'Duplicate'): pilot.summarize(rows + rows[:1], {}, pairs)

    def adapter(self, *, fail_load=False):
        class Adapter:
            calls = []; initializations = 0; closed = False
            def __init__(self, options): self.options = options
            def initialize(self):
                self.initializations += 1
                if fail_load: raise RuntimeError('fixture model load failed')
                return {'device_map': "{'': 0}"}
            def predict(self, request):
                self.calls.append(request)
                assert 'candidate_answer' not in json.dumps(self.options)
                assert not request.modality_paths and not request.allowed_answers
                output = Path(self.options['diagnostic_output_dir']) / 'generation_diagnostics.json'
                traces = json.loads(output.read_text()) if output.exists() else {}
                traces[request.request_id] = {'success': True, 'hit_token_budget': False}
                atomic_json(output, traces)
                answer = 'ANSWER: YES\nEVIDENCE: Synthetic CPU fixture.' if '__evidence_' in request.request_id else 'YES'
                return ModelResponse(request.request_id, answer, 'ok', .1)
            def close(self): self.closed = True
        self.Adapter = Adapter
        return lambda _, options: Adapter(options)

    def test_cpu_pipeline_one_load_persists_inputs_and_never_changes_smoke_gate(self):
        proof = self.root / 'colab_smoke_passed.json'; proof.write_bytes(b'{"success":false}')
        before = (self.inputs / 'requests.jsonl').read_bytes()
        bundle = {'bundle_sha256': 'fixture', 'protocol': {'question_type': 'candidate_Q1_Q4_clean_unscored'}}
        adapters = []
        factory = self.adapter()
        def create(backend, options):
            adapter = factory(backend, options); adapters.append(adapter); return adapter
        with patch.object(colab, 'verify_bundle', return_value=bundle), patch.object(colab, 'environment', return_value={'fixture': True}), \
             patch('planetary_vlm.models.registry.create_adapter', side_effect=create), \
             patch('benchmark_services.spacellava.diagnostics.diagnose', return_value={'success': True}):
            result = pilot.run_pilot(self.root, self.root / 'drive', 'pilot', 'a' * 32)
        self.assertTrue(result['success']); self.assertFalse(result['benchmark_admitted'])
        self.assertEqual(len(adapters), 1); self.assertEqual(adapters[0].initializations, 1)
        self.assertEqual(len(adapters[0].calls), 16); self.assertTrue(adapters[0].closed)
        durable = Path(result['persistent_dir'])
        self.assertEqual(len(list(durable.glob('*.png'))), 2)
        self.assertEqual(len(read_jsonl(durable / 'predictions.jsonl')), 16)
        self.assertTrue((durable / 'vision_diagnostics.json').is_file())
        self.assertTrue((durable / 'evidence_report.json').is_file())
        self.assertEqual((self.inputs / 'requests.jsonl').read_bytes(), before)
        self.assertEqual(proof.read_bytes(), b'{"success":false}')
        with self.assertRaisesRegex(ValueError, 'exists'):
            pilot.run_pilot(self.root, self.root / 'drive', 'pilot', 'a' * 32)

    def test_model_load_failure_persists_partial_report_and_images(self):
        bundle = {'bundle_sha256': 'fixture', 'protocol': {'question_type': 'candidate_Q1_Q4_clean_unscored'}}
        with patch.object(colab, 'verify_bundle', return_value=bundle), patch.object(colab, 'environment', return_value={}), \
             patch('planetary_vlm.models.registry.create_adapter', side_effect=self.adapter(fail_load=True)):
            with self.assertRaisesRegex(RuntimeError, 'fixture model load failed'):
                pilot.run_pilot(self.root, self.root / 'drive', 'failed', 'b' * 32)
        durable = self.root / 'drive/runs/SpaceLLaVA/failed_evidence_bbbbbbbb'
        self.assertEqual(len(list(durable.glob('*.png'))), 2)
        report = json.loads((durable / 'evidence_report.json').read_text())
        self.assertEqual(report['completed'], 0)
        self.assertEqual(len(report['missing_request_ids']), 16)
        self.assertFalse((self.root / 'colab_smoke_passed.json').exists())

    def test_cli_dispatch_and_notebook_defaults(self):
        with patch.object(pilot, 'run_pilot', return_value={'success': True}) as mocked, redirect_stdout(StringIO()):
            self.assertEqual(colab.main(['evidence-pilot', '--workspace', str(self.root), '--persist', str(self.root / 'drive'),
                '--run-id', 'pilot', '--session-id', 'c' * 32]), 0)
        mocked.assert_called_once()
        if not (ROOT / 'notebooks/spacellava_language_colab.ipynb').is_file():
            self.skipTest('Generated legacy notebook intentionally removed')
        notebook = json.loads((ROOT / 'notebooks/spacellava_language_colab.ipynb').read_text(encoding='utf-8'))
        sources = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        self.assertIn('spacellava_precision_v12.zip', sources)
        self.assertIn("'precision-pilot'", sources)
        self.assertIn('RUN_PILOT = False', sources)
        self.assertIn('DOWNLOAD_WEIGHTS = False', sources)
        self.assertEqual(sources.count('SESSION_ID = uuid.uuid4().hex'), 1)
        self.assertIn("PRECISION_RUN['session_id']", sources)
        self.assertNotIn("step('run'", sources)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertFalse(cell['outputs'])


if __name__ == '__main__': unittest.main()
