import { initialProtocol } from '../domain';
import type { Condition, Dataset, Model, Report, Run } from '../types';

export const model: Model = { id: 7, name: 'SpaceLLaVA', version: 'v1', model_id: 'fixture/model', backend: 'mock',
  revision: null, options: {}, seed: 42, max_new_tokens: 128, enabled: true, config_sha256: 'model-hash' };
export const dataset: Dataset = { id: 3, name: 'Лунная выборка', version: 'v1', planet: 'moon', task_id: 'presence',
  manifest_path: 'data_orbital/fixture/manifest.jsonl', requests_path: null, targets_path: null,
  splits: ['val', 'test'], conditions: ['clean'], preprocessing: {}, provenance: {}, config_sha256: 'dataset-hash' };
export const run: Run = { id: 'fixture-run', run_id: 'fixture-run', model_config_id: 7, dataset_config_id: 3,
  kind: 'vlm', status: 'completed', phase: 'finished', total: 60, completed: 60, error_message: null,
  protocol: initialProtocol, protocol_sha256: 'matching-protocol', created_at: '2026-10-02T12:00:00Z',
  updated_at: '2026-10-02T12:00:00Z', output_dir: 'outputs/SpaceLLaVA/fixture-run', progress: { completed: 60, total: 60, percent: 100 } };
export const condition: Condition = { runtime: { n_expected: 10, n_responses: 10, n_ok: 10, n_error: 0,
  n_unsupported: 0, success_rate: 1, mean_seconds: .42, median_seconds: .4 }, quality: { accuracy: .8, macro_f1: .75, coverage: 1 },
  metric_family: 'closed_form_classification', quality_status: 'available', quality_reason: null, change_from_clean: 0 };
export const report: Report = { run_id: run.id, status: 'completed', phase: 'finished', error: null, mock: true,
  model: { name: model.name, version: 'v1', backend: 'mock' }, dataset: { id: 3, name: dataset.name, planet: 'moon', task_id: 'presence' },
  metrics_status: 'calculated', protocol_sha256: 'matching-protocol', output_directory: run.output_dir,
  metrics: { conditions: { clean: condition }, n_source_images: 10, n_scene_groups: 3, experiments: {} } };
