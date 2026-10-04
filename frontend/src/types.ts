export type Effect = 'shear' | 'radiation' | 'lens_flare' | 'hard_shadows';
export type Parameters = Record<string, Record<string, unknown>>;
export interface Dataset {
  id: number; name: string; version: string; planet: 'moon' | 'mars'; task_id: string;
  manifest_path: string; requests_path: string | null; targets_path: string | null;
  splits: string[]; conditions: string[]; preprocessing: Record<string, unknown>;
  provenance: Record<string, unknown>; config_sha256: string;
}
export interface Model {
  id: number; name: string; version: string; model_id: string; backend: string;
  revision: string | null; options: Record<string, unknown>; seed: number;
  max_new_tokens: number; enabled: boolean; config_sha256: string;
}
export interface Protocol {
  image_count: number; split: 'val' | 'test'; seed: number; include_clean: boolean;
  distortions: Effect[][]; parameters: Parameters; prompt: string; allowed_answers: string[];
}
export interface Run {
  id: string; run_id: string; model_config_id: number; dataset_config_id: number;
  kind: 'vlm' | 'vit'; status: string; phase: string; total: number; completed: number;
  error_message: string | null; protocol: Protocol; protocol_sha256: string;
  created_at: string; updated_at: string; output_dir: string;
  progress: { completed: number; total: number; percent: number };
}
export interface Condition {
  runtime: { n_expected: number; n_responses: number; n_ok: number; n_error: number;
    n_unsupported: number; success_rate: number; mean_seconds: number | null; median_seconds: number | null };
  quality: Record<string, unknown> | null;
  metric_family: string | null; quality_status: string; quality_reason: string | null;
  change_from_clean: number | null;
}
export interface Report {
  run_id: string; status: string; phase: string; error: string | null; mock: boolean;
  model: { name: string; version: string; backend: string };
  dataset: { id: number; name: string; planet: string; task_id: string };
  metrics_status: string; protocol_sha256: string; output_directory: string;
  metrics: { conditions: Record<string, Condition>; n_source_images: number; n_scene_groups: number;
    experiments: Record<string, { status: string; reason?: string }>; [key: string]: unknown } | null;
}
export interface Result {
  id: number; run_id: string; request_id: string; source_sample_id: string;
  condition_id: string; output_type: string; status: string; raw_response: string;
  elapsed_seconds: number; error_message: string | null; mock: boolean;
  image_url: string; map_url: string | null;
}
export interface Comparison {
  runs: Report[]; quality_comparable: boolean; mock: boolean; note: string | null;
  contrasts: { condition: string; comparable: boolean; metric: string | null; right_minus_left: number | null;
    paired_interval?: { interval: number[] | null; n_groups: number; warning: string | null } }[];
}
export interface DatasetMetadata {
  name: string; version: string; planet: string; task_id: string;
  provenance: Record<string, unknown>; preprocessing: Record<string, unknown>;
}
