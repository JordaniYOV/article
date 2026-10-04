import type { Comparison, Dataset, DatasetMetadata, Model, Protocol, Report, Result, Run } from './types';

const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');
export const assetUrl = (path: string) => `${base}${path}`;

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try { response = await fetch(`${base}${path}`, init); }
  catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new ApiError('Не удалось связаться с бэкендом. Проверьте, что он запущен.', 0);
  }
  if (response.status === 204) return undefined as T;
  const text = await response.text();
  let data: { detail?: unknown } | null = null;
  try { data = JSON.parse(text); } catch { /* A stopped proxy can return plain text. */ }
  if (!response.ok) {
    const detail = data?.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => `${(item.loc || []).filter((p: string) => p !== 'body').join('.')}: ${item.msg}`).join('; ')
      : typeof detail === 'string' ? detail : response.status >= 500
        ? 'Бэкенд недоступен. Запустите сервер и повторите запрос.' : `Ошибка запроса (${response.status})`;
    throw new ApiError(message, response.status);
  }
  if (data === null) throw new ApiError('Бэкенд вернул неожиданный ответ. Проверьте адрес сервера.', response.status);
  return data as T;
}

const json = (value?: unknown): RequestInit => ({ method: 'POST',
  ...(value === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) }) });

async function all<T>(path: string, signal?: AbortSignal): Promise<T[]> {
  const rows: T[] = [];
  for (let offset = 0; ; offset += 500) {
    const page = await request<T[]>(`${path}?limit=500&offset=${offset}`, { signal });
    rows.push(...page);
    if (page.length < 500) return rows;
  }
}

export const api = {
  health: (signal?: AbortSignal) => request<{ status: string }>('/health', { signal }),
  datasets: (signal?: AbortSignal) => all<Dataset>('/datasets', signal),
  models: (signal?: AbortSignal) => all<Model>('/model-configs', signal),
  runs: (signal?: AbortSignal) => all<Run>('/runs', signal),
  defaults: () => request('/configurations/import-defaults', json()),
  upload: (file: File, metadata: DatasetMetadata) => {
    const body = new FormData(); body.append('file', file); body.append('metadata', JSON.stringify(metadata));
    return request<Dataset>('/datasets/upload', { method: 'POST', body });
  },
  fromUrl: (url: string, metadata: DatasetMetadata) => request<Dataset>('/datasets/from-url', json({ url, ...metadata })),
  saveDataset: (value: unknown) => request<Dataset>('/dataset-configs', json(value)),
  saveModel: (value: unknown, id?: number) => request<Model>(id ? `/model-configs/${id}` : '/model-configs',
    { ...json(value), method: id ? 'PUT' : 'POST' }),
  deleteDataset: (id: number) => request<void>(`/dataset-configs/${id}`, { method: 'DELETE' }),
  deleteModel: (id: number) => request<void>(`/model-configs/${id}`, { method: 'DELETE' }),
  start: (model: Model, datasetId: number, protocol: Protocol, reuseRunId?: string) =>
    request<Run>(model.backend === 'ibm_imp' ? '/vit/runs' : '/vlm/runs', json({
      model_name: model.name, model_version: model.version, dataset_id: datasetId,
      ...protocol, ...(reuseRunId ? { reuse_run_id: reuseRunId } : {}),
    })),
  control: (id: string, action: 'cancel' | 'resume') => request<Run>(`/runs/${id}/${action}`, json()),
  metrics: (id: string) => request(`/runs/${id}/metrics`, json()),
  report: (id: string, signal?: AbortSignal) => request<Report>(`/runs/${id}/report`, { signal }),
  results: (id: string, condition: string, offset = 0, signal?: AbortSignal) => {
    const query = new URLSearchParams({ limit: '8', offset: String(offset) });
    if (condition) query.set('condition', condition);
    return request<Result[]>(`/runs/${id}/results?${query}`, { signal });
  },
  compare: (ids: string[], signal?: AbortSignal) => request<Comparison>('/comparisons', { ...json({ run_ids: ids }), signal }),
};
