import { beforeEach, describe, expect, it, vi } from 'vitest';
import { api, request } from '../api';
import { initialProtocol } from '../domain';
import { model } from './fixtures';

describe('FastAPI client', () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  beforeEach(() => { fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 202 })); vi.stubGlobal('fetch', fetchMock); });
  it('routes exactly one VLM to the VLM endpoint with an explicit ordered protocol', async () => {
    await api.start(model, 3, initialProtocol);
    const [url, config] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/vlm/runs');
    const body = JSON.parse(config.body);
    expect(body.model_name).toBe('SpaceLLaVA');
    expect(body.dataset_id).toBe(3);
    expect(body.distortions.at(-1)).toEqual(['shear', 'hard_shadows', 'lens_flare', 'radiation']);
    expect(body).not.toHaveProperty('model_ids');
    expect(body).not.toHaveProperty('answer');
  });
  it('routes IBM to ViT and sends the source-run reference for matched selection', async () => {
    await api.start({ ...model, name: 'NASA-IBM-Lunar-Foundation-Model', backend: 'ibm_imp' }, 3, initialProtocol, 'old-run');
    expect(fetchMock.mock.calls[0][0]).toBe('/api/vit/runs');
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).reuse_run_id).toBe('old-run');
  });
  it('uploads a ZIP as multipart without a manually fabricated content-type boundary', async () => {
    const file = new File(['fixture'], 'dataset.zip');
    await api.upload(file, { name: 'Moon', version: 'v1', planet: 'moon', task_id: 'presence', provenance: {}, preprocessing: {} });
    const config = fetchMock.mock.calls[0][1];
    expect(config.body).toBeInstanceOf(FormData);
    expect(config.headers).toBeUndefined();
    expect(config.body.get('file').name).toBe('dataset.zip');
  });
  it('handles server validation errors and a stopped non-JSON proxy', async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ detail: [{ loc: ['body', 'seed'], msg: 'positive value required' }] }), { status: 422 }));
    await expect(request('/vlm/runs')).rejects.toThrow('seed: positive value required');
    fetchMock.mockResolvedValueOnce(new Response('proxy error', { status: 502 }));
    await expect(request('/health')).rejects.toThrow('Бэкенд недоступен');
  });
  it('encodes condition filters and preserves paginated offset', async () => {
    await api.results('run1', 'shear+hard_shadows', 8);
    expect(fetchMock.mock.calls[0][0]).toContain('condition=shear%2Bhard_shadows');
    expect(fetchMock.mock.calls[0][0]).toContain('offset=8');
  });
});
