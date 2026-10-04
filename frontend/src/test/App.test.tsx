import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App';
import { api } from '../api';
import { dataset, model, report, run } from './fixtures';

vi.mock('../api', () => ({ assetUrl: (path: string) => '/api' + path,
  api: Object.fromEntries(['health', 'datasets', 'models', 'runs', 'defaults', 'upload', 'fromUrl', 'saveDataset', 'saveModel',
    'deleteDataset', 'deleteModel', 'start', 'control', 'metrics', 'report', 'results', 'compare'].map((key) => [key, vi.fn()])) }));
// Chart math/values are tested separately; jsdom has no browser layout engine.
vi.mock('recharts', async () => {
  const { createElement } = await import('react');
  return Object.fromEntries(['Bar', 'BarChart', 'CartesianGrid', 'Cell', 'LabelList', 'Legend', 'Line', 'LineChart', 'ResponsiveContainer', 'Tooltip', 'XAxis', 'YAxis']
    .map((name) => [name, ({ children }: { children?: React.ReactNode }) => createElement('div', {}, children)]));
});

beforeEach(() => {
  vi.mocked(api.health).mockResolvedValue({ status: 'ok' });
  vi.mocked(api.datasets).mockResolvedValue([dataset]);
  vi.mocked(api.models).mockResolvedValue([model]);
  vi.mocked(api.runs).mockResolvedValue([]);
  vi.mocked(api.results).mockResolvedValue([]);
  vi.mocked(api.report).mockResolvedValue(report);
});

describe('evaluation workspace', () => {
  it('shows empty-state metrics without inventing scores', async () => {
    render(<App />);
    await screen.findByText('Бэкенд подключён');
    expect(screen.getByText('Создайте запуск, соберите ответы и рассчитайте метрики.')).toBeVisible();
    expect(screen.queryByText('0,80')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Рассчитать метрики' })).toBeDisabled();
  });
  it('starts one configured model and shows its queued progress', async () => {
    const queued = { ...run, status: 'queued', phase: 'waiting', completed: 0, progress: { completed: 0, total: 60, percent: 0 } };
    vi.mocked(api.start).mockImplementation(async () => { vi.mocked(api.runs).mockResolvedValue([queued]); return queued; });
    const user = userEvent.setup(); render(<App />);
    await screen.findByText('Бэкенд подключён');
    await user.click(screen.getByRole('button', { name: 'Запустить сбор ответов' }));
    await waitFor(() => expect(api.start).toHaveBeenCalled());
    const [selectedModel, selectedDataset, protocol] = vi.mocked(api.start).mock.calls[0];
    expect(selectedModel.id).toBe(7); expect(selectedDataset).toBe(3);
    expect(protocol.image_count).toBe(10); expect(protocol.distortions).toHaveLength(5);
    await screen.findByText('В очереди');
    expect(screen.getByRole('button', { name: 'Остановить запуск fixture-' })).toBeEnabled();
  });
  it('uses completed reports, marks fixture scores, and invokes metric calculation', async () => {
    vi.mocked(api.runs).mockResolvedValue([run]); vi.mocked(api.metrics).mockResolvedValue({});
    const user = userEvent.setup(); render(<App />);
    await screen.findByText('Тестовые ответы · эти значения не являются научными результатами');
    expect(screen.getAllByText('0,80').length).toBeGreaterThan(0);
    await user.click(screen.getByRole('button', { name: 'Обновить метрики' }));
    await waitFor(() => expect(api.metrics).toHaveBeenCalledWith(run.id));
    await user.click(screen.getByRole('tab', { name: 'Примеры ответов' }));
    expect(screen.getByText('Ответов пока нет')).toBeVisible();
  });
  it('disables model launches while a configuration is turned off', async () => {
    vi.mocked(api.models).mockResolvedValue([{ ...model, enabled: false }]);
    render(<App />); await screen.findByText('Включите запуск в настройках модели.');
    expect(screen.getByRole('button', { name: 'Запустить сбор ответов' })).toBeDisabled();
  });
  it('requires a selection within the uploaded dataset size', async () => {
    vi.mocked(api.datasets).mockResolvedValue([{ ...dataset, provenance: { sample_count: 3 } }]);
    const user = userEvent.setup(); render(<App />);
    await screen.findByText('В датасете всего 3 снимков. Уменьшите выборку.');
    expect(screen.getByRole('button', { name: 'Запустить сбор ответов' })).toBeDisabled();
    await user.clear(screen.getByLabelText('Снимков'));
    await user.type(screen.getByLabelText('Снимков'), '3');
    expect(screen.getByRole('button', { name: 'Запустить сбор ответов' })).toBeEnabled();
  });
  it('clears old comparison scores while switching to an unscored run', async () => {
    const secondRun = { ...run, id: 'second-fixture', run_id: 'second-fixture' };
    const secondReport = { ...report, run_id: secondRun.id };
    vi.mocked(api.runs).mockResolvedValue([run, secondRun]);
    vi.mocked(api.report).mockImplementation(async (id) => id === run.id ? report : secondReport);
    vi.mocked(api.compare).mockResolvedValue({ runs: [report, secondReport], quality_comparable: true, mock: true, note: null, contrasts: [] });
    const user = userEvent.setup(); render(<App />);
    await screen.findByText('Тестовые ответы · эти значения не являются научными результатами');
    await user.selectOptions(screen.getByLabelText('Второй запуск'), secondRun.id);
    await waitFor(() => expect(api.compare).toHaveBeenCalled());
    vi.mocked(api.report).mockImplementation(() => new Promise(() => {}));
    await user.selectOptions(screen.getByLabelText('Основной запуск'), secondRun.id);
    expect(screen.queryAllByText('0,80')).toHaveLength(0);
    expect(screen.getByLabelText('Второй запуск')).toHaveValue('');
  });
  it('restores a matched protocol and locks the source controls', async () => {
    vi.mocked(api.runs).mockResolvedValue([{ ...run, protocol: { ...run.protocol, image_count: 2, seed: 23, distortions: [['shear']] } }]);
    const user = userEvent.setup(); render(<App />); await screen.findByText('Бэкенд подключён');
    await user.selectOptions(screen.getByLabelText('Повторить выборку'), run.id);
    expect(screen.getByLabelText('Снимков')).toHaveValue(2);
    expect(screen.getByLabelText('Seed')).toHaveValue(23);
    expect(screen.getByLabelText('Снимков')).toBeDisabled();
    expect(screen.getByLabelText('Выберите модель')).toBeEnabled();
  });
  it('handles an offline backend without a fabricated connected indicator', async () => {
    vi.mocked(api.health).mockRejectedValue(new Error('Сервер остановлен'));
    render(<App />); await screen.findByText('Нет соединения с бэкендом');
    expect(screen.getByRole('button', { name: 'Повторить' })).toBeEnabled();
    expect(screen.queryByText('Бэкенд подключён')).not.toBeInTheDocument();
  });
  it('submits a direct dataset URL with user-provided metadata', async () => {
    vi.mocked(api.fromUrl).mockResolvedValue(dataset);
    const user = userEvent.setup(); render(<App />); await screen.findByText('Бэкенд подключён');
    const panel = document.getElementById('datasets')!;
    await user.click(within(panel).getByRole('button', { name: 'По ссылке' }));
    await user.type(within(panel).getByLabelText('Ссылка на датасет'), 'https://example.org/images.zip');
    await user.type(within(panel).getByLabelText('Название датасета'), 'Moon test');
    await user.click(within(panel).getByRole('button', { name: 'Подключить датасет' }));
    await waitFor(() => expect(api.fromUrl).toHaveBeenCalledWith('https://example.org/images.zip', expect.objectContaining({ name: 'Moon test', planet: 'moon' })));
  });
});
