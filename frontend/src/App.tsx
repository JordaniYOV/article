import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, BarChart3, Box, Database, FileText, FlaskConical, Play, RefreshCw, WifiOff } from 'lucide-react';
import { api } from './api';
import { initialProtocol } from './domain';
import type { BuiltinPlanet, BuiltinSelection, Comparison, Dataset, Model, Protocol, Report, Result, Run } from './types';
import DatasetPanel from './components/DatasetPanel';
import ModelPanel from './components/ModelPanel';
import RunPanel from './components/RunPanel';
import RunHistory from './components/RunHistory';
import ResultsPanel from './components/ResultsPanel';
import { Notice } from './components/ui';

const message = (error: unknown) => error instanceof Error ? error.message : 'Не удалось выполнить действие';
const aborted = (error: unknown) => error instanceof DOMException && error.name === 'AbortError';
const NAV = [ { id: 'datasets', label: 'Датасеты', icon: Database }, { id: 'models', label: 'Модели', icon: Box },
  { id: 'launch', label: 'Новый запуск', icon: Play }, { id: 'results', label: 'Метрики', icon: BarChart3 }, { id: 'history', label: 'История', icon: FileText } ];

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [builtins, setBuiltins] = useState<BuiltinPlanet[]>([]);
  const [builtinSelection, setBuiltinSelection] = useState<BuiltinSelection | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [online, setOnline] = useState<boolean | null>(null);
  const [catalogError, setCatalogError] = useState('');
  const [modelId, setModelId] = useState(0), [datasetId, setDatasetId] = useState(0);
  const [protocol, setProtocol] = useState<Protocol>(initialProtocol);
  const [reuseId, setReuseId] = useState('');
  const [activeId, setActiveId] = useState(''), [secondId, setSecondId] = useState('');
  const [report, setReport] = useState<Report | null>(null), [secondReport, setSecondReport] = useState<Report | null>(null);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [reportError, setReportError] = useState(''), [comparisonError, setComparisonError] = useState('');
  const [rows, setRows] = useState<Result[]>([]);
  const [answersLoading, setAnswersLoading] = useState(false), [answersError, setAnswersError] = useState('');
  const [conditionFilter, setConditionFilter] = useState(''), [offset, setOffset] = useState(0);
  const [pending, setPending] = useState('');
  const [revision, setRevision] = useState(0);
  const [nav, setNav] = useState('results');
  const [notice, setNotice] = useState<{ message: string; kind: 'error' | 'success' } | null>(null);
  const activeRun = runs.find((run) => run.id === activeId);
  const refreshInFlight = useRef(false);
  const mutationInFlight = useRef(false);

  const refresh = useCallback(async (signal?: AbortSignal, invalidate = true) => {
    if (refreshInFlight.current && !invalidate) return;
    refreshInFlight.current = true;
    try {
      const [health, nextDatasets, nextModels, nextRuns, nextBuiltins] = await Promise.all([
        api.health(signal), api.datasets(signal), api.models(signal), api.runs(signal), api.builtins(signal),
      ]);
      if (signal?.aborted) return;
      setOnline(health.status === 'ok'); setCatalogError('');
      setDatasets(nextDatasets); setModels(nextModels); setRuns(nextRuns);
      setBuiltins(nextBuiltins);
      setModelId((current) => nextModels.some((value) => value.id === current) ? current : nextModels[0]?.id || 0);
      setDatasetId((current) => nextDatasets.some((value) => value.id === current) ? current : nextDatasets[0]?.id || 0);
      setActiveId((current) => current || nextRuns[0]?.id || '');
      if (invalidate) setRevision((value) => value + 1);
    } catch (error) {
      if (!aborted(error) && !signal?.aborted) { setOnline(false); setCatalogError(message(error)); }
    } finally { refreshInFlight.current = false; }
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    const timer = setInterval(() => { if (!document.hidden) void refresh(controller.signal, false); }, 3000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [refresh]);

  useEffect(() => {
    if (!builtinSelection) return;
    const subset = builtins.find((item) => item.planet === builtinSelection.planet)?.subsets.find((item) => item.id === builtinSelection.subset);
    const id = subset?.status === 'ready' ? subset.dataset_id : null;
    setDatasetId(id || 0);
    const dataset = datasets.find((item) => item.id === id);
    if (dataset && !dataset.splits.includes(protocol.split)) {
      setProtocol((value) => ({ ...value, split: dataset.splits.includes('test') ? 'test' : 'val' }));
    }
  }, [builtins, builtinSelection, datasets, protocol.split]);

  useEffect(() => {
    const controller = new AbortController();
    setReportError('');
    if (!activeId) { setReport(null); return; }
    void api.report(activeId, controller.signal).then(setReport).catch((error) => { if (!aborted(error)) { setReport(null); setReportError(message(error)); } });
    return () => controller.abort();
  }, [activeId, activeRun?.updated_at, revision]);
  useEffect(() => {
    const controller = new AbortController();
    setAnswersError('');
    if (!activeId) { setRows([]); return; }
    setAnswersLoading(true);
    void api.results(activeId, conditionFilter, offset, controller.signal).then(setRows).catch((error) => {
      if (!aborted(error)) { setRows([]); setAnswersError(message(error)); }
    }).finally(() => { if (!controller.signal.aborted) setAnswersLoading(false); });
    return () => controller.abort();
  }, [activeId, activeRun?.completed, conditionFilter, offset, revision]);
  useEffect(() => {
    const controller = new AbortController(); setComparisonError(''); setComparison(null);
    if (!secondId) { setSecondReport(null); return; }
    setSecondReport(null);
    void (async () => {
      const other = await api.report(secondId, controller.signal); setSecondReport(other);
      if (report?.metrics && other.metrics) setComparison(await api.compare([activeId, secondId], controller.signal));
    })().catch((error) => { if (!aborted(error)) setComparisonError(message(error)); });
    return () => controller.abort();
  }, [activeId, secondId, report?.metrics_status, revision]);

  function selectRun(id: string) { setActiveId(id); setSecondId(''); setReport(null); setSecondReport(null); setComparison(null); setRows([]); setOffset(0); setConditionFilter(''); }
  function selectSecond(id: string) { setSecondId(id); setSecondReport(null); setComparison(null); }
  const perform = async (key: string, action: () => Promise<unknown>, success?: string) => {
    if (mutationInFlight.current) return;
    mutationInFlight.current = true; setPending(key); setNotice(null);
    try { await action(); if (success) setNotice({ message: success, kind: 'success' }); await refresh(); }
    catch (error) { setNotice({ message: message(error), kind: 'error' }); }
    finally { setPending(''); mutationInFlight.current = false; }
  };
  const showError = (value: string) => setNotice({ message: value, kind: 'error' });
  function chooseDataset(id: number) { setBuiltinSelection(null); setDatasetId(id); }
  function chooseBuiltin(value: BuiltinSelection) {
    if (mutationInFlight.current) return;
    setBuiltinSelection(value); setDatasetId(0);
    void perform('builtin', async () => {
      const result = await api.ensureBuiltin(value);
      if (result.dataset) {
        setDatasets((current) => [...current.filter((item) => item.id !== result.dataset!.id), result.dataset!]);
        setDatasetId(result.dataset.id);
      }
    });
  }
  const scroll = (id: string) => { setNav(id); document.getElementById(id)?.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' }); };
  return <div className="app-shell"><a href="#workspace" className="skip-link">Перейти к содержимому</a>
    <aside className="sidebar"><a href="#workspace" className="brand-mark" aria-label="Planetary Eval — главная"><Box size={27} /></a>
      <nav aria-label="Навигация по рабочему пространству">{NAV.map(({ id, label, icon: Icon }) => <button type="button" key={id} className={nav === id ? 'active' : ''} onClick={() => scroll(id)} aria-label={label} title={label}><Icon size={23} strokeWidth={1.8} /></button>)}</nav>
      <span className="sidebar-bottom" title="Planetary orbital imagery"><FlaskConical size={19} /><span>PE</span></span>
    </aside>
    <main id="workspace"><header className="workspace-header"><div><div className="eyebrow">PLANETARY EVAL <span>/</span> РАБОЧЕЕ ПРОСТРАНСТВО</div><h1>Оценка моделей<span className="title-dot">.</span></h1><p>Орбитальные снимки, ответы моделей и воспроизводимые эксперименты.</p></div>
      <div className="header-actions"><span className={`connection ${online === true ? 'connected' : online === false ? 'disconnected' : ''}`}>
        {online === false ? <WifiOff size={14} /> : <span className="tiny-dot" />}{online === null ? 'Подключение…' : online ? 'Бэкенд подключён' : 'Нет соединения'}</span>
        <button type="button" className="icon-button refresh-button" onClick={() => void refresh()} aria-label="Обновить данные" title="Обновить данные"><RefreshCw size={18} /></button></div>
    </header>
    {catalogError && <div className="connection-banner" role="status"><WifiOff size={19} /><div><strong>Нет соединения с бэкендом</strong><p>{catalogError}</p></div><button type="button" className="secondary-button small-button" onClick={() => void refresh()}>Повторить</button></div>}
    <div className="setup-grid"><DatasetPanel datasets={datasets} pending={pending} perform={perform} onSelect={chooseDataset} error={showError} />
      <ModelPanel models={models} pending={pending} perform={perform} onSelect={setModelId} error={showError} />
      <RunPanel datasets={datasets} models={models} runs={runs} modelId={modelId} datasetId={datasetId} setModelId={setModelId} setDatasetId={chooseDataset}
        builtins={builtins} builtinSelection={builtinSelection} chooseBuiltin={chooseBuiltin}
        protocol={protocol} setProtocol={setProtocol} reuseId={reuseId} setReuseId={setReuseId} pending={pending} error={showError}
        start={(value) => { const model = models.find((item) => item.id === modelId); if (!model) return;
          void perform('start', async () => { const run = await api.start(model, datasetId, value, reuseId || undefined); selectRun(run.id); setProtocol(value); }, 'Запуск добавлен в очередь'); }} />
    </div>
    <RunHistory runs={runs} models={models} activeId={activeId} select={selectRun} pending={pending}
      control={(id, action) => void perform(id, () => api.control(id, action), action === 'cancel' ? 'Запрошена остановка' : 'Запуск возвращён в очередь')} />
    <ResultsPanel runs={runs} models={models} activeId={activeId} setActiveId={selectRun} secondId={secondId} setSecondId={selectSecond}
      report={report} secondReport={secondReport} comparison={comparison} comparisonError={comparisonError} reportError={reportError}
      pending={pending} calculate={(id) => void perform('metrics', () => api.metrics(id), 'Метрики рассчитаны и сохранены')}
      rows={rows} answersLoading={answersLoading} answersError={answersError} conditionFilter={conditionFilter} setConditionFilter={setConditionFilter} offset={offset} setOffset={setOffset} />
    <footer className="workspace-footer"><span><Box size={14} />Planetary Eval</span><span><Activity size={13} />Снимки · Ответы · Метрики</span></footer>
    </main>{notice && <Notice {...notice} close={() => setNotice(null)} />}
  </div>;
}
