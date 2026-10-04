import { useEffect, useState } from 'react';
import { BarChart3, ChevronDown, Download, FileChartColumn, Info, RefreshCw, SlidersHorizontal } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { compactCondition, conditionLabel, csvCell, fmt, METRICS, metricValue, saveFile, shortModel } from '../domain';
import type { Comparison, Model, Report, Result, Run } from '../types';
import Answers from './Answers';
import { Busy, Empty } from './ui';

const COLORS = ['#7151fb', '#3798ff'];
const TABS = ['Сравнение', 'Таблица метрик', 'Графики', 'Примеры ответов'];

function MetricChart({ metricKey, reports, condition }: { metricKey: string; reports: Report[]; condition: string }) {
  const metric = METRICS.find((item) => item.key === metricKey)!;
  const data = reports.map((report, index) => ({ name: shortModel(report.model.name), value: metricValue(report.metrics?.conditions[condition], metricKey), fill: COLORS[index] }));
  const populated = data.some((row) => row.value !== null);
  return <article className="metric-card"><header><h3>{metric.name}</h3><span className="info-icon" title={metric.hint}><Info size={14} /></span>{metric.unit && <span className="metric-unit">{metric.unit}</span>}</header>
    {populated ? <div className="bar-chart" role="img" aria-label={`${metric.name}, ${conditionLabel(condition)}`}><ResponsiveContainer width="100%" height={168}>
      <BarChart data={data} margin={{ top: 25, right: 7, left: -26, bottom: 0 }} barCategoryGap="32%">
        <CartesianGrid stroke="#e9edf4" strokeDasharray="4 4" vertical={false} />
        <XAxis dataKey="name" axisLine={{ stroke: '#dce1eb' }} tickLine={false} tick={{ fontSize: 10, fill: '#778099' }} interval={0} />
        <YAxis domain={metric.unit ? [0, 'auto'] : [0, 1]} tickLine={false} axisLine={false} tick={{ fontSize: 10, fill: '#778099' }} width={50} />
        <Tooltip cursor={{ fill: '#f5f3ff' }} formatter={(value) => [fmt(typeof value === 'number' ? value : null) + (metric.unit ? ` ${metric.unit}` : ''), metric.name]} />
        <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={64} isAnimationActive={false}>
          {data.map((row, index) => <Cell key={index} fill={row.fill} />)}<LabelList dataKey="value" position="top" formatter={(value) => fmt(typeof value === 'number' ? value : null)} style={{ fontSize: 11, fill: '#3d4561' }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer></div> : <div className="chart-unavailable"><span>—</span><small>{reports.some((r) => r.metrics) ? 'Для этой задачи метрика недоступна' : 'После расчёта метрик'}</small></div>}
    <div className="metric-values">{data.length ? data.map((row, index) => <span key={index} title={row.name}><i style={{ background: COLORS[index] }} />{row.name}<b>{fmt(row.value)}</b></span>) : <span className="field-note">Нет сохранённого отчёта</span>}</div>
  </article>;
}

function MetricsTable({ reports }: { reports: Report[] }) {
  const rows = reports.flatMap((report) => Object.entries(report.metrics?.conditions || {}).map(([key, value]) => ({ report, key, value })));
  return <div className="table-box"><div className="table-title"><h3>Сводная таблица метрик</h3><button type="button" className="text-button" disabled={!rows.length} onClick={() => {
    const header = ['Модель', 'Запуск', 'Тестовый', 'Условие', 'Accuracy', 'F1', 'Mean IoU', 'Dice IMP', 'Coverage', 'Время, с', 'Успешно', 'Всего'];
    const csv = [header, ...rows.map(({ report, key, value }) => [report.model.name, report.run_id, report.mock ? 'Да' : 'Нет', conditionLabel(key),
      ...['accuracy', 'macro_f1', 'mean_iou', 'dice', 'coverage', 'mean_seconds'].map((metric) => metricValue(value, metric)), value.runtime.n_ok, value.runtime.n_expected])];
    saveFile('\uFEFF' + csv.map((row) => row.map(csvCell).join(';')).join('\r\n'), 'planetary-metrics.csv', 'text/csv;charset=utf-8');
  }}><Download size={14} />CSV</button></div><div className="table-scroll"><table><thead><tr><th>Модель / условие</th><th>Accuracy</th><th>F1</th><th>Mean IoU</th><th>Dice · IMP</th><th title="Валидные ответы или успешно оценённые карты">Coverage</th><th>Время, с</th><th>Ответы</th></tr></thead>
    <tbody>{rows.length ? rows.map(({ report, key, value }) => <tr key={report.run_id + key}><td><span className="table-model"><i style={{ background: COLORS[reports.indexOf(report)] }} />{shortModel(report.model.name)}{report.mock && <span className="fixture-tag">Тест</span>}</span><small>{conditionLabel(key)}</small></td>
      {['accuracy', 'macro_f1', 'mean_iou', 'dice', 'coverage', 'mean_seconds'].map((metric) => <td key={metric}>{fmt(metricValue(value, metric))}</td>)}<td>{value.runtime.n_ok} / {value.runtime.n_expected}</td></tr>) : <tr><td colSpan={8} className="no-table-data">Сохранённые метрики появятся после расчёта</td></tr>}</tbody></table></div></div>;
}

function ConditionCharts({ reports, selectedMetrics }: { reports: Report[]; selectedMetrics: string[] }) {
  return <div className="condition-charts">{selectedMetrics.map((key) => {
    const metric = METRICS.find((item) => item.key === key)!;
    const conditions = [...new Set(reports.flatMap((report) => Object.keys(report.metrics?.conditions || {})))];
    const data = conditions.map((condition) => Object.assign({ condition, name: compactCondition(condition) },
      ...reports.map((report, index) => ({ [`value${index}`]: metricValue(report.metrics?.conditions[condition], key) }))));
    const present = reports.some((report) => conditions.some((condition) => metricValue(report.metrics?.conditions[condition], key) !== null));
    return <article key={key} className="metric-card wide-chart"><header><h3>{metric.name} по условиям</h3><span title={metric.hint} className="info-icon"><Info size={14} /></span></header>
      {present ? <ResponsiveContainer width="100%" height={245}><LineChart data={data} margin={{ top: 18, right: 18, left: -22, bottom: 15 }}>
        <CartesianGrid stroke="#e9edf4" strokeDasharray="4 4" vertical={false} /><XAxis dataKey="name" tick={{ fontSize: 10, fill: '#778099' }} tickLine={false} interval={0} angle={-12} height={48} />
        <YAxis domain={metric.unit ? [0, 'auto'] : [0, 1]} tick={{ fontSize: 11, fill: '#778099' }} axisLine={false} tickLine={false} />
        <Tooltip formatter={(value) => fmt(typeof value === 'number' ? value : null)} labelFormatter={(_, payload) => conditionLabel(payload?.[0]?.payload?.condition || '')} />
        <Legend wrapperStyle={{ fontSize: 11 }} />{reports.map((report, index) => <Line key={report.run_id} dataKey={`value${index}`} name={shortModel(report.model.name)} stroke={COLORS[index]} strokeWidth={2.5} dot={{ r: 4 }} connectNulls={false} isAnimationActive={false} />)}
      </LineChart></ResponsiveContainer> : <div className="chart-unavailable"><span>—</span><small>Для графика нужны рассчитанные значения</small></div>}
    </article>;
  })}</div>;
}

export default function ResultsPanel({ runs, models, activeId, setActiveId, secondId, setSecondId, report, secondReport,
  comparison, comparisonError, reportError, pending, calculate, rows, answersLoading, answersError, conditionFilter, setConditionFilter, offset, setOffset }:
  { runs: Run[]; models: Model[]; activeId: string; setActiveId: (id: string) => void; secondId: string; setSecondId: (id: string) => void;
    report: Report | null; secondReport: Report | null; comparison: Comparison | null; comparisonError: string; reportError: string;
    pending: string; calculate: (id: string) => void; rows: Result[]; answersLoading: boolean; answersError: string;
    conditionFilter: string; setConditionFilter: (value: string) => void; offset: number; setOffset: (value: number) => void }) {
  const [tab, setTab] = useState(0);
  const [condition, setCondition] = useState('clean');
  const [selectedMetrics, setSelectedMetrics] = useState(['accuracy', 'macro_f1', 'mean_iou', 'mean_seconds']);
  const active = runs.find((run) => run.id === activeId);
  const reports = comparison?.runs || [report, secondReport].filter((item): item is Report => !!item);
  const conditions = Object.keys(report?.metrics?.conditions || {});
  useEffect(() => { if (conditions.length && !conditions.includes(condition)) setCondition(conditions[0]); }, [conditions.join('|'), condition]);
  useEffect(() => {
    const families = Object.values(report?.metrics?.conditions || {}).map((item) => item.metric_family);
    if (families.includes('imp_segmentation')) setSelectedMetrics(['mean_iou', 'dice', 'coverage', 'mean_seconds']);
    else if (families.includes('closed_form_classification')) setSelectedMetrics(['accuracy', 'macro_f1', 'coverage', 'mean_seconds']);
  }, [report?.run_id, report?.metrics_status]);
  const rowConditions = active ? [...(active.protocol.include_clean ? ['clean'] : []), ...active.protocol.distortions.map((chain) => chain.join('+'))] : [];
  return <section id="results" className="panel results-panel"><header className="results-heading"><div className="results-title"><BarChart3 size={22} /><h2>Результаты и метрики</h2></div>
    <div className="results-actions"><details className="metric-picker"><summary><SlidersHorizontal size={15} />Метрики<span className="count-badge">{selectedMetrics.length}</span><ChevronDown size={13} /></summary>
      <div className="metric-picker-menu">{METRICS.map((metric) => <label key={metric.key} className="checkbox-label" title={metric.hint}><input type="checkbox" checked={selectedMetrics.includes(metric.key)} onChange={() => setSelectedMetrics(selectedMetrics.includes(metric.key) ? selectedMetrics.filter((key) => key !== metric.key) : [...selectedMetrics, metric.key])} />{metric.name}</label>)}</div></details>
      <button type="button" className="secondary-button small-button" disabled={!report?.metrics} onClick={() => saveFile(JSON.stringify(comparison || report, null, 2), `report-${activeId.slice(0, 8)}.json`)}><Download size={15} />Отчёт</button></div></header>
    <div className="results-selection"><label>Основной запуск<select value={activeId} onChange={(event) => setActiveId(event.target.value)}><option value="">Выберите запуск</option>{runs.map((run) => <option key={run.id} value={run.id}>{shortModel(models.find((item) => item.id === run.model_config_id)?.name || 'Модель')} · {run.id.slice(0, 8)}</option>)}</select></label>
      <label>Второй запуск<select value={secondId} disabled={!active || active.status !== 'completed'} onChange={(event) => setSecondId(event.target.value)}><option value="">Без сравнения</option>{runs.filter((run) => run.id !== activeId && run.status === 'completed').map((run) =>
        <option key={run.id} value={run.id} disabled={run.protocol_sha256 !== active?.protocol_sha256}>{shortModel(models.find((item) => item.id === run.model_config_id)?.name || 'Модель')} · {run.id.slice(0, 8)}{run.protocol_sha256 !== active?.protocol_sha256 ? ' · другая выборка' : ''}</option>)}</select></label>
      <button type="button" className="secondary-button calculate-button" disabled={!active || active.status !== 'completed' || !!pending} onClick={() => calculate(activeId)}>{pending === 'metrics' ? <Busy /> : <RefreshCw size={15} />}{report?.metrics ? 'Обновить метрики' : 'Рассчитать метрики'}</button>
    </div>
    {secondReport && !secondReport.metrics && <div className="inline-info report-note">Для второго запуска ещё не рассчитаны метрики.<button type="button" className="text-button" disabled={!!pending} onClick={() => calculate(secondId)}>Рассчитать</button></div>}
    {reports.some((item) => item.mock) && <div className="fixture-banner">Тестовые ответы · эти значения не являются научными результатами</div>}
    {comparison && !comparison.quality_comparable && <div className="inline-info report-note"><Info size={16} />Показатели качества имеют разные определения или недоступны. Accuracy текстовых ответов и IoU карт оцениваются отдельно.</div>}
    {(reportError || comparisonError) && <div className="inline-error report-note" role="alert">{reportError || comparisonError}</div>}
    <div className="result-tabs" role="tablist" aria-label="Вид результатов">{TABS.map((label, index) => <button key={label} type="button" role="tab" aria-selected={tab === index} aria-controls={`result-tab-${index}`} id={`result-tab-button-${index}`} tabIndex={tab === index ? 0 : -1}
      className={tab === index ? 'active' : ''} onClick={() => setTab(index)} onKeyDown={(event) => { if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        const next = (index + (event.key === 'ArrowRight' ? 1 : -1) + TABS.length) % TABS.length;
        setTab(next); document.getElementById(`result-tab-button-${next}`)?.focus(); event.preventDefault();
      } }}>{label}</button>)}</div>
    <div className="results-body" role="tabpanel" id={`result-tab-${tab}`} aria-labelledby={`result-tab-button-${tab}`}>
      {tab === 3 ? activeId ? <Answers rows={rows} offset={offset} condition={conditionFilter} setCondition={setConditionFilter} setOffset={setOffset} conditions={rowConditions} loading={answersLoading} error={answersError} />
        : <Empty icon={FileChartColumn} title="Выберите запуск">Здесь будут снимки, ответы и карты сегментации.</Empty>
        : tab === 1 ? <MetricsTable reports={reports} /> : tab === 2 ? <ConditionCharts reports={reports} selectedMetrics={selectedMetrics} /> : <>
          <div className="comparison-toolbar"><div className="legend"><span><i style={{ background: COLORS[0] }} />{report ? shortModel(report.model.name) : 'Основной запуск'}</span>
            {secondId && <span><i style={{ background: COLORS[1] }} />{secondReport ? shortModel(secondReport.model.name) : 'Второй запуск'}</span>}</div>
            <label className="inline-field">Условие<select value={condition} disabled={!conditions.length} onChange={(event) => setCondition(event.target.value)}>
              {conditions.length ? conditions.map((key) => <option key={key} value={key}>{conditionLabel(key)}</option>) : <option value="clean">Оригинал</option>}</select></label></div>
          <div className="metric-grid">{selectedMetrics.map((key) => <MetricChart key={key} metricKey={key} reports={reports} condition={condition} />)}</div>
          {!selectedMetrics.length && <p className="list-empty">Выберите показатели в меню «Метрики».</p>}
          {!report?.metrics && <div className="metrics-hint"><FileChartColumn size={18} /><span>{active?.status === 'completed' ? 'Ответы собраны. Рассчитайте метрики, чтобы построить графики.' : activeId ? 'После завершения запуска здесь появятся рассчитанные показатели.' : 'Создайте запуск, соберите ответы и рассчитайте метрики.'}</span></div>}
          {comparison?.quality_comparable && <div className="contrast-summary">{comparison.contrasts.filter((item) => item.condition === condition).map((item) => <span key={item.condition}>Разница второй модели относительно первой: <b>{fmt(item.right_minus_left, 3)}</b>
            {item.paired_interval?.interval ? ` · 95% ДИ [${fmt(item.paired_interval.interval[0], 3)}; ${fmt(item.paired_interval.interval[1], 3)}]` : item.paired_interval ? ' · недостаточно независимых групп для интервала' : ''}</span>)}</div>}
          <MetricsTable reports={reports} />
        </>}
    </div>
  </section>;
}
