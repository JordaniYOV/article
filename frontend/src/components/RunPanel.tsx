import { useEffect, useState } from 'react';
import { ArrowDown, ArrowUp, Check, Layers3, Play, Plus, Settings2, X } from 'lucide-react';
import { ALL_EFFECTS, cleanProtocol, conditionLabel, EFFECTS, shortModel } from '../domain';
import type { BuiltinPlanet, BuiltinSelection, BuiltinSubset, Dataset, Effect, Model, Planet, Protocol, Run } from '../types';
import { Busy, Panel } from './ui';

export default function RunPanel({ datasets, models, runs, modelId, datasetId, setModelId, setDatasetId,
  protocol, setProtocol, reuseId, setReuseId, pending, start, error, builtins, builtinSelection, chooseBuiltin }:
  { datasets: Dataset[]; models: Model[]; runs: Run[]; modelId: number; datasetId: number;
    setModelId: (id: number) => void; setDatasetId: (id: number) => void;
    protocol: Protocol; setProtocol: (value: Protocol) => void; reuseId: string; setReuseId: (id: string) => void;
    pending: string; start: (protocol: Protocol) => void; error: (message: string) => void;
    builtins: BuiltinPlanet[]; builtinSelection: BuiltinSelection | null; chooseBuiltin: (value: BuiltinSelection) => void }) {
  const [draftChain, setDraftChain] = useState<Effect[]>([]);
  const [parametersText, setParametersText] = useState(JSON.stringify(protocol.parameters, null, 2));
  const [answersText, setAnswersText] = useState(protocol.allowed_answers.join(', '));
  useEffect(() => { setParametersText(JSON.stringify(protocol.parameters, null, 2)); }, [protocol.parameters]);
  useEffect(() => { setAnswersText(protocol.allowed_answers.join(', ')); }, [protocol.allowed_answers]);
  const model = models.find((item) => item.id === modelId);
  const dataset = datasets.find((item) => item.id === datasetId);
  const builtinPlanet = builtins.find((item) => item.planet === builtinSelection?.planet);
  const builtinSubset = builtinPlanet?.subsets.find((item) => item.id === builtinSelection?.subset);
  const locked = !!reuseId;
  const splitCounts = dataset?.provenance.split_counts as Record<string, number> | undefined;
  const sampleCount = splitCounts?.[protocol.split] ?? dataset?.provenance.sample_count;
  const sampleLimit = typeof sampleCount === 'number' && sampleCount > 0 ? Math.min(10000, sampleCount) : 10000;
  const variants = protocol.distortions.length + Number(protocol.include_clean);
  const valid = model?.enabled && dataset && (!builtinSelection || (builtinSubset?.status === 'ready' && builtinSubset.dataset_id === datasetId)) && protocol.image_count >= 1 && protocol.image_count <= sampleLimit &&
    Number.isInteger(protocol.image_count) && Number.isInteger(protocol.seed) && protocol.seed >= 0 &&
    protocol.seed <= 2**32 - 1 && variants > 0 && protocol.prompt.trim() && dataset.splits.includes(protocol.split);
  function toggle(chain: Effect[]) {
    const exists = protocol.distortions.some((value) => value.join('+') === chain.join('+'));
    setProtocol({ ...protocol, distortions: exists ? protocol.distortions.filter((value) => value.join('+') !== chain.join('+')) : [...protocol.distortions, chain] });
  }
  const selected = (chain: Effect[]) => protocol.distortions.some((value) => value.join('+') === chain.join('+'));
  const reuse = (id: string) => {
    setReuseId(id);
    const run = runs.find((item) => item.id === id);
    if (run) { setDatasetId(run.dataset_config_id); setProtocol(cleanProtocol(run.protocol)); }
  };
  return <Panel id="launch" icon={Play} title="Сбор ответов" number="03" className="run-panel">
    <form onSubmit={(event) => {
      event.preventDefault();
      try {
        const parameters = JSON.parse(parametersText);
        if (!parameters || typeof parameters !== 'object' || Array.isArray(parameters)) throw new Error('Параметры искажений должны быть JSON-объектом.');
        const allowed_answers = answersText.split(',').map((item) => item.trim()).filter(Boolean);
        const normalized = allowed_answers.map((item) => item.toLowerCase().replace(/\s+/g, ' '));
        if (new Set(normalized).size !== normalized.length) throw new Error('Варианты ответов должны быть уникальными.');
        start({ ...protocol, parameters, allowed_answers });
      } catch (issue) { error(issue instanceof Error ? issue.message : 'Проверьте параметры'); }
    }}>
      <label>Выберите модель<select value={modelId || ''} required onChange={(event) => setModelId(Number(event.target.value))}>
        <option value="" disabled>Выберите конфигурацию</option>{models.map((item) => <option key={item.id} value={item.id}>{shortModel(item.name)} · {item.version}{!item.enabled ? ' · выключена' : ''}</option>)}
      </select></label>
      <div className="output-note"><span className="tiny-dot" />{model?.backend === 'ibm_imp' ? 'ViT · карты сегментации IMP' : 'VLM · текстовые ответы'}<span>{model ? `${model.max_new_tokens} токенов` : 'Одна модель на запуск'}</span></div>
      <label>Повторить выборку<select value={reuseId} onChange={(event) => reuse(event.target.value)}><option value="">Новая выборка снимков</option>
        {runs.map((run) => <option key={run.id} value={run.id}>{run.id.slice(0, 8)} · {run.protocol.image_count} снимков · {shortModel(models.find((item) => item.id === run.model_config_id)?.name || 'Модель')}</option>)}
      </select></label>
      {locked && <div className="inline-info"><Layers3 size={15} />Выборка и параметры сохранённого запуска закреплены.</div>}
      <fieldset disabled={locked || !!pending}>
        <label>Выберите датасет<select value={builtinSelection ? `builtin:${builtinSelection.planet}` : datasetId || ''} required onChange={(event) => {
          if (event.target.value.startsWith('builtin:')) {
            const planet = event.target.value.slice(8) as Planet;
            chooseBuiltin({ planet, subset: planet === 'moon' && model?.backend === 'ibm_imp' ? 'imp' : 'segmentation' });
            return;
          }
          const id = Number(event.target.value); setDatasetId(id);
          const value = datasets.find((item) => item.id === id);
          if (value && !value.splits.includes(protocol.split)) setProtocol({ ...protocol, split: value.splits.includes('test') ? 'test' : 'val' });
        }}><option value="" disabled>Выберите датасет</option>
          <optgroup label="Датасеты проекта"><option value="builtin:mars">Марс</option><option value="builtin:moon">Луна</option></optgroup>
          <optgroup label="Подключённые выборки">{datasets.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.planet === 'moon' ? 'Луна' : 'Марс'}</option>)}</optgroup>
        </select></label>
        {builtinSelection && <div className="builtin-subset"><label>Подвыборка<select value={builtinSelection.subset} onChange={(event) => chooseBuiltin({ ...builtinSelection, subset: event.target.value as BuiltinSubset })}>
          <option value="segmentation">Снимки с масками</option><option value="height">Снимки с картами высот</option>
          {builtinSelection.planet === 'moon' && <option value="imp">IMP · готовая голова IBM</option>}
        </select></label>
          <div className="builtin-state" role="status">{pending === 'builtin' ? <Busy>Подключение выборки…</Busy> :
            builtinSubset?.status === 'ready' && datasetId ? <>Локальные данные · {builtinSubset.sample_count} снимков. {Object.entries(builtinSubset.split_counts).map(([split, count]) => `${split}: ${count}`).join(' · ')}</> :
            builtinSubset?.status === 'queued' || builtinSubset?.status === 'downloading' ? <Busy>Загрузка и подготовка · {builtinSubset.completed} / {builtinSubset.total}</Busy> :
            builtinSubset?.error || 'Локальных файлов нет. Подготовка запускается автоматически при выборе.'}</div>
          {builtinSubset?.status === 'failed' && <button type="button" className="text-button" onClick={() => chooseBuiltin(builtinSelection)}>Повторить подготовку</button>}
          <p className="field-note">Маски и карты высот сохраняются как разметка; модель получает снимок.</p>
        </div>}
        <div className="field-row triple"><label>Снимков<input type="number" min={1} max={sampleLimit} required value={protocol.image_count} onChange={(event) => setProtocol({ ...protocol, image_count: Number(event.target.value) })} /></label>
          <label>Выборка<select value={protocol.split} onChange={(event) => setProtocol({ ...protocol, split: event.target.value as 'val' | 'test' })}>
            {(['test', 'val'] as const).filter((value) => !dataset || dataset.splits.includes(value)).map((value) => <option key={value} value={value}>{value}</option>)}
          </select></label><label>Seed<input type="number" min={0} max={2**32 - 1} required value={protocol.seed} onChange={(event) => setProtocol({ ...protocol, seed: Number(event.target.value) })} /></label></div>
        {protocol.image_count > sampleLimit && <p className="warning-text">{splitCounts ? `В выборке ${protocol.split} доступно` : 'В датасете всего'} {sampleLimit} снимков. Уменьшите выборку.</p>}
        <div className="distortion-box"><div className="control-heading"><Settings2 size={16} /><strong>Условия эксперимента</strong></div>
          <label className="checkbox-label clean-toggle"><input type="checkbox" checked={protocol.include_clean} onChange={(event) => setProtocol({ ...protocol, include_clean: event.target.checked })} />Оригинальные снимки<span className="mini-tag">baseline</span></label>
          <div className="effect-grid">{EFFECTS.map((effect) => <label key={effect.id} title={effect.hint} className={`effect-pill ${selected([effect.id]) ? 'selected' : ''}`}>
            <input type="checkbox" checked={selected([effect.id])} onChange={() => toggle([effect.id])} /><span className="pill-check">{selected([effect.id]) && <Check size={11} />}</span>{effect.label}</label>)}</div>
          <label className="checkbox-label combination-toggle"><input type="checkbox" checked={selected(ALL_EFFECTS)} onChange={() => toggle([...ALL_EFFECTS])} />Комбинация всех четырёх</label>
          <details className="advanced"><summary>Свои комбинации и сила искажений</summary><div className="advanced-body">
            <p className="field-note">Выбирайте эффекты в порядке применения.</p>
            <div className="chain-selector">{EFFECTS.map((effect) => <button type="button" className={draftChain.includes(effect.id) ? 'selected' : ''} key={effect.id} onClick={() => setDraftChain(draftChain.includes(effect.id) ? draftChain.filter((item) => item !== effect.id) : [...draftChain, effect.id])}>{effect.label}</button>)}</div>
            {draftChain.length > 0 && <ol className="chain-order">{draftChain.map((effect, index) => <li key={effect}><span>{index + 1}. {EFFECTS.find((item) => item.id === effect)?.label}</span>
              <button type="button" className="icon-button" disabled={index === 0} aria-label={`Поднять ${effect}`} onClick={() => { const copy = [...draftChain]; [copy[index - 1], copy[index]] = [copy[index], copy[index - 1]]; setDraftChain(copy); }}><ArrowUp size={13} /></button>
              <button type="button" className="icon-button" disabled={index === draftChain.length - 1} aria-label={`Опустить ${effect}`} onClick={() => { const copy = [...draftChain]; [copy[index], copy[index + 1]] = [copy[index + 1], copy[index]]; setDraftChain(copy); }}><ArrowDown size={13} /></button></li>)}</ol>}
            <button type="button" className="secondary-button small-button" disabled={draftChain.length < 2 || selected(draftChain)} onClick={() => { toggle([...draftChain]); setDraftChain([]); }}><Plus size={14} />Добавить комбинацию</button>
            {protocol.distortions.filter((chain) => chain.length > 1).map((chain) => <div className="chain-item" key={chain.join('+')}><span>{conditionLabel(chain.join('+'))}</span><button type="button" className="icon-button" aria-label={`Убрать комбинацию ${chain.join('+')}`} onClick={() => toggle(chain)}><X size={13} /></button></div>)}
            <label>Параметры искажений<textarea className="code-input" rows={8} spellCheck={false} value={parametersText} onChange={(event) => setParametersText(event.target.value)} /></label>
          </div></details>
        </div>
        <details className="advanced prompt-settings"><summary>Вопрос и формат ответа</summary><div className="advanced-body">
          <label>Вопрос модели<textarea rows={3} required value={protocol.prompt} onChange={(event) => setProtocol({ ...protocol, prompt: event.target.value })} /></label>
          <label>Варианты ответов<input value={answersText} onChange={(event) => setAnswersText(event.target.value)} placeholder="Например: YES, NO" /></label>
          <p className="field-note">Оставьте пустым для свободного описания. Accuracy/F1 требуют заданных вариантов и соответствующих ответов в разметке.</p>
        </div></details>
      </fieldset>
      {model && !model.enabled && <p className="warning-text">Включите запуск в настройках модели.</p>}
      {model?.backend === 'mock' && <p className="warning-text">Тестовый адаптер: ответы не являются научными результатами.</p>}
      <div className="run-estimate"><span>{protocol.image_count > 0 ? protocol.image_count : '—'} снимков × {variants} условий</span><strong>{protocol.image_count * variants || 0} ответов</strong></div>
      <button className="primary-button full-width launch-button" disabled={!!pending || !valid}>
        {pending === 'start' ? <Busy>Создание запуска…</Busy> : <><Play size={17} fill="currentColor" />Запустить сбор ответов</>}</button>
      <p className="field-note centered">Ответы сохраняются по мере обработки</p>
    </form>
  </Panel>;
}
