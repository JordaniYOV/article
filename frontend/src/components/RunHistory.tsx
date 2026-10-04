import { CirclePause, Clock3, Play, RotateCcw } from 'lucide-react';
import { PHASE, shortModel, STATUS } from '../domain';
import type { Model, Run } from '../types';
import { Busy } from './ui';

export default function RunHistory({ runs, models, activeId, select, pending, control }:
  { runs: Run[]; models: Model[]; activeId: string; select: (id: string) => void; pending: string;
    control: (id: string, action: 'cancel' | 'resume') => void }) {
  return <section id="history" className="panel history-panel"><header className="history-heading"><h2><Clock3 size={18} />История запусков<span className="count-badge">{runs.length}</span></h2>
    <span className="field-note">Последовательная обработка моделей</span></header>
    {runs.length === 0 ? <div className="history-empty"><span className="tiny-dot" />Создайте первый запуск — здесь появятся прогресс и сохранённые ответы.</div> :
      <div className="run-list">{runs.map((run) => <div key={run.id} className={`run-row ${run.id === activeId ? 'active' : ''}`}>
        <button type="button" className="run-select" onClick={() => select(run.id)} aria-label={`Открыть запуск ${run.id.slice(0, 8)}`}>
          <span className={`run-icon ${run.status === 'running' ? 'processing' : ''}`}><Play size={16} /></span>
          <span className="run-title"><strong>{shortModel(models.find((item) => item.id === run.model_config_id)?.name || 'Модель')}<code>{run.id.slice(0, 8)}</code></strong>
            <small>{run.protocol.image_count} снимков · {PHASE[run.phase] || run.phase}</small></span>
          <span className={`status-badge ${run.status}`}>{run.status === 'running' && <span className="tiny-dot" />}{STATUS[run.status] || run.status}</span>
          <span className="run-progress"><span>{run.completed} / {run.total}</span><span className="progress-track"><span style={{ width: `${Math.min(100, run.progress.percent)}%` }} /></span></span>
        </button>
        {(run.status === 'queued' || run.status === 'running') && <button type="button" className="icon-button" disabled={!!pending || run.phase === 'cancelling'} onClick={() => control(run.id, 'cancel')} aria-label={`Остановить запуск ${run.id.slice(0, 8)}`}>
          {pending === run.id ? <Busy /> : <CirclePause size={19} />}</button>}
        {['cancelled', 'interrupted', 'blocked', 'failed'].includes(run.status) && <button type="button" className="icon-button" disabled={!!pending} onClick={() => control(run.id, 'resume')} aria-label={`Продолжить запуск ${run.id.slice(0, 8)}`}>
          {pending === run.id ? <Busy /> : <RotateCcw size={18} />}</button>}
      </div>)}</div>}
    {runs.find((run) => run.id === activeId)?.error_message && <div className="run-error" role="alert">{runs.find((run) => run.id === activeId)?.error_message}</div>}
  </section>;
}
