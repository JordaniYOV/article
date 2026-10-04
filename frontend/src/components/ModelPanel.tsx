import { useState } from 'react';
import { Box, Check, Pencil, Plus, Settings2, Trash2 } from 'lucide-react';
import { api } from '../api';
import { modelPayload, shortModel } from '../domain';
import type { Model } from '../types';
import { Busy, DropZone, Modal, Panel, type Perform } from './ui';

const defaultConfig = { name: 'SpaceLLaVA', version: 'v1', model_id: 'remyxai/SpaceLLaVA',
  backend: 'spacellava', revision: null as string | null, options: {} as Record<string, unknown>, seed: 42,
  max_new_tokens: 128, enabled: false };

export default function ModelPanel({ models, pending, perform, onSelect, error }:
  { models: Model[]; pending: string; perform: Perform; onSelect: (id: number) => void; error: (message: string) => void }) {
  const [editor, setEditor] = useState<ReturnType<typeof modelPayload> | null>(null);
  const [editId, setEditId] = useState<number | undefined>();
  const [options, setOptions] = useState('{}');
  const [removeId, setRemoveId] = useState<number | null>(null);
  function edit(value: ReturnType<typeof modelPayload>, id?: number) { setEditor(value); setEditId(id); setOptions(JSON.stringify(value.options, null, 2)); }
  async function file(value: File) {
    if (!value.name.toLowerCase().endsWith('.json') || value.size > 1024 * 1024) { error('Выберите JSON-конфигурацию до 1 МБ. Веса модели должны находиться на сервере.'); return; }
    try {
      const data = JSON.parse(await value.text());
      if (!data || typeof data !== 'object' || Array.isArray(data) || !data.name || !data.backend || !data.model_id) throw new Error('В конфигурации нужны name, model_id и backend.');
      const { id: _id, created_at: _created, updated_at: _updated, config_sha256: _hash, ...payload } = data;
      edit({ ...defaultConfig, ...payload });
    } catch (issue) { error(issue instanceof Error ? issue.message : 'Не удалось прочитать конфигурацию'); }
  }
  return <Panel id="models" icon={Box} title="Подключение моделей" number="02">
    <div className="panel-top-note"><span className="tiny-dot" />Локальные модели и готовые головы</div>
    <DropZone accept=".json,application/json" title="Перетащите конфигурацию" subtitle="JSON с настройками и путями к весам" onFile={(value) => void file(value)} disabled={!!pending} />
    <div className="model-help"><Settings2 size={17} /><p>Веса загружаются при старте задания. Для IBM подключается энкодер и готовая голова IMP.</p></div>
    <button type="button" className="secondary-button full-width" disabled={!!pending} onClick={() => void perform('defaults', () => api.defaults(), 'Конфигурации проекта подключены')}>
      {pending === 'defaults' ? <Busy /> : <Plus size={16} />}Подключить конфигурации проекта</button>
    <div className="section-label"><span>Подключённые модели <b>{models.length}</b></span><button type="button" className="text-button" onClick={() => edit({ ...defaultConfig })}><Plus size={13} />Новая</button></div>
    <div className="resource-list model-list">{models.length === 0 ? <p className="list-empty">SpaceLLaVA и NASA–IBM можно подключить кнопкой выше</p> : models.map((model) =>
      <div className="resource-row" key={model.id}><span className={`resource-icon model-icon ${model.backend === 'ibm_imp' ? 'blue' : ''}`}><Box size={21} /></span>
        <button type="button" className="resource-name" onClick={() => onSelect(model.id)} title={model.name}><strong>{shortModel(model.name)}</strong><small>{model.backend === 'ibm_imp' ? 'ViT · сегментация' : model.backend === 'mock' ? 'Тестовый адаптер' : 'VLM · текст'} · {model.version}</small></button>
        <span className={`model-state ${model.enabled ? 'enabled' : ''}`} title={model.enabled ? 'Запуск разрешён; загрузка весов проверяется worker' : 'Запуск выключен'}>{model.enabled ? <Check size={13} /> : <span className="tiny-dot" />}</span>
        <button type="button" className="icon-button" disabled={!!pending} onClick={() => edit(modelPayload(model), model.id)} aria-label={`Настроить ${model.name}`}><Pencil size={15} /></button>
        <button type="button" className="icon-button" disabled={!!pending} onClick={() => setRemoveId(model.id)} aria-label={`Удалить конфигурацию ${model.name}`}><Trash2 size={15} /></button>
      </div>)}</div>
    <p className="field-note">Одна модель на запуск. Для сравнения повторите ту же выборку на второй модели.</p>
    {editor && <Modal title={editId ? 'Настройки модели' : 'Новая конфигурация модели'} close={() => setEditor(null)}><form onSubmit={(event) => {
      event.preventDefault(); void perform('model', async () => {
        const parsed = JSON.parse(options);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Параметры должны быть JSON-объектом');
        const saved = await api.saveModel({ ...editor, options: parsed }, editId); onSelect(saved.id); setEditor(null);
      }, 'Конфигурация модели сохранена');
    }}>
      <div className="field-row"><label>Название<input value={editor.name} required maxLength={128} onChange={(event) => setEditor({ ...editor, name: event.target.value })} /></label>
        <label>Версия<input value={editor.version} required onChange={(event) => setEditor({ ...editor, version: event.target.value })} /></label></div>
      <label>Checkpoint ID<input value={editor.model_id} required onChange={(event) => setEditor({ ...editor, model_id: event.target.value })} /></label>
      <div className="field-row"><label>Система<select value={editor.backend} onChange={(event) => setEditor({ ...editor, backend: event.target.value })}><option value="spacellava">SpaceLLaVA · VLM</option><option value="ibm_imp">NASA–IBM · IMP head</option><option value="mock">Mock · тестовый адаптер</option></select></label>
        <label>Лимит токенов<input type="number" min={1} required value={editor.max_new_tokens} onChange={(event) => setEditor({ ...editor, max_new_tokens: Number(event.target.value) })} /></label></div>
      <label>Revision<input value={editor.revision || ''} placeholder="Закреплённая версия checkpoint" onChange={(event) => setEditor({ ...editor, revision: event.target.value || null })} /></label>
      <label>Пути к весам и параметры<textarea className="code-input" rows={9} value={options} onChange={(event) => setOptions(event.target.value)} spellCheck={false} /></label>
      <p className="field-note">Пути относятся к компьютеру, где работает бэкенд. Конфигурация использованного запуска фиксируется: для изменения создайте новую версию.</p>
      <label className="checkbox-label"><input type="checkbox" checked={editor.enabled} onChange={(event) => setEditor({ ...editor, enabled: event.target.checked })} />Разрешить запуск этой конфигурации</label>
      <button className="primary-button full-width" disabled={!!pending}>{pending === 'model' ? <Busy /> : <Check size={17} />}Сохранить конфигурацию</button>
    </form></Modal>}
    {removeId !== null && <Modal title="Удалить конфигурацию модели?" close={() => setRemoveId(null)}><p className="modal-description">Файлы весов сохранятся. Удаление конфигурации с сохранёнными запусками недоступно.</p>
      <div className="modal-actions"><button type="button" className="secondary-button" onClick={() => setRemoveId(null)}>Отмена</button><button type="button" className="danger-button" disabled={!!pending} onClick={() => void perform('delete', async () => { await api.deleteModel(removeId); setRemoveId(null); }, 'Конфигурация удалена')}>Удалить конфигурацию</button></div></Modal>}
  </Panel>;
}
