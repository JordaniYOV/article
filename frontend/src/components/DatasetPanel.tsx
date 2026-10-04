import { useState } from 'react';
import { ArrowRight, Database, FileArchive, FileText, FolderOpen, Link, Plus, Trash2, X } from 'lucide-react';
import { api } from '../api';
import { sizeLabel } from '../domain';
import type { Dataset, DatasetMetadata } from '../types';
import { Busy, DropZone, Modal, Panel, type Perform } from './ui';

export default function DatasetPanel({ datasets, pending, perform, onSelect, error }:
  { datasets: Dataset[]; pending: string; perform: Perform; onSelect: (id: number) => void; error: (message: string) => void }) {
  const [mode, setMode] = useState<'file' | 'url'>('file');
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [name, setName] = useState('');
  const [planet, setPlanet] = useState('moon');
  const [task, setTask] = useState('description');
  const [version, setVersion] = useState('v1');
  const [license, setLicense] = useState('');
  const [imp, setImp] = useState(false);
  const [local, setLocal] = useState(false);
  const [manifest, setManifest] = useState('');
  const [removeId, setRemoveId] = useState<number | null>(null);
  const metadata = (): DatasetMetadata => ({ name: name.trim(), version: version.trim(), planet,
    task_id: imp ? 'imp_segmentation' : task.trim(), provenance: license ? { license } : {},
    preprocessing: imp ? { input_protocol: 'curated_imp_png_reflectance_0_0.12_v1' } : {} });
  function chooseFile(value: File) {
    if (!value.name.toLowerCase().endsWith('.zip')) { error('Выберите ZIP-архив со снимками и manifest.jsonl.'); return; }
    if (value.size > 256 * 1024 * 1024) { error('Архив превышает 256 МБ. Подключите локальный manifest или измените лимит на сервере.'); return; }
    setFile(value); if (!name) setName(value.name.replace(/\.zip$/i, ''));
  }
  const submit = () => perform('dataset', async () => {
    const dataset = mode === 'file' ? await api.upload(file!, metadata()) : await api.fromUrl(url.trim(), metadata());
    onSelect(dataset.id); setFile(null); setUrl(''); setName('');
  }, 'Датасет подключён');
  return <Panel id="datasets" icon={Database} title="Загрузка датасета" number="01">
    <div className="segmented" aria-label="Способ загрузки датасета">
      <button type="button" className={mode === 'file' ? 'active' : ''} onClick={() => setMode('file')}><FileArchive size={15} />Архив</button>
      <button type="button" className={mode === 'url' ? 'active' : ''} onClick={() => setMode('url')}><Link size={15} />По ссылке</button>
    </div>
    {mode === 'file' ? <DropZone accept=".zip,application/zip" title="Перетащите датасет сюда" subtitle="Снимки + разметка в ZIP · до 256 МБ" onFile={chooseFile} disabled={pending === 'dataset'} />
      : <div className="url-box"><span className="upload-icon"><Link size={26} /></span><strong>Ссылка на ZIP-архив</strong>
        <label className="visually-hidden" htmlFor="dataset-url">Ссылка на датасет</label><input id="dataset-url" type="url" value={url} onChange={(event) => setUrl(event.target.value)} placeholder="https://…/dataset.zip" />
        <small>Прямая ссылка на архив со снимками</small></div>}
    {file && mode === 'file' && <div className="selected-file"><FileArchive size={20} /><span><strong>{file.name}</strong><small>{sizeLabel(file.size)} · готов к загрузке</small></span>
      <button type="button" className="icon-button" onClick={() => setFile(null)} aria-label="Убрать выбранный файл"><X size={15} /></button></div>}
    <div className="field-row"><label>Название датасета<input value={name} onChange={(event) => setName(event.target.value)} placeholder="Например, moon-test" required maxLength={128} /></label>
      <label className="planet-field">Планета<select value={planet} onChange={(event) => setPlanet(event.target.value)}><option value="moon">Луна</option><option value="mars">Марс</option></select></label></div>
    <details className="advanced"><summary>Разметка и происхождение</summary><div className="advanced-body">
      <div className="field-row"><label>Задача<input value={task} disabled={imp} onChange={(event) => setTask(event.target.value)} placeholder="presence" /></label><label>Версия<input value={version} onChange={(event) => setVersion(event.target.value)} /></label></div>
      <label>Лицензия<input value={license} onChange={(event) => setLicense(event.target.value)} placeholder="Например, CC-BY-4.0" /></label>
      <label className="checkbox-label"><input type="checkbox" checked={imp} onChange={(event) => setImp(event.target.checked)} />Лунные IMP, grayscale PNG 256×256</label>
      <p className="field-note">Для оценки качества нужен manifest с ответами или масками. IMP требует соответствующего протокола подготовки снимков.</p>
    </div></details>
    <button type="button" className="secondary-button full-width" disabled={!!pending || !name.trim() || !version.trim() || (!imp && !task.trim()) || (mode === 'file' ? !file : !/^https?:\/\//.test(url))} onClick={submit}>
      {pending === 'dataset' ? <Busy>Загрузка…</Busy> : <>Подключить датасет<ArrowRight size={16} /></>}</button>
    <div className="section-label"><span>Подключённые датасеты <b>{datasets.length}</b></span><button type="button" className="text-button" onClick={() => setLocal(true)}><Plus size={13} />Локальный</button></div>
    <div className="resource-list">{datasets.length === 0 ? <p className="list-empty">Загрузите архив или подключите датасет проекта</p> : datasets.map((dataset) =>
      <div className="resource-row" key={dataset.id}><span className="resource-icon"><FileText size={20} /></span><button type="button" className="resource-name" onClick={() => onSelect(dataset.id)} title={dataset.name}><strong>{dataset.name}</strong>
        <small>{dataset.planet === 'moon' ? 'Луна' : 'Марс'} · {dataset.task_id} · {dataset.version}{typeof dataset.provenance.sample_count === 'number' ? ` · ${dataset.provenance.sample_count} снимков` : ''}</small></button>
        <button type="button" className="icon-button" disabled={!!pending} onClick={() => setRemoveId(dataset.id)} aria-label={`Удалить регистрацию ${dataset.name}`}><Trash2 size={16} /></button></div>)}</div>
    {local && <Modal title="Подключить локальный датасет" close={() => setLocal(false)}><form onSubmit={(event) => {
      event.preventDefault(); void perform('dataset', async () => {
        const dataset = await api.saveDataset({ ...metadata(), manifest_path: manifest, splits: ['val', 'test'], conditions: ['clean'] });
        onSelect(dataset.id); setLocal(false);
      }, 'Локальный датасет подключён');
    }}><p className="modal-description">Укажите существующий manifest со снимками в папке data_orbital.</p>
      <label>Название<input value={name} required onChange={(event) => setName(event.target.value)} /></label>
      <label>Manifest<input value={manifest} required placeholder="data_orbital/…/manifest.jsonl" onChange={(event) => setManifest(event.target.value)} /></label>
      <div className="field-row"><label>Планета<select value={planet} onChange={(event) => setPlanet(event.target.value)}><option value="moon">Луна</option><option value="mars">Марс</option></select></label>
        <label>Задача<input value={task} required onChange={(event) => setTask(event.target.value)} /></label></div>
      <label className="checkbox-label"><input type="checkbox" checked={imp} onChange={(event) => setImp(event.target.checked)} />Протокол лунных IMP PNG</label>
      <button className="primary-button full-width" disabled={!!pending}>{pending === 'dataset' ? <Busy /> : <FolderOpen size={17} />}Подключить</button>
    </form></Modal>}
    {removeId !== null && <Modal title="Удалить регистрацию датасета?" close={() => setRemoveId(null)}><p className="modal-description">Файлы снимков сохранятся. Удалить регистрацию с сохранёнными запусками нельзя.</p>
      <div className="modal-actions"><button type="button" className="secondary-button" onClick={() => setRemoveId(null)}>Отмена</button><button type="button" className="danger-button" disabled={!!pending} onClick={() => void perform('delete', async () => { await api.deleteDataset(removeId); setRemoveId(null); }, 'Регистрация удалена')}>Удалить регистрацию</button></div></Modal>}
  </Panel>;
}
