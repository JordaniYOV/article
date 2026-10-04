import { useEffect, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, ImageIcon, Layers3 } from 'lucide-react';
import { assetUrl } from '../api';
import { conditionLabel, fmt } from '../domain';
import type { Result } from '../types';
import { Busy, Empty } from './ui';

function ResultImage({ result, overlay }: { result: Result; overlay: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [failed, setFailed] = useState(false);
  const [mapError, setMapError] = useState(false);
  useEffect(() => {
    if (!overlay || !result.map_url || !canvas.current) return;
    let alive = true;
    const map = new Image(); map.crossOrigin = 'anonymous';
    map.onload = () => {
      if (!alive || !canvas.current) return;
      try {
        const context = canvas.current.getContext('2d'); if (!context) return;
        canvas.current.width = map.naturalWidth; canvas.current.height = map.naturalHeight;
        context.drawImage(map, 0, 0);
        const pixels = context.getImageData(0, 0, map.naturalWidth, map.naturalHeight);
        for (let i = 0; i < pixels.data.length; i += 4) {
          const imp = pixels.data[i] === 1;
          pixels.data[i] = 139; pixels.data[i + 1] = 85; pixels.data[i + 2] = 255; pixels.data[i + 3] = imp ? 190 : 0;
        }
        context.putImageData(pixels, 0, 0); setMapError(false);
      } catch { setMapError(true); }
    };
    map.onerror = () => { if (alive) setMapError(true); };
    map.src = assetUrl(result.map_url);
    return () => { alive = false; map.onload = null; map.onerror = null; };
  }, [result.map_url, overlay]);
  return <div className="answer-image">{failed ? <div className="image-unavailable"><ImageIcon size={27} />Снимок недоступен</div> :
    <img src={assetUrl(result.image_url)} alt={`Орбитальный снимок ${result.source_sample_id}, ${conditionLabel(result.condition_id)}`} loading="lazy" onError={() => setFailed(true)} />}
    {overlay && result.map_url && !mapError && <canvas ref={canvas} className="map-overlay" aria-label="Прогноз сегментации IMP" />}
    {overlay && result.map_url && <span className="image-label"><span className="tiny-dot" />{mapError ? 'Карта недоступна' : 'IMP · прогноз'}</span>}
  </div>;
}

export default function Answers({ rows, offset, condition, setCondition, setOffset, conditions, loading, error }:
  { rows: Result[]; offset: number; condition: string; setCondition: (value: string) => void; setOffset: (value: number) => void;
    conditions: string[]; loading: boolean; error: string }) {
  const [overlay, setOverlay] = useState(true);
  return <div className="answers-view"><div className="table-toolbar"><label className="inline-field">Условие<select value={condition} onChange={(event) => { setCondition(event.target.value); setOffset(0); }}><option value="">Все условия</option>
    {conditions.map((value) => <option key={value} value={value}>{conditionLabel(value)}</option>)}</select></label>
    {rows.some((row) => row.map_url) && <label className="checkbox-label"><input type="checkbox" checked={overlay} onChange={(event) => setOverlay(event.target.checked)} /><Layers3 size={15} />Показывать прогноз IBM</label>}</div>
    {error ? <div className="inline-error" role="alert">{error}</div> : loading ? <div className="loading-state"><Busy>Загрузка ответов…</Busy></div> : rows.length === 0 ? <Empty icon={ImageIcon} title="Ответов пока нет">Они появятся по мере обработки снимков.</Empty> :
      <div className="answer-grid">{rows.map((row) => <article key={row.id} className="answer-card"><ResultImage result={row} overlay={overlay} />
        <div className="answer-content"><div className="answer-heading"><strong>{row.source_sample_id}</strong><span className={`answer-status ${row.status}`}>{row.status === 'ok' ? 'Готово' : row.status === 'error' ? 'Ошибка' : 'Не поддерживается'}</span></div>
          <div className="answer-meta"><span title={conditionLabel(row.condition_id)}>{conditionLabel(row.condition_id)}</span><span>{fmt(row.elapsed_seconds)} с</span></div>
          {row.mock && <span className="fixture-tag">Тестовый ответ</span>}
          {row.output_type === 'imp_segmentation' && row.status === 'ok' ? <p>Карта классов «фон / IMP». Фиолетовым отмечен прогноз модели.</p> : <p className="response-text">{row.error_message || row.raw_response || 'Пустой ответ'}</p>}
          <details className="raw-response"><summary>{row.output_type === 'imp_segmentation' ? 'Данные ответа' : 'Полный ответ'}</summary><pre>{row.raw_response || row.error_message}</pre></details>
        </div></article>)}</div>}
    <div className="pagination"><span>{rows.length ? `${offset + 1}–${offset + rows.length}` : '0'} ответов</span><div>
      <button type="button" className="secondary-button small-button" disabled={offset === 0 || loading} onClick={() => setOffset(Math.max(0, offset - 8))}><ChevronLeft size={15} />Назад</button>
      <button type="button" className="secondary-button small-button" disabled={rows.length < 8 || loading} onClick={() => setOffset(offset + 8)}>Далее<ChevronRight size={15} /></button></div></div>
  </div>;
}
