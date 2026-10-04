import type { Condition, Effect, Model, Protocol } from './types';

export const EFFECTS: { id: Effect; label: string; hint: string }[] = [
  { id: 'shear', label: 'Shear', hint: 'Плавные смещения строк и столбцов' },
  { id: 'radiation', label: 'Радиация', hint: 'Следы частиц на снимке' },
  { id: 'lens_flare', label: 'Lens flare', hint: 'Блики и засветка' },
  { id: 'hard_shadows', label: 'Жёсткие тени', hint: 'Области затенения' },
];
export const ALL_EFFECTS: Effect[] = ['shear', 'hard_shadows', 'lens_flare', 'radiation'];
export const initialProtocol: Protocol = {
  image_count: 10, split: 'test', seed: 42, include_clean: true,
  distortions: [...EFFECTS.map((item) => [item.id]), [...ALL_EFFECTS]],
  parameters: { shear: { max_shift_px: 4, control_spacing_px: 32, scan_axis: 'both' },
    radiation: { num_hits: 80, max_streak_length: 10 }, lens_flare: { strength: .55 }, hard_shadows: { strength: .65 } },
  prompt: 'Опиши видимые особенности поверхности на орбитальном снимке. Укажи неопределённость, если детали не различимы.',
  allowed_answers: [],
};
export const STATUS: Record<string, string> = {
  queued: 'В очереди', running: 'Выполняется', completed: 'Завершён', cancelled: 'Остановлен',
  interrupted: 'Прерван', blocked: 'Заблокирован', failed: 'Ошибка',
};
export const PHASE: Record<string, string> = { waiting: 'Ожидание обработчика', preparing: 'Подготовка снимков',
  loading: 'Загрузка модели', inference: 'Получение ответов', cancelling: 'Остановка после текущего снимка', finished: 'Готово', stopped: 'Обработка остановлена' };
export const shortModel = (name: string) => name === 'NASA-IBM-Lunar-Foundation-Model' ? 'NASA–IBM Lunar FM' : name;
export const conditionLabel = (key: string) => key === 'clean' ? 'Оригинал' :
  key.split('+').map((effect) => EFFECTS.find((item) => item.id === effect)?.label || effect).join(' + ');
export const compactCondition = (key: string) => key.includes('+') ? `Комбинация (${key.split('+').length})` : conditionLabel(key);
export const fmt = (value: number | null | undefined, places = 2) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('ru-RU', { minimumFractionDigits: places, maximumFractionDigits: places }) : '—';
export const sizeLabel = (bytes: number) => bytes < 1024 * 1024 ? `${fmt(bytes / 1024, 0)} КБ` : `${fmt(bytes / 1024 / 1024, 1)} МБ`;
export const modelPayload = (model: Model) => ({ name: model.name, version: model.version,
  model_id: model.model_id, backend: model.backend, revision: model.revision,
  options: model.options, seed: model.seed, max_new_tokens: model.max_new_tokens, enabled: model.enabled });
export function cleanProtocol(value: Protocol): Protocol {
  return { image_count: value.image_count, split: value.split, seed: value.seed,
    include_clean: value.include_clean, distortions: value.distortions.map((chain) => [...chain]),
    parameters: value.parameters, prompt: value.prompt, allowed_answers: [...value.allowed_answers] };
}
export const METRICS = [
  { key: 'accuracy', name: 'Accuracy', hint: 'Точность ответов из заданного списка, включая ошибки и невалидные ответы в знаменателе.', unit: '' },
  { key: 'macro_f1', name: 'F1 score', hint: 'Среднее F1 по классам с разметкой.', unit: '' },
  { key: 'mean_iou', name: 'Mean IoU', hint: 'Среднее пересечение / объединение по классам сегментации.', unit: '' },
  { key: 'dice', name: 'Dice · IMP', hint: 'Dice для класса IMP по успешным картам.', unit: '' },
  { key: 'pixel_accuracy', name: 'Pixel accuracy', hint: 'Точность классов на валидных пикселях успешных карт.', unit: '' },
  { key: 'coverage', name: 'Coverage', hint: 'Доля валидных текстовых ответов или успешно оценённых карт. Определение зависит от задачи.', unit: '' },
  { key: 'success_rate', name: 'Успешные ответы', hint: 'Доля запросов без ошибки выполнения.', unit: '' },
  { key: 'mean_seconds', name: 'Время ответа', hint: 'Среднее время успешного запроса без первоначальной загрузки модели.', unit: 'с' },
];
export function metricValue(condition: Condition | undefined, key: string): number | null {
  if (!condition) return null;
  let value: unknown;
  if (key === 'mean_seconds' || key === 'success_rate') value = condition.runtime[key];
  else if (key === 'coverage') value = condition.quality?.coverage ?? condition.quality?.prediction_coverage;
  else if (key === 'dice') value = (condition.quality?.per_class as Record<string, { dice?: number }> | undefined)?.IMP?.dice;
  else value = condition.quality?.[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}
export function saveFile(content: string, name: string, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export const csvCell = (value: unknown) => {
  let text = String(value ?? '');
  if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
};
