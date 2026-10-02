# Результаты запусков

Полные системы используют отдельные каталоги:

- `SpaceLLaVA/<run_id>/predictions.jsonl` — тексты SpaceLLaVA.
- `NASA-IBM-Lunar-Foundation-Model/<run_id>/predictions.jsonl` — ссылки и hashes карт IBM.
- `NASA-IBM-Lunar-Foundation-Model/<run_id>/maps/` — PNG class IDs и NPZ logits/probabilities.

Подготовка выполняется командой `python -m planetary_vlm full-systems --prepare-only`.
Описание запуска: `src/planetary_vlm/inference/FULL_SYSTEMS.md`.
Каталоги созданы; научных ответов моделей пока нет.

- `runs/<run_id>/` — resolved config, environment, requests, responses, failures,
  timings и сведения о checkpoint/precision. Сырые ответы сохраняются неизменно.
- `reports/<run_id>/` — метрики, exclusions и paired comparisons.
- `figures/<run_id>/` — воспроизводимо построенные графики.

Каждый запуск получает новый ID; существующие результаты не перезаписываются.
Ground truth присоединяется в evaluation, а не в model request. Публикация
отобранных артефактов — отдельное решение после проверки лицензий.
