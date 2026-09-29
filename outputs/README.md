# Результаты запусков

- `runs/<run_id>/` — resolved config, environment, requests, responses, failures,
  timings и сведения о checkpoint/precision. Сырые ответы сохраняются неизменно.
- `reports/<run_id>/` — метрики, exclusions и paired comparisons.
- `figures/<run_id>/` — воспроизводимо построенные графики.

Каждый запуск получает новый ID; существующие результаты не перезаписываются.
Ground truth присоединяется в evaluation, а не в model request. Публикация
отобранных артефактов — отдельное решение после проверки лицензий.
