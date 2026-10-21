# Искажённые Mars-снимки: один набор, последовательная оценка

Этот CLI **не обучает** модели. Он использует `best.pt`, исходные TRAIN mean/std,
размер входа, precision и порог, ранее выбранный на чистом val. Исходные результаты,
датасет и код `adaptation/` не изменяются. Код обучения входит в identity checkpoint,
поэтому новый pipeline намеренно расположен вне `adaptation/`.

## Одна команда для всех подходов

Из корня проекта, **в том же окружении и на том же GPU, где обучались модели**:

```powershell
$adaptPython = ".\.venv-adaptation\Scripts\python.exe"
& $adaptPython -B -m planetary_vlm.robustness run --split test --allow-test
```

В отличие от прежних команд обучения, здесь `test` явно разрешён пользователем.
Нельзя подбирать по его результатам новые thresholds, веса или параметры искажений
и затем называть этот test слепым. Оценённый ранее test уже не является новым holdout.
Для предварительного прогона используй `--split val` (это default).
Реальный GPU robustness smoke в этой реализации ещё требуется на твоём компьютере;
успешные software-тесты не гарантируют VRAM/скорость pretrained IBM/SAM.

Команда выполняет:

1. Проверку завершённости всех девяти обученных runs и исходного SAM smoke/config.
   SHA-256 checkpoint проверяется **до** его загрузки. Нужны доверенные собственные
   checkpoints, а не произвольные файлы из интернета.
2. Однократную материализацию общей папки `data_orbital/mars_corruptions_v1/`.
   Для каждого из 900 снимков выбранного split и двух TRAIN controls создаются
   одинаковые для всех моделей 13 вариантов native grayscale 512×512.
3. Проверку полной готовности всех файлов **до первого модельного прогона**.
4. По очереди: linear probe seeds 13/37/73 → frozen decoder seeds 13/37/73 →
   LoRA decoder seeds 13/37/73 → один SAM baseline. Каждый run — отдельный процесс,
   модель загружается один раз для всех его условий. Каждый процесс захватывает
   тот же OS GPU-lock, что обучение и UI worker. Останови другие benchmark jobs.
5. Inference-only smoke каждого процесса: два TRAIN controls (пустая/непустая
   каталожная маска), каждый вариант дважды. Проверяются native grid, бинарность,
   finite logits и повторяемость маски с допуском 0,1% пикселей; не «нужная точность».
   Модели с одинаковыми бинарными ответами на controls автоматически не бракуются.
6. Оценку и сохранение предсказанных масок, метрик по условиям и source groups,
   mean/sample std по всем seeds, paired-by-seed `clean − condition`.

900 × 13 = **11 700 предсказаний на один run**, 117 000 для девяти runs + SAM.
Это полный, потенциально долгий inference, особенно для SAM; он не запускается
при установке/просмотре плана. Время не измерялось. PNG читаются по одному,
но для файлов потребуется несколько GB; точный размер зависит от сжатия.

## Что именно искажается

Используются существующие `benchmark_services.common.transforms.apply_chain`
и `planetary_vlm.distortion`, без второй реализации:

| Условие | Уровень 1 / 2 / 3 |
|---|---|
| `clean` | Исходные PNG, побайтовая копия |
| `shear` | `max_shift_px=1 / 3 / 6`, control spacing 32 |
| `radiation` | 20 / 80 / 200 ярких событий, длина до 5, strength 1 |
| `lens_flare` | strength 0,15 / 0,35 / 0,60, radius 0,3 |
| `hard_shadows` | strength 0,25 / 0,50 / 0,75, один полигон |

Условия независимые, **не** последовательное накопление всех четырёх эффектов.
Random seed фиксирован для source ID и эффекта, одинаков для всех методов/seeds
обучения. Расположение теней/бликов и исходное случайное поле сдвига сохраняются
между уровнями. Для radiation растёт число событий. Фактический сдвиг может быть
уменьшен существующей функцией для ограничения локального градиента.

Это синтетические intensity stress tests на THEMIS thermal-IR, а не физически
калиброванные модели радиации, оптики или отбрасываемых рельефом теней. Уровни
заранее фиксированы в `protocol.toml`; не обещается монотонное падение качества.

При `shear` та же координатная карта применяется к GT с nearest-neighbor.
Вышедшие за границу пиксели получают **255 = ignore**, не фон и не кратер.
Они исключаются из TP/FP/FN, а их количество сохраняется. У фотометрических
эффектов GT неизменен: оценивается полное восстановление каталожных областей,
в том числе под наложенными тенями/бликами. Изменение области valid при shear
надо учитывать при интерпретации clean-relative drops.

**Маски никогда не передаются в энкодер/SAM.** Они доступны только подготовке
геометрически выровненных targets и отдельному оценщику. Каталожная маска не
является исчерпывающим census всех видимых кратеров; FP нельзя автоматически
называть hallucination или физической ошибкой.

## Plan, отдельная подготовка и pilot

План ничего не пишет, не импортирует Torch и не загружает модель:

```powershell
& $adaptPython -B -m planetary_vlm.robustness plan --split test --allow-test
```

Можно заранее создать/проверить весь набор на CPU, отдельно от оценки:

```powershell
& $adaptPython -B -m planetary_vlm.robustness prepare --split test --allow-test
& $adaptPython -B -m planetary_vlm.robustness verify --split test --allow-test
```

Для безопасного короткого **val pilot** с теми же всеми seeds и SAM:

```powershell
& $adaptPython -B -m planetary_vlm.robustness run --split val --limit 8 --bundle data_orbital/mars_corruptions_pilot8_v1 --output outputs/mars_robustness_pilot8_v1
```

`--limit` выбирает фиксированное подмножество по seed/ID, не по ответам/GT.
Для полного прогона используй другую папку и `--limit 0` (default).
Общая папка не должна смешивать val/test или pilot/full.

Исходный clean output по умолчанию `outputs/mars_adaptation_v1`; другая папка:
`--clean-output outputs/my_training_run`. Все seeds обязательны. Если SAM ещё
не подготовлен, **явно** ограничь список `--methods linear_probe frozen_decoder lora_decoder`.
Он не будет молча пропущен. Добавление SAM потом — отдельный output с новым
полным списком методов; уже созданный общий bundle можно использовать повторно.
Настройки SAM берутся из исходного `sam_baseline/smoke.json`, не из нового шаблона.

## Resume, окружения и логи

Повтори **ту же команду**: готовые варианты не генерируются заново; их hashes
проверяются. При прерывании подготовки используются готовые sidecar records.
При прерывании оценки проверяются сохранённые prediction PNG и заново считаются
их метрики, но модель для них повторно не вызывается. Процесс загружает модель
и повторяет inference smoke даже при полностью готовых prediction records.
Любой сбой останавливает очередь; следующие подходы не запускаются.
Ctrl+C завершает активный дочерний процесс. Нет автоматического OOM fallback,
смены precision/size, обучения, установки зависимостей или скачивания весов.

Нельзя обновлять Torch/NumPy/OpenCV или переустанавливать пакеты в исходном
training-окружении: его полный список версий входит в старые identities.
CPU-подготовке нужен существующий OpenCV (`cv2`); GPU-оценке готовых файлов он
не нужен. Если OpenCV отсутствует в training-окружении, выполни `prepare` в уже
имеющейся API-среде с Pillow/NumPy/OpenCV (тот же исходный checkout), а `run` —
в неизменённой `.venv-adaptation`. Готовый bundle переносит hashes и renderer
provenance между средами; набор не пересоздаётся. Новые пакеты не устанавливаются
автоматически. Не ослабляй проверку training identity ради запуска.

Файлы:

- `data_orbital/mars_corruptions_v1/identity.json`, `renderer.json`, `complete.json`:
  исходные manifests, параметры/seed, hashes кода и версии генератора.
- `inputs.jsonl` отдельно от `targets.jsonl`; `images/`, `targets/`, `records/`.
- `outputs/mars_robustness_v1/logs/*.log`: полный stdout/stderr subprocess.
- `outputs/mars_robustness_v1/events.log` и `<method>/seed_<n>/events.log`:
  этапы и прогресс; одновременно выводятся в консоль.
- `<method>/seed_<n>/<condition>/report.json`, `masks/`, `records/`, `progress.json`.
  У SAM директория без seed.
- `outputs/mars_robustness_v1/summary.json` и `summary.md`: итоговая таблица,
  все seeds, clean-relative изменения без выбора лучшего seed.

Для пересборки сводки без моделей: та же команда с `summarize` вместо `run`.
Менять источники, условия, seed, код преобразований или выборку в старой папке
нельзя: потребуется новый versioned bundle и output. Оригиналы не удаляются.
