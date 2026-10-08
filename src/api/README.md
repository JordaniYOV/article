# API оценки моделей

Один запуск — одна модель, один датасет, фиксированная выборка снимков и выбранные
цепочки искажений. Отдельный worker получает ответы последовательно. API возвращает
прогресс, результаты и JSON отчёта с таблицами и данными для графиков будущего сайта.

## Запуск из корня проекта

```powershell
& .\.venv-benchmark\Scripts\python.exe -m pip install -r requirements/app.txt
& .\.venv-benchmark\Scripts\python.exe -m pip install --no-deps -e .
& .\.venv-benchmark\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

В другом терминале:

```powershell
& .\.venv-benchmark\Scripts\python.exe -m api.worker
```

`python -m api.worker --once` обрабатывает одно задание. Без worker задания остаются
queued. Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
API стартует без весов и GPU. Веса и зависимости моделей устанавливаются отдельно;
совместимость RTX 5070 Ti с реальными моделями до smoke test не подтверждена.
Поддержаны SpaceLLaVA (`spacellava`), IBM IMP head (`ibm_imp`) и явно тестовый VLM `mock`.
Для новой архитектуры нужен адаптер: одного checkpoint ID недостаточно.

## Таблицы и сохранение

| Таблица | Содержимое |
|---|---|
| model_configs | Название/version, backend, локальные weights/revisions/options и hash |
| dataset_configs | Manifest, planet/task, splits, preprocessing, license/provenance |
| runs | Очередь одной модели, выборка, протокол, прогресс и остановка |
| model_results | Raw response, статус, время, source/group/split, карты и hashes |
| metric_analyses | Метрики, версия метода, IDs/hash результатов, признак mock |

SQLite: `.state/planetary.sqlite3`. ZIP-датасеты: `data_orbital/uploads/<id>/`.
Подготовленные входы: `data_orbital/inference/api/<run_id>/images/`.
Результаты: `outputs/<название модели>/<run_id>/`:

- run.json — конфигурация и версии библиотек;
- selection.json — источники, splits/groups/hashes и evaluation-only metadata;
- requests.jsonl — только разрешённые модели входы, без ground truth;
- loading.json — provenance загрузки весов;
- predictions.jsonl — ответы;
- status.json — состояние после остановки или завершения;
- metrics.json — после расчёта метрик;
- maps/ — PNG классов и NPZ logits/probabilities IBM.

SQLite — основной журнал: ответ и счётчик прогресса сохраняются одной транзакцией.
Worker восстанавливает predictions.jsonl из базы при продолжении. Конфигурации
фиксируются сразу после создания запуска; изменение требует новой версии.
Управляемые worker ответы нельзя подставить через CRUD или удалить отдельно.

## 1. Датасет и конфигурация модели

### Постоянные датасеты Марса и Луны

`GET /datasets/builtin` всегда возвращает Марс и Луну с подвыборками
`segmentation`, `height` и lunar `imp`. Чтение каталога не запускает загрузки.
Локальные данные находятся в `data_orbital/orbital_300_v3/<planet>/`.

`POST /datasets/builtin/{planet}/ensure` с `{"subset":"segmentation"}`:

- **200:** локальная выборка проверена по SHA-256 и зарегистрирована в SQLite;
  возвращаются `status:ready` и `dataset`. Повторное обращение использует тот же ID.
- **202:** нужные файлы отсутствуют; запущено восстановление в фоне,
  возвращается `job_id`. Состояние — `GET /datasets/builtin/jobs/{job_id}`
  или общий каталог; модель до подготовки не запускается.

Восстановление использует закреплённые source IDs/revisions, существующие
конвертеры из `scripts/` и описание выборки в
`configs/datasets/orbital_builtin_recovery.json`. Этот файл содержит только
метаданные и хеши; изображения, маски и массивы в него не включены.
Не выполняется новая случайная выборка и не подменяются отсутствующие данные.
Каждый восстановленный артефакт должен совпасть с сохранённым SHA-256.
Для четырёх Mars-Bench boulder test-пар встроенные в Parquet TIFF сохраняются
в PNG, как при исходной сборке коллекции; исходные TIFF-байты не совпадают с
SHA-256 подготовленных PNG. Преобразование сохраняет значения пикселей и классов
маски, после него по-прежнему проверяется исходный записанный SHA-256.
Используются только выбранные val/test кадры; для Parquet требуется скачать
содержащий их исходный shard. Полный набор, train и веса моделей не загружаются.
Готовые локальные файлы не перезаписываются; повреждённые хеши дают ошибку.

Для сетевого восстановления установите `requirements/orbital.txt` в окружение
API. Каталог и подключение уже существующих файлов не требуют pyarrow/h5py.

Если загрузка падает с `CERTIFICATE_VERIFY_FAILED: unable to get local issuer
certificate`, обновите корневые сертификаты в том же окружении, где работает API:

```powershell
& .\.venv-benchmark\Scripts\python.exe -m pip install --upgrade certifi
```

Перезапустите API и повторите загрузку в интерфейсе. HTTPS использует системные
корневые сертификаты и дополнительно `certifi`. Если сеть использует прокси или
антивирусную HTTPS-фильтрацию, задайте в терминале API
`$env:SSL_CERT_FILE = "C:\certificates\trusted-ca-bundle.pem"` с доверенным PEM
CA-файлом этой сети перед запуском API. Явный файл дополняет системные корни;
`certifi` тогда не добавляется. Проверка цепочки и имени сервера всегда включена.

В этой локальной версии фоновые загрузки исполняются в одном процессе API:
при перезапуске задача отмечается прерванной, повторный выбор продолжает по
сохранённым файлам. Для загрузки датасета GPU worker не нужен.
Кэш исходных файлов — `data_orbital/.builtin-cache/`, журнал задач —
`.state/builtin-datasets/`. Ошибки доступа к источнику показываются в интерфейсе.

На каждую планету создаются отдельные immutable конфигурации для масок и высот.
Для IBM только lunar `imp` объявляет нужный входной протокол; другие маски не
интерпретируются как IMP. Height/DEM метаданные сохраняются отдельно; обычный
inference получает только снимок. DEM не становится автоматически depth GT,
а Mars MMLSv2 source-scaled высоты не интерпретируются как метры.

`POST /configurations/import-defaults` импортирует две модели и лунный IMP поднабор
из configs/full_systems.toml, возвращает model_config_ids/dataset_config_ids.
Defaults имеют enabled:false: подготовьте локальные assets, затем включите модель
через `PUT /model-configs/{id}` (полное тело конфигурации с enabled:true).
Для собственного адаптера используйте `POST /model-configs`.

Существующий manifest регистрируется через `POST /dataset-configs`. Пути должны
быть внутри data_orbital. Для новых API-запусков используется исходный manifest;
старые CLI requests не переиспользуются. Отдельный targets_path необходимо сначала
перенести в source manifest. Из смешанного curated manifest IMP выбирается по
задаче, источнику, eligibility и split.

### Загрузка ZIP

`POST /datasets/upload`: multipart поля file (ZIP) и metadata (JSON-строка):

```json
{"name":"moon-demo","version":"v1","planet":"moon","task_id":"presence","provenance":{"license":"CC-BY-4.0"}}
```

Структура архива:

```text
manifest.jsonl
images/001.png
images/002.png
masks/002.png
```

Пример manifest.jsonl (две отдельные строки):

```json
{"sample_id":"s1","image":"images/001.png","source_split":"test","scene_group_id":"scene1","answer":"YES","derivation_version":"manual-v1"}
{"sample_id":"s2","image":"images/002.png","source_split":"test","scene_group_id":"scene2","mask":"masks/002.png","classes":["Background","IMP"],"mask_semantics":"imp"}
```

Пути внутри ZIP относительны к корню. Сервер сохраняет SHA-256 и нормализованный
manifest; source-local labels, splits, groups, revisions, license/provenance
сохраняются. Один снимок допускает либо mask, либо dem. DEM хранится как target;
оценка DEM/маршрутов пока не реализована.

Для ZIP только со снимками manifest необязателен: кладите их в images/;
сервер создаст IDs и split test. Для разметки необходим явный manifest, чтобы
маска не стала входным снимком. Без GT доступны ответы и время, качество unavailable.

### По ссылке

`POST /datasets/from-url`:

```json
{"url":"https://example.org/dataset.zip","name":"moon-demo","version":"v1","planet":"moon","task_id":"presence","provenance":{"license":"CC-BY-4.0"}}
```

Принимается прямая ссылка на ZIP, не страница GitHub/Hugging Face.
Импорт синхронный, ограничен размером/временем. Для уже скачанных больших коллекций
регистрируйте локальный manifest. Запрещены выход из папки, symlinks, executables,
частные/локальные URL и переходы к ним. Список: `GET /datasets`.

IBM IMP требует planet=moon, task_id=imp_segmentation, uint8 grayscale 256×256 и
preprocessing.input_protocol=curated_imp_png_reflectance_0_0.12_v1 в датасете
(для upload/from-url — внутри metadata), а также options.input_protocol модели.
Это исследовательский PNG track с приближённым восстановлением отражательной
способности; он не воспроизводит исходные TIFF. Mars и другие задачи эта head
не поддерживает. Сам encoder без готовой головы здесь не оценивается как QA.

## 2. Запуск одной модели

`POST /vlm/runs` возвращает текст; `POST /vit/runs` запускает IBM segmentation head.
Пример VLM:

```json
{
  "model_name": "SpaceLLaVA",
  "model_version": "v1",
  "dataset_id": 1,
  "image_count": 10,
  "split": "test",
  "seed": 42,
  "include_clean": true,
  "prompt": "Are irregular mare patches present? Answer only YES or NO.",
  "allowed_answers": ["YES", "NO"],
  "distortions": [["shear"], ["radiation"], ["lens_flare"], ["hard_shadows"], ["shear", "hard_shadows", "lens_flare", "radiation"]],
  "parameters": {
    "shear": {"max_shift_px": 4, "control_spacing_px": 32, "scan_axis": "both"},
    "radiation": {"num_hits": 80, "max_streak_length": 10},
    "lens_flare": {"strength": 0.55},
    "hard_shadows": {"strength": 0.65}
  }
}
```

10 снимков × (clean + 5 вариантов) = 60 ответов; include_clean:false даёт 50.
Порядок внутри каждой цепочки сохраняется. Seed эффекта зависит от общего seed,
source ID и названия эффекта: одинаковые поля/случайные параметры доступны для
отдельного эффекта и комбинации. Выборка одинакова для всех условий.
Значения параметров без явной настройки берутся из функций искажений.
HTTP 202 возвращает run_id, status/results/report URLs и прогресс.

| Ручка | Действие |
|---|---|
| GET /runs | Список, filters status/model_config_id, offset/limit |
| GET /runs/{id} | Статус, phase, completed/total/percent, ошибка, source IDs |
| POST /runs/{id}/cancel | Остановить очередь или после текущего шага |
| POST /runs/{id}/resume | Продолжить cancelled/interrupted/blocked/failed |
| GET /runs/{id}/results | Ответы, filters condition/status, offset/limit, file URLs |
| GET /runs/{id}/results/{result_id}/image | Подготовленный снимок |
| GET /runs/{id}/results/{result_id}/map | Карта классов IBM |

Статусы: queued → running → completed; также cancelled, interrupted, blocked, failed.
Phase показывает подготовку, загрузку, inference. Одновременно работает одна модель.
Остановка сохраняет текущий GPU-ответ, затем прекращает обработку. После падения
worker при новом старте running становится interrupted; требуется resume.
Продолжение отклоняется при изменении inputs/config/model provenance.
Ошибка загрузки весов/библиотек даёт blocked с объяснением; веса не скачиваются.
Ошибочные ответы сохраняются и не повторяются при resume: нужен новый запуск.

## 3. Метрики, отчёт, сравнение

`POST /runs/{id}/metrics` после completed рассчитывает и сохраняет:

- counts ok/error/unsupported, время успешных запросов без загрузки модели;
- при allowed_answers и answer в manifest — accuracy, coverage, confusion matrix,
  per-class precision/recall/F1 и для YES/NO FPR/FNR;
- IBM — pixel accuracy, IoU/Dice по классам, mean IoU и coverage карт;
- показатели каждого условия и изменение относительно clean.

Парсер принимает ответ целиком из заданного списка, нормализуя регистр/пробелы.
Из свободного описания YES/NO не извлекается. Invalid/error входят в основной
знаменатель accuracy. Без GT метрика unavailable. GT не передаётся модели.
Сегментационная маска получает то же shear-поле: nearest-neighbor, exterior=255
ignored. Photometric эффекты маску не меняют. IoU/Dice считаются по успешным
картам; сбои отражаются в coverage. DEM/mask/predicted depth не становятся
физической проходимостью без отдельного измеренного terrain/cost protocol.

`GET /runs/{id}/report` возвращает модель, датасет, прогресс, condition summaries
и chart data. E3 использует confusion matrices E1/E2. E4 остаётся unsupported;
E5 требует отдельных согласованных запусков. E1–E5 — исследования, а не пять
функций искажений. Готовая веб-страница в этот этап не входит.

Для второй модели повторите параметры с её model_name, правильной VLM/ViT ручкой
и reuse_run_id первого запуска. Dataset/count/split/seed, prompt/labels,
generation limit, chains/parameters должны совпадать. Вычислите метрики и вызовите
`POST /comparisons` с телом:

```json
{"run_ids":["first-run-id","second-run-id"]}
```

Ответ содержит два отчёта и condition contrasts. Для QA accuracy есть парный
bootstrap по scene groups (95%, 1000 повторов); один group не даёт интервала.
Варианты одного source не считаются независимыми снимками. Native segmentation
и text QA отображаются рядом с quality_comparable:false: accuracy и IoU имеют
разные определения. Научное ранжирование требует общего выходного протокола,
overlap audit и достаточной независимой выборки. Mock помечается и не является
научным результатом; сравнение mock/real запрещено.

## Настройки и прежние ручки

Settings(BaseSettings) читает .env в корне и переменные с префиксом PLANETARY_API_:

```dotenv
PLANETARY_API_DATABASE_PATH=.state/planetary.sqlite3
PLANETARY_API_SQL_ECHO=false
PLANETARY_API_CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
PLANETARY_API_MAX_UPLOAD_BYTES=268435456
PLANETARY_API_MAX_UNPACKED_BYTES=1073741824
PLANETARY_API_MAX_ARCHIVE_FILES=10000
PLANETARY_API_DOWNLOAD_TIMEOUT=30
PLANETARY_API_WORKER_POLL_SECONDS=2
```

Относительные пути привязаны к корню проекта. Foreign keys и WAL включены.
create_all добавляет runs к прежним четырём таблицам; изменение столбцов
потребует миграций. Приложение рассчитано на локальную работу без авторизации.

CRUD /model-configs, /dataset-configs, /model-results, /metric-analyses и импорт
сохранённых CLI-запусков /imports/results сохранены. CRUD-анализ — ручной импорт;
/runs/{id}/metrics — вычисление. JSON result_ids анализов проверяются API, а не
внешними ключами SQLite. Конфигурации уникальны по (name, version), ответы — по
(model_config_id, dataset_config_id, run_id, request_id).

Проверка: установить .[api,api-test], затем
`python -m unittest discover -s tests -v`. Тесты используют временные синтетические
данные и тестовые ответы/карты, без реальной загрузки GPU-моделей.
Документация: [FastAPI UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/),
[SQLModel](https://sqlmodel.tiangolo.com/tutorial/fastapi/session-with-dependency/),
[Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
