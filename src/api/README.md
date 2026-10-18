# API оценки моделей

## Новый локальный heads-трек: Windows / RTX3060Ti

Обучение выполняется **только через CLI**, API не запускает train.
`linear_head_approach publish` создаёт `import_in_ui.json` — его можно перетащить
в UI-блок моделей. `POST /head-deployments` проверяет server-local deployment и
атомарно регистрирует backend `frozen_crater` + WAC catalog validation/test.
Новый приватный `benchmark_services.heads.app` на8003 использует обучение-venv
`.venv-heads3060`; публичный API остаётся без Torch в `.venv-api`.
Общая GPU-блокировка не допускает одновременный CLI train и UI inference.

Первый трек: **clean lunar WAC catalog crater segmentation**, живой frozen
encoder + trained head, не Q1–Q4/IMP и не cached features вместо inference.
Worker получает RGB и paired native VIS, никогда masks/targets; CLIP использует
RGB, IBM native VIS5. Разные входные сведения указаны как ограничение.
Сервер блокирует train scenes, вопросы и искажения для этого трека. Карты512²
хранят raw class IDs0/1 с PNG palette; scoring — scene-macro Crater IoU/Dice.
`PLANETARY_API_SERVICE_MOCK=false` нужен для реальных моделей. Mock карты —
только software checks. Старые IMP/VLM пути ниже остаются отдельными.

Полные команды: [Windows/3060Ti runbook](../../linear_head_approach/gpu_runbook.md).

Один запуск — одна модель, один датасет, фиксированная выборка снимков и выбранные
цепочки искажений. Два изолированных сервиса получают ответы последовательно. API возвращает
прогресс, результаты и JSON отчёта с таблицами и данными для графиков будущего сайта.

## Запуск сервисов

### Отмена задания

`POST /runs/{run_id}/cancel` — отдельный рут отмены конкретного запуска:

- `queued` (включая подготовку и ожидание сервиса) → `cancelled/stopped`.
  Worker не загружает модель для снятого с очереди задания.
- `running` → `cancel_requested=true`, `phase=cancelling`; worker завершает
  текущую операцию (например, загрузку модели или inference), сохраняет ответ
  и останавливается перед следующей. Это не принудительное прерывание CUDA.
- Повторная отмена уже отменённого запуска возвращает его состояние (200).
  Неизвестный ID даёт 404; завершённый/заблокированный запуск — 409.

Отмена и получение задания worker используют атомарные SQL-переходы: запрос
отмены не теряется, если worker одновременно начинает выполнение. Сохранённые
ответы, файлы и сам запуск не удаляются. `POST /runs/{run_id}/resume` возвращает
остановленное задание в очередь, сохраняя проверки воспроизводимости.

### Логи worker

Общий модуль `src/api/logging.py` использует `structlog`: JSON-события запуска,
загрузки, обработки, отмены, ошибок и выхода процесса. Поля: UTC timestamp,
level, component, PID, run/request ID, статус, прогресс и время операции.
У ошибок пишется тип, без текста исключения, prompts, ответов, GT и токенов.

Логи дочернего worker дописываются в `service-worker.log` в `output_dir` запуска
(обычно `outputs/<model>/<run_id>/service-worker.log`) и одновременно выводятся
в консоль его model service. Worker пишет напрямую в файл; фоновое зеркало
дублирует только новые байты текущей попытки, без повтора старых логов при resume.
Логи API-диспетчера и
контроллера сервиса выводятся в их терминалы. Логи не являются оценками моделей.
В этот же файл попадают stdout/stderr сторонних библиотек в исходном виде:
гарантия JSON и фильтрация полей относятся к нашим structlog-событиям, а не ко
всему внешнему выводу. Не публикуйте полный лог без проверки его содержимого.
`structlog` входит во все три requirements; после обновления перезапустите
сервисы. Изменившийся pipeline может отказаться от resume старого запуска —
создайте новый, не меняя сохранённые метаданные вручную.

API больше не загружает модели и не запускает старый `api.worker`.
Работают три HTTP процесса: API на 8000, SpaceLLaVA на 8001 и IBM на 8002.
API-диспетчер выбирает конкретное задание; каждый сервис запускает его своим
Python в отдельном дочернем процессе. SQLite и рабочая директория общие,
модели выполняются последовательно под общей блокировкой GPU.

Полная инструкция для Windows, CPU mock-проверки, отдельных окружений и
подготовленного Docker Compose: [deploy/README.md](../../deploy/README.md).
API устанавливается через `requirements/api.txt`, модели отдельно через
`requirements/spacellava.txt` и `requirements/ibm-imp.txt`.
В каждом окружении установите текущий проект: `python -m pip install --no-deps -e .`.

В корневом `.env` задайте один `PLANETARY_API_SERVICE_TOKEN` для всех сервисов,
`PLANETARY_API_DISPATCHER_ENABLED=true` и для CPU проверки
`PLANETARY_API_SERVICE_MOCK=true`. Веса/зависимости GPU не загружаются автоматически.
Отсутствующий токен блокирует приватные endpoints сервиса; wrong backend/hash
не запускают процесс. `GET /services` показывает доступность сервисов и режим mock.

```powershell
# Каждый процесс в отдельном терминале; для CPU mock можно один лёгкий venv.
& .\.venv-api\Scripts\python.exe -m uvicorn --app-dir src api.main:app --host 127.0.0.1 --port 8000
& .\.venv-api\Scripts\python.exe -m uvicorn --app-dir src benchmark_services.spacellava.app:app --host 127.0.0.1 --port 8001
& .\.venv-api\Scripts\python.exe -m uvicorn --app-dir src benchmark_services.ibm.app:app --host 127.0.0.1 --port 8002
```

Для реальных моделей замените Python в двух последних командах на интерпретаторы
их окружений; API остаётся в `.venv-api`. Model services не запускайте с
`--reload` или несколькими uvicorn workers. Проверка worker entrypoint без очереди:
`python src/benchmark_services/spacellava/worker.py --help` и аналогично `ibm/worker.py`.
GPU/16 GB совместимость всё ещё требует отдельного smoke test.
Synthetic `service_mock` сохраняется в SQLite, run.json, результатах и отчётах:
это не научные результаты даже при backend `spacellava`/`ibm_imp`.

## Таблицы и сохранение

| Таблица | Содержимое |
|---|---|
| model_configs | Название/version, backend, локальные weights/revisions/options и hash |
| dataset_configs | Manifest, planet/task, splits, preprocessing, license/provenance |
| runs | Очередь одной модели, выборка, протокол, прогресс и остановка |
| model_results | Raw response, статус, время, source/group/split, карты и hashes |
| metric_analyses | Метрики, версия метода, IDs/hash результатов, признак mock |

SQLite: `.state/planetary.sqlite3`. ZIP-датасеты: `data_orbital/uploads/<id>/`.
Подготовленные входы: `data_orbital/inference/api/<protocol_sha256>/images/`.
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

Зависимости сетевого восстановления уже входят в `requirements/api.txt`.
Каталог и подключение уже существующих файлов не требуют pyarrow/h5py.

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
Диспетчер после освобождения GPU lock переводит оставшиеся running в interrupted; требуется resume.
Продолжение отклоняется при изменении inputs/config/model provenance.
Ошибка загрузки весов/библиотек даёт blocked с объяснением; веса не скачиваются.
Ошибочные ответы сохраняются и не повторяются при resume: нужен новый запуск.

## 3. Метрики, отчёт, сравнение

`POST /runs/{id}/metrics` после completed рассчитывает и сохраняет:

- counts ok/error/unsupported, время успешных запросов без загрузки модели;
- для канонических текстовых ответов при allowed_answers и допущенном answer в manifest — accuracy, coverage, confusion matrix,
  per-class precision/recall/F1 и для YES/NO FPR/FNR;
- для native IMP masks — pixel accuracy, IoU/Dice по классам, mean IoU и coverage карт;
- показатели каждого условия и изменение относительно clean.

Парсер принимает ответ целиком из заданного списка, нормализуя регистр/пробелы.
Из свободного описания YES/NO не извлекается. Invalid/error входят в основной
знаменатель accuracy. Без GT метрика unavailable. Кандидатные/неизвестные эталоны
дают `unscored_reference`, а не отрицательную метку или ошибку модели.
Для crater Q1–Q4 API требует явного `reference_status: REFERENCE_READY`,
`review_status: accepted` или `scoring_eligible: true`; противоречащие flags отклоняются.
Наличие `answer` само по себе не допускает текущие crater candidates к scoring.
Поле `scored:false` означает «ещё не оценено», не заменяет статус допуска.
GT не передаётся модели.
Сегментационная маска получает то же shear-поле: nearest-neighbor, exterior=255
ignored. Photometric эффекты маску не меняют. IoU/Dice считаются по успешным
картам; сбои отражаются в coverage. DEM/mask/predicted depth не становятся
физической проходимостью без отдельного измеренного terrain/cost protocol.

После shear исходные ответы crater Q1–Q4 автоматически не переиспользуются:
нужны явные `condition_answers` для соответствующего condition ID. Сравнение с
clean не считает delta при различающемся наборе допущенных сцен.
При частичном GT/покрытии карт возвращается `quality_status: partial`.

`GET /runs/{id}/report` возвращает модель, датасет, прогресс, condition summaries
и chart data. E1/E2 отражают доступность GT и выбранных условий; без искажений
E2 — `not_selected`. E3 даёт только false-positive proxy вопроса presence с
явным знаменателем; неподтверждённые утверждения в свободном тексте не оцениваются.
E4 остаётся unsupported;
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

Метрики выбираются по `output_type`/задаче, а не по `run.kind`: канонические
Q1–Q4-ответы от VLM и объявленного encoder readout используют одно ядро оценки.
Это не добавляет crater head в HTTP worker и не импортирует Colab ZIP в SQLite
автоматически. Native masks и текст по-прежнему не дают общего ranking.
Сравнение native IoU требует полного совпадающего покрытия карт и GT.
Версия метода — `orbital-api-metrics-v2`; старые результаты не перезаписываются,
для них следует заново вызвать `/metrics`. Перед расчётом/сравнением проверяются
замороженные protocol/selection, prepared requests, image hashes и происхождение
каждого ответа. Детали CLI и формата эталонов: `docs/protocols/evaluation_v2.md`.

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
