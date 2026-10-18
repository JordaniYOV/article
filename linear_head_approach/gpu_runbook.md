# Windows 10/11 · RTX 3060 Ti 8 GiB · 16 GiB RAM

Текущий способ: **обучение через CLI**, **benchmark через UI + FastAPI**.
Новый Windows GPU smoke ещё НЕ выполнен. Не использовать Colab lock-файл для
нативной Windows: в нём Linux-only Triton/NCCL/NVIDIA packages.

## 1. Окружение обучения и heads-worker

Все команды ниже из корня проекта. Нужны Git, uv и совместимый NVIDIA driver.
Для выбранного CUDA13-профиля NVIDIA указывает Windows10 **22H2**. Полный CUDA
Toolkit/Visual Studio не требуется для использования готовых PyTorch wheels;
если pip пытается собирать зависимости, остановись и сохрани log.

```powershell
py -m pip install --user uv
# Если uv не появился в PATH, installer попробует py -m uv / python -m uv.
powershell -ExecutionPolicy Bypass -File linear_head_approach/setup_windows.ps1 -CheckUv
powershell -ExecutionPolicy Bypass -File linear_head_approach/setup_windows.ps1 -Install
$headPython = '.\.venv-heads3060\Scripts\python.exe'
& $headPython -B -m linear_head_approach doctor
& $headPython -B -m unittest discover -s linear_head_approach/tests -v
```

Installer создаёт только `.venv-heads3060`, устанавливает requirements и проект;
не скачивает модели и не начинает обучение. Профиль Python3.12.14 x64.
Фактически установленные версии сохраняются в
`outputs/environments/heads3060_frozen.txt`; повторная установка использует этот
lock, не выбирает новые transitive dependencies. Сохраняй lock вместе с runs.
Runtime-код IBM, если его ещё нет (не веса/данные):

```powershell
& $headPython -B -m linear_head_approach runtime --allow-download
```

Существующий IBM runtime другой версии/с правками не сбрасывается автоматически.

## 2. Данные, место и явная загрузка весов

Используются уже выбранные `data_orbital/head_training_v2/`: 50 train / 39 val /
31 test. Не скачивается полный датасет. Маски, RGB и native TIFF не меняются.
Colab ZIP остаётся прежним зафиксированным артефактом; не распаковывай его код
поверх нового Windows checkout. Если переносишь данные на другой компьютер,
сохрани именно этот подкаталог с manifests/provenance и проверь `audit`.

```powershell
& $headPython -B -m linear_head_approach audit
# Запрашивают сеть/веса только эти явные команды:
& $headPython -B -m linear_head_approach assets --encoder ibm --assets weights/encoder_heads --allow-download
& $headPython -B -m linear_head_approach assets --encoder spacellava --assets weights/encoder_heads --allow-download
```

Скачиваются original IBM backbone и только один SpaceLLaVA shard с CLIP-тензорами
(не полная LLM). Предварительно проверяются реальные remote sizes и свободное
место на диске назначения: download bytes +2GiB reserve. Это **не гарантия**,
что 18GiB хватит на чистую установку всего окружения. `pip --no-cache-dir`
уменьшает дублирование; существующие системные pip/HF caches не удаляем.
Можно указать `--assets D:/planetary_weights` и передать тот же путь дальше.

## 3. Обучение через CLI

```powershell
& $headPython -u -B -m linear_head_approach train --encoder ibm --resume
& $headPython -u -B -m linear_head_approach train --encoder spacellava --resume
```

Каждая команда: audit → verify local assets → **2-image encoder smoke** → все
признаки → synthetic head smoke → linear/spatial ×3seeds → validation.
Веса автоматически не скачиваются. Test-метрики не вычисляются и не участвуют
в выборе головы/threshold. `audit` проверяет целостность изображений и масок
всех splits, включая test; это не оценка предсказаний на test.
Отдельный дочерний процесс на каждую стадию освобождает VRAM после неё.
Дефолтные paths: `outputs/heads_rtx3060ti_v1/{features,runs,logs}`.
`--experiment outputs/my_new_run` меняет корень эксперимента; для продолжения
передавай тот же путь. GPU batch encoder=1, головы=4, FP32. No AMP/quantization.
TF32 выключен в extraction и inference. Bilinear head resize реализован через
фиксированные separable matrices для deterministic CUDA backward.
CLI и UI используют одну `.state/api-worker.lock`: параллельный GPU job закрыт.

При нехватке VRAM остановиться и сохранить log, а не молча менять размер входа
только одной модели. Параметры/seed/source splits прежние, смена GPU не создаёт
новый научный метод. Старые Colab cache/run с другими fingerprints не resume:
новый Windows эксперимент начинается в отдельном каталоге.

Полезные независимые стадии доступны через `--help`; `fit --resume` хранит
optimizer/scheduler/RNG после полной эпохи. Прерванная эпоха повторяется.
Изменение кода/версий/данных блокирует reuse. Логи: консоль + `events.log` + logs/.

### Если smoke упал с Missing/Unexpected keys для CLIP

Веса SpaceLLaVA используют `vision_model.*`; Transformers 5.15.1 ожидает эти
параметры непосредственно как `embeddings.*`, `encoder.*` и т.д. Текущий loader
сопоставляет весь набор из391 tensor names с фактическим model.state_dict и
применяет только точное взаимно-однозначное переименование. `strict=True`
сохранён; недостающие/лишние параметры и несовместимые shapes не игнорируются.

Перенеси обновлённый `linear_head_approach/encoders.py` на GPU-компьютер. Веса и
окружение не переустанавливать. После изменения кода используй новый каталог:
старый `identity.json` создаётся ещё перед загрузкой, даже если smoke упал.
Не удаляй/редактируй identity для обхода проверки и не смешивай старые и новые runs.

```powershell
& $headPython -u -B -m linear_head_approach train --encoder spacellava --experiment outputs/heads_rtx3060ti_clipfix_v1 --resume
```

Эта команда действительно запустит GPU smoke и продолжит обучение только после
его успеха. Если нужна лишь проверка загрузки, можно отдельно запустить
`extract --smoke` с новым `--output` и `--session-id`; он извлекает только два
изображения, без fit/test. Предварительно успешно обученные головы/кэши другой
версии сохранить отдельно; экспорт требует тот же code/environment, что fit.
В следующих командах publish/evaluate/preview указывай новый experiment path,
а не прежний `outputs/heads_rtx3060ti_v1` из примеров ниже.

### Если publish старой IBM-головы сообщил Training identity changed

Первоначальный code hash охватывает все Python-модули, поэтому изменение CLIP
блокировало даже готовую IBM-голову. В текущем exporter/worker есть узкая audited
compatibility policy: IBM до CLIP-fix и обе модели после fix. Проверяются полные
исторические manifests исходников (включая варианты Git LF/CRLF) и точный текущий
runtime. Неизвестные правки и изменения Python/packages по-прежнему запрещены.

Это только совместимость **экспорта завершённой головы и frozen inference**,
не разрешение продолжать старое обучение под новым кодом. `identity.json`,
`best.pt`, features и threshold не переписываются. В deployment отдельно остаются
исходная training identity и проверяемая code_compatibility attestation.

Перенеси обновлённую папку `linear_head_approach/`,
`src/benchmark_services/heads/adapter.py` и
`src/benchmark_services/common/execution.py` на GPU-компьютер. Затем повтори прежний
publish для IBM в новом output-каталоге. Скачивать веса и переобучать голову из-за
этой конкретной CLIP-правки не требуется. Если сообщение теперь говорит именно о
Python/dependency versions, восстанови frozen training environment, а не меняй
сохранённую identity вручную.

## 4. Экспорт обученных голов для UI

После успешного fit (пример одной spatial head; повторить для всех seeds/families):

```powershell
& $headPython -B -m linear_head_approach publish --run outputs/heads_rtx3060ti_v1/runs/ibm/spatial/seed_13 --assets weights/encoder_heads --output outputs/head_deployments/ibm_spatial_13 --name IBM-spatial-13
& $headPython -B -m linear_head_approach publish --run outputs/heads_rtx3060ti_v1/runs/spacellava/spatial/seed_13 --assets weights/encoder_heads --output outputs/head_deployments/clip_spatial_13 --name SpaceLLaVA-CLIP-spatial-13
```

В каждой deployment-папке: immutable `head.pt`, input-only held-out index,
`deployment.json` и **`import_in_ui.json`**. Export не обучает и не оценивает test.
Threshold берётся из best.pt (выбран на val), не подбирается в API.
Только свои checkpoints; не загружать .pt от неизвестного источника.

## 5. Окружение API — без uv в PATH

Если уже опубликованы шесть spatial-голов, шаги1–4 повторять не нужно. Ниже
нет повторного обучения, скачивания весов или обновления `.venv-heads3060`.
Все команды выполнять из корня проекта `article`, где находятся `src` и
`linear_head_approach`. Переменные PowerShell не переносятся в новый терминал.

API использует отдельное окружение `.venv-api` без Torch из `requirements/api.txt`.
Создаём его через **уже установленный** Python окружения обучения; системный
Python3.15 и доступность команды `uv` не имеют значения:

```powershell
$headPython = '.\.venv-heads3060\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $headPython)) { throw 'Training Python is missing. Check the project directory.' }
if (-not (Test-Path -LiteralPath '.\.venv-api\Scripts\python.exe')) {
    if (Test-Path -LiteralPath '.\.venv-api') { throw 'An incomplete .venv-api already exists; do not overwrite it blindly.' }
    & $headPython -m venv .venv-api
    if ($LASTEXITCODE -ne 0) { throw 'API environment creation failed.' }
}
$apiPython = '.\.venv-api\Scripts\python.exe'
& $apiPython -m pip install --no-cache-dir -r requirements/api.txt
if ($LASTEXITCODE -ne 0) { throw 'API dependency installation failed.' }
& $apiPython -m pip install --no-cache-dir --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Project installation in the API environment failed.' }
& $apiPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'API dependency check failed.' }
```

Если готовая `.venv-api` уже установлена, повторно ставить пакеты не обязательно:
можно перейти к шагу6. **Не устанавливай API requirements в `.venv-heads3060`**:
worker должен сохранить точные Python/packages, на которых обучались головы.

## 6. Общий .env для API и heads-worker

Открой существующий `.env` в корне проекта; не перезаписывай его целиком шаблоном.
Нужны следующие значения:

```dotenv
PLANETARY_API_DISPATCHER_ENABLED=true
PLANETARY_API_HEADS_SERVICE_URL=http://127.0.0.1:8003
PLANETARY_API_SERVICE_MOCK=false
```

В этом же файле `PLANETARY_API_SERVICE_TOKEN` должен быть непустым одинаковым
значением для API и worker (минимум16 ASCII-символов). Существующий токен оставь.
Если он пустой, сгенерируй новый и вручную вставь вывод после
`PLANETARY_API_SERVICE_TOKEN=`; не отправляй значение в чат/публичный log:

```powershell
& .\.venv-api\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))"
```

Для настоящих моделей `PLANETARY_API_SERVICE_MOCK=false` обязателен. Значение
`true` создаёт synthetic fixture masks, не научные результаты. Переменные
окружения PowerShell имеют приоритет над `.env`: если раньше задавал `$env:...`,
проверь, что они не переопределяют эти настройки. После изменения настроек
перезапусти API и worker.

## 7. Три терминала: API, heads-worker, интерфейс

Каждый терминал открой в корне проекта. В командах полные пути к нужным Python,
поэтому активация venv, `uv` и переменная `$headPython` здесь не требуются.
Сначала дождись завершения CLI train: GPU общий с benchmark.

Терминал1 — API, порт8000:

```powershell
& .\.venv-api\Scripts\python.exe -m uvicorn --app-dir src api.main:app --host 127.0.0.1 --port 8000
```

Терминал2 — единственный heads-worker для **обоих энкодеров**, порт8003:

```powershell
& .\.venv-heads3060\Scripts\python.exe -m uvicorn --app-dir src benchmark_services.heads.app:app --host 127.0.0.1 --port 8003
```

Новые пакеты в этот venv не ставить. Не добавлять `--reload` или несколько
uvicorn workers. Старые SpaceLLaVA/IBM-IMP сервисы на8001/8002 для этих шести
голов запускать не нужно. Модели worker выполняет последовательно.

Терминал3 — интерфейс. Нужен установленный Node.js/npm. Если frontend-зависимости
ещё не установлены, один раз выполнить (не касается Python/весов):

```powershell
npm.cmd --prefix frontend ci
```

Затем запустить интерфейс. `npm.cmd` не требует разрешения запуска `npm.ps1`:

```powershell
npm.cmd --prefix frontend run dev -- --host localhost --port 5173 --strictPort
```

Открыть `http://localhost:5173`. При занятом порте не менять его молча: CORS
по умолчанию разрешает именно этот origin. Документация API:
`http://127.0.0.1:8000/docs`. Серверы останавливаются через Ctrl+C в своих терминалах.

## 8. Проверка серверов без GPU benchmark

В четвёртом PowerShell-терминале выполнить только HTTP-проверки:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/services' | ConvertTo-Json -Depth 6
```

API должен отвечать; сервис `heads` должен быть доступен и показывать `mock=false`.
Недоступные `spacellava` и `ibm` здесь нормальны: это другие, не используемые
сейчас сервисы. Health-проверка не подтверждает загрузку энкодера, VRAM-fit или
качество предсказаний. GPU проверяется следующим маленьким UI-прогоном.

## 9. Импорт шести опубликованных spatial-голов

В UI-блок подключения **моделей** по одному перетащи `import_in_ui.json` из
каждой из шести папок `outputs/head_deployments/`:

- `clip_spatial_13`, `clip_spatial_37`, `clip_spatial_73`;
- `ibm_spatial_13`, `ibm_spatial_37`, `ibm_spatial_73`.

Используй только успешные актуальные deployments. Если после hotfix имя папки
получилось другим (например, `_compat_v1`), бери JSON оттуда; старый и новый
экспорт одной головы — не два независимых обучения. Не импортируй `head.pt` или
`deployment.json` вместо `import_in_ui.json`, не загружай папки как ZIP-датасет.

Сервер проверит files/hashes и зарегистрирует модель + общий датасет
**WAC_catalog_craters_heads**. Датасет создаётся один раз, не шесть копий.
Пути deployment/weights должны относиться к компьютеру, где работает API/worker.

## 10. Сначала маленький validation-прогон, затем benchmark

В форме запуска выбрать одну импортированную модель и в **Подключённых выборках**
датасет **WAC_catalog_craters_heads** — не встроенную «Луна / IMP / высоты».

Для первого проверочного запуска:

- Выборка: `val`; снимков: `2` (или1).
- Только clean; никаких вопросов Q1–Q4 и искажений для этого режима.
- Seed в форме запуска: например42, одинаковый для всех моделей. Это seed
  выбора снимков, не seed обучения13/37/73 в имени головы.

Дождись завершения. В панели **«Результаты и метрики»** выбери этот запуск и
нажми **«Рассчитать метрики»**; карты можно открыть во вкладке **«Примеры ответов»**.
Проверь, что результат не mock, карты появились и рассчитаны Crater IoU/Dice.
Если загрузка упала — сохранить error/log, не обновлять пакеты
обучающего venv вслепую. Повтори проверку на второй архитектуре. Для остальных
голов можно выбрать первый запуск в **«Повторить выборку»**, чтобы сцены совпали.

После успешной технической проверки и фиксации протокола выполнить все шесть
моделей на одинаковой выборке. Полный val —39 изображений (он уже использован
для выбора эпохи/threshold, не независимая финальная оценка). Финальный
source-test — **31 изображение**, split `test`; не50 и не120. Выбор test явный,
не использовать его результаты для подбора головы, порога или seed.

Во всех шести test-прогонах сохранить одинаковые split/count/selection seed,
clean-условие и тот же датасет. Используй «Повторить выборку» от первого
test-прогона, проверь параметры перед запуском. GPU jobs выполняются по одному;
одна модель на запуск. При отмене прогресса не удалять папки/метаданные: resume
допустим только при совпадении сохранённых кода, данных и настроек.

## 11. Результаты и сравнение

Worker запускает **живой encoder + trained head**, не подставляет GT и не
использует feature cache как предсказание. Измеренное время не включает загрузку
модели, но включает encoder/head/map save. Raw masks/logits сохраняются на диск.
API считает scene-macro **Crater IoU / Crater Dice** и показывает карты.
Это не IMP mean IoU, не Q1–Q4 и не route/hazard benchmark.

Сохрани отдельные результаты для13/37/73, затем своди среднее по трём seed
отдельно для CLIP и IBM, не выбирай лучший. UI умеет парные сравнения запусков;
автоматическая сводка трёх seed и её доверительный интервал этим не обещаются.
Logs worker: `service-worker.log` в output-папке каждого запуска и консоль
heads-service. Не выдавать mock-тесты или validation-оценку за финальный test.

### Ограничения первого UI-трека

Обновление: привязка к прежнему WAC allowlist остаётся только у старого протокола.
Для новой `crater_seg_100_v1` в UI-блоке датасетов нажми
**«Подключить crater_seg_100_v1»**, затем выбери `Crater_seg_100_v1_moon` или
`Crater_seg_100_v1_mars` в подключённых выборках. По 50 source-test снимков:
split `test`, сначала 1–2 для технического smoke. CLIP поддерживает RGB обеих
планет; текущая IBM-голова требует реальные VIS5 и доступна только на лунной
части. Маски оцениваются через IoU/Dice без вопросов. Новых обучений/exports
не нужно; перезапустить API/heads/frontend и создать новый run, не resume старого.
Подробности и сохранённые проверки — в `src/api/README.md`.

Ниже ограничения именно прежнего WAC-протокола:

- Только **clean lunar WAC** held-out scenes. Изображения train запрещены.
- Искажения/Q1–Q4 заблокированы и UI, и сервером. Нельзя изменять только RGB,
  оставляя native VIS IBM без искажения: это несопоставимые условия.
- IBM native VIS5 vs CLIP RGB — не equal-information. Передача обеих модальностей
  в request документирована, но CLIP использует только RGB. Нет доказательства
  независимости от pretraining или полноты каталожной crater-разметки.
- Новые Windows encoder/head GPU jobs ещё не проверены. CPU/mock tests — только
  software checks; не основание обещать вместимость IBM FP32 в8GiB.

Источник совместимости ОС: [NVIDIA CUDA13 Windows guide](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-installation-guide-microsoft-windows/index.html#system-requirements).
