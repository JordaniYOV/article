# Windows 10 · RTX 3060 Ti 8 GiB · 16 GiB RAM

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

## 5. FastAPI + worker + UI

API остаётся отдельным лёгким окружением `.venv-api` из `requirements/api.txt`.
Heads-worker использует **то же окружение**, что и обучение. Старые full-VLM/IMP
сервисы для этого трека запускать не нужно. Если `.venv-api` ещё нет:

```powershell
uv venv --seed --python 3.12.14 .venv-api
& .\.venv-api\Scripts\python.exe -m pip install --no-cache-dir -r requirements/api.txt
& .\.venv-api\Scripts\python.exe -m pip install --no-deps -e .
```

В `.env` одинаковый непустой `PLANETARY_API_SERVICE_TOKEN` для API и worker;
`PLANETARY_API_DISPATCHER_ENABLED=true`, heads URL `http://127.0.0.1:8003`.
**Для реального benchmark**: `PLANETARY_API_SERVICE_MOCK=false`. Иначе будут
явно помеченные synthetic fixture masks, не результаты обученной модели.

В отдельных терминалах, из корня проекта:

```powershell
& .\.venv-api\Scripts\python.exe -m uvicorn --app-dir src api.main:app --host 127.0.0.1 --port 8000
& .\.venv-heads3060\Scripts\python.exe -m uvicorn --app-dir src benchmark_services.heads.app:app --host 127.0.0.1 --port 8003
```

Третий терминал: `npm --prefix frontend run dev`.

В UI перетащи `import_in_ui.json` в блок подключения **моделей**. Сервер
проверит files/hashes и атомарно зарегистрирует модель + общую WAC-выборку.
Потом выбрать модель, **подключённую WAC_catalog_craters_heads**, сначала `val`.
Начать с 1–2 изображений. После фиксации протокола вручную выбрать `test`.
Повторить ту же выборку на второй голове; сравнивать все3seeds, не лучший.

Worker запускает **живой encoder + trained head**, не подставляет GT и не
использует feature cache как предсказание. Измеренное время не включает загрузку
модели, но включает encoder/head/map save. Raw masks/logits сохраняются на диск.
API считает scene-macro **Crater IoU / Crater Dice** и показывает карты.
Это не IMP mean IoU, не Q1–Q4 и не route/hazard benchmark.

### Ограничения первого UI-трека

- Только **clean lunar WAC** held-out scenes. Изображения train запрещены.
- Искажения/Q1–Q4 заблокированы и UI, и сервером. Нельзя изменять только RGB,
  оставляя native VIS IBM без искажения: это несопоставимые условия.
- IBM native VIS5 vs CLIP RGB — не equal-information. Передача обеих модальностей
  в request документирована, но CLIP использует только RGB. Нет доказательства
  независимости от pretraining или полноты каталожной crater-разметки.
- Новые Windows encoder/head GPU jobs ещё не проверены. CPU/mock tests — только
  software checks; не основание обещать вместимость IBM FP32 в8GiB.

Источник совместимости ОС: [NVIDIA CUDA13 Windows guide](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-installation-guide-microsoft-windows/index.html#system-requirements).
