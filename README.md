# Evaluating Vision Models on Planetary Surfaces

## Активный этап — Mars binary adaptation (19 октября 2026)

Новый основной эксперимент: способы адаптации NASA–IBM Lunar-FM к бинарной
сегментации кратеров на Mars THEMIS. Отдельные папки и CLI-пайплайны A linear probe,
B frozen encoder + decoder, C LoRA + тот же decoder и D frozen SAM baseline:
[`src/planetary_vlm/adaptation/README.md`](src/planetary_vlm/adaptation/README.md).
Данные: `mars_crater_binary_v1`, 3600 train / 900 val / 900 test. Новые методы
ещё не проходили реальный pretrained GPU smoke; обучение не запускалось.
Старые `linear_head_approach` и UI deployments сохранены, не переподключены к Марсу.
Новый CLI: `python -m planetary_vlm.adaptation linear_probe plan` (без моделей).
Остальные разделы ниже описывают сохранённые предыдущие направления проекта.

Рабочий проект для сравнения vision-language models (VLM) и визуальных энкодеров
на орбитальных снимках поверхности Луны и Марса, включая оценку устойчивости к
искажениям. Научный замысел — [`research_plan.md`](research_plan.md).

Название охватывает оба протокола: сравнение полных систем VLM и encoder с
готовой task head, а также сравнение замороженных визуальных энкодеров с отдельно
обучаемыми головами одного типа. Во втором протоколе языковая часть VLM не
участвует; выводы относятся к визуальным признакам.

**Текущий этап (30 сентября 2026):** основной замысел перенесён на реальные
орбитальные снимки Марса и Луны. Новая коллекция, маски, provenance и
ограничения описаны в [`data_orbital/README.md`](data_orbital/README.md).
`data_rover/` исключён из текущего исследования и защищён: запрещено читать,
использовать, копировать, перемещать, переименовывать, менять или удалять его
содержимое и сам каталог. Работа с данными ведётся только в `data_orbital/`.
В отдельном supporting-протоколе сравниваем **замороженные визуальные энкодеры**
[`remyxai/SpaceLLaVA`](https://huggingface.co/remyxai/SpaceLLaVA) и
[`NASA–IBM Lunar Foundation Model`](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model)
с отдельно обучаемыми пространственными головами одного типа. Первый пилот
использует лунные WAC-маски кратеров; его дизайн и ограничения описаны в
[`linear_head_approach/protocol.md`](linear_head_approach/protocol.md). Сравнение
полных систем и вопросы E1–E5 остаются отдельным анализом в
[`docs/protocols/orbital_study.md`](docs/protocols/orbital_study.md).
Текущий локальный target — **Windows10 / RTX3060Ti 8GiB / RAM16GiB**.
Обучение голов через CLI, benchmark через UI/FastAPI и heads-service на8003.
Установка, smoke, resume и экспорт модели описаны в
[`linear_head_approach/gpu_runbook.md`](linear_head_approach/gpu_runbook.md).
Готовые Colab notebooks и ZIP-комплекты удалены при очистке 19 октября 2026.
Обучение выполняется через CLI, оценка — через API/UI. Код и исходные identities
обученных голов сохранены без изменения. Текущие данные и сохранённое происхождение
описаны в [`data_orbital/README.md`](data_orbital/README.md).
Веса не загружены, GPU-запуск не проверен. Разделы ниже описывают также прежний
rover-каркас.

## Что работает

- Общий контракт `ModelRequest → ModelResponse` и registry адаптеров.
- Строгие JSONL manifests: вопросы и входы отдельно от targets.
- Runner: один backend за запуск, сырые ответы/сбои, fingerprints и resume.
- Evaluator: accuracy, macro-F1, confusion matrix, FPR/FNR, invalid/failure/coverage,
  срезы task×condition; парный cluster-bootstrap accuracy.
- Image processor: clean, blur, noise, low light, contrast, occlusion.
- NumPy/OpenCV-функции shear, radiation, lens flare, жёстких теней и их
  комбинации: [`src/planetary_vlm/distortion/README.md`](src/planetary_vlm/distortion/README.md).
- Фиксированный rendering готовых depth NPY и композиция выровненных изображений.
- Конвертер native semantic masks в вопросы terrain/rock coverage с provenance.

Модули моделей и evaluator независимы: новый адаптер подключается через registry,
а его ответы оцениваются тем же кодом. Оценщик не загружает модель; processor не
читает targets. Подробный формат — `docs/protocols/manifest_v1.md`.

## Начальные модели исторического rover-протокола

SelenoVLM использует собственные MAE/Perceiver/cross-attention и текстовую
Qwen2.5-7B-Instruct. Текстовая Qwen имеет роль контроля без изображения;
готовый general VLM до lunar-обучения Seleno пока не найден.
Qwen2.5-VL-7B-Instruct предусмотрен как отдельная general VLM для rover images.
Это сравнение систем; эффект fine-tuning нельзя изолировать этой парой.

Native Seleno входы и ограничения описаны в `docs/protocols/model_adapters.md`
и `docs/research/seleno_baseline.md`. Совместимость rover RGB и размещение Seleno
в 16 GB требуют отдельной проверки. GPU inference в этой реализации ещё не
проверен. Веса не загружены. Текущая орбитальная коллекция содержит 300
уникальных снимков Луны и Марса; это не прежняя задача сбора 300 лунных
rover-view кадров.

## API и изолированные сервисы моделей

`src/api/` обслуживает сайт и очередь; `src/benchmark_services/common/`
содержит общий pipeline, хранение и подготовку входов; `spacellava/` и `ibm/`
содержат отдельные HTTP сервисы, workers и адаптеры. API не импортирует модели.
Каждый сервис использует свой Python; общий GPU lock допускает одно задание.
Одинаковые image-only варианты готовятся один раз на стороне API и проверяются
по SHA-256 обоими worker. Старый `src/api/worker.py` удалён.

Команды трёх процессов, CPU mock-проверки без RTX и подготовленный Docker Compose:
[deploy/README.md](deploy/README.md). Docker/GPU запуск не проверен; CPU fixtures
не являются научными результатами. Исторические датасеты и результаты сохранены.

## Установка

В `requirements/` три самостоятельных файла: `api.txt` для API, orbital I/O
и CPU/mock-тестов; `spacellava.txt` для SpaceLLaVA; `ibm-imp.txt` для NASA–IBM.
Корневого `requirements.txt` и вложенных профилей больше нет. Модели используют
разные окружения: версии библиотек конфликтуют. Команды и ограничения описаны в
[deploy/README.md](deploy/README.md). GPU-профили пока не прошли smoke test;
requirements не устанавливает Windows C++ runtime, драйвер или веса.

Рабочее окружение — `.venv-benchmark` (Python 3.12), проект установлен editable.
Старое неработающее `.venv` удалено при очистке 15 октября 2026.

```powershell
$benchmarkPython = ".\.venv-benchmark\Scripts\python.exe"
& $benchmarkPython -m planetary_vlm --help
```

На другой машине: Python 3.11+, `python -m venv .venv-benchmark`, затем
`.\.venv-benchmark\Scripts\python.exe -m pip install -e ".[images]"`.
HF adapters используют extra `hf`; CUDA PyTorch выбирается для конкретной
среды перед GPU-пилотом. Для Seleno требуется локальная копия upstream с его
зависимостями и точными checkpoints, согласно upstream lockfile.

## Прогон без GPU

```powershell
& $benchmarkPython -m planetary_vlm validate --requests examples/toy/requests.jsonl --targets examples/toy/targets.jsonl --check-assets
& $benchmarkPython -m planetary_vlm run --config examples/toy/run.toml
& $benchmarkPython -m planetary_vlm evaluate --requests outputs/runs/toy-demo/requests.jsonl --targets examples/toy/targets.jsonl --predictions outputs/runs/toy-demo/predictions.jsonl --output outputs/reports/toy-demo.json
```

Это искусственная software fixture: картинка 4×4, произвольные targets и заданные
ответы mock. Отчёт помечен `mock=true`; его цифры не являются научными результатами.
Повторный `run` требует нового run_id либо `--resume`. Выходные файлы не затираются.

```powershell
& $benchmarkPython -m planetary_vlm prepare --requests examples/toy/requests.jsonl --config examples/toy/transforms.toml --output outputs/prepared/toy-demo
& $benchmarkPython -m planetary_vlm expand-targets --targets examples/toy/targets.jsonl --mapping outputs/prepared/toy-demo/variants.jsonl --output outputs/prepared/toy-demo/targets.jsonl
```

Параметры toy transforms демонстрационные; научные severity ещё не зафиксированы.
Для следующего прогона TOML run указывает на подготовленный requests manifest.

## Структура

Подготовка датасета отделена от преобразований изображений и запуска моделей:

```powershell
& $benchmarkPython -m planetary_vlm dataset prepare --planet mars --output data_rover/processed/dataset_v2
```

Это локальная сборка из уже скачанного Mars-Bench, без сети и опроса VLM.
`dataset_v1` не изменяется. Для повторной сборки выбирайте новую папку.
Старый `prepare --requests ... --config ...` по-прежнему создаёт деградации.

Два входных скрипта можно вызывать отдельно или через CLI:

```powershell
# Сеть запускается только по этой явной команде:
& $benchmarkPython -m planetary_vlm dataset download --planet mars --mars-archive
# Проверка одной уже скачанной стереопары; новая папка результата:
& $benchmarkPython -m planetary_vlm dataset check-pair --audit data_rover/source_audit_v1/mastcam_stereo_pilot/audit.json --sol 0703 --refine-pointing --output data_rover/interim/sol703_new_version
```

Полный список команд и границы лунного адаптера: `docs/protocols/dataset_lifecycle.md`.

Для выборки, где у каждого кадра есть восстановленная стереоглубина, используйте
`dataset depth-batch`: стадии `plan`, `acquire`, `run`. Проверка возобновляется
через `--resume`; `--build-output` собирает финальную папку только после полного
прохода. Допуск требует совмещения с размеченным кадром и >=50% валидной глубины.
Это технически проверенная реконструкция, не независимо измеренный depth GT.
Подробные команды и пороги находятся в протоколе выше.

```text
configs/                     кандидаты, эксперименты, configs/runs для запусков
examples/toy/                воспроизводимая программная демонстрация
src/planetary_vlm/
  contracts.py, io.py, cli.py общий контракт и команды
  datasets/
    base.py                  BaseData: общий интерфейс, пути и версии
    mars.py                  MarsData: загрузка, пары, стерео, проверки, сборка
    lune.py                  LuneData: лунные источники, разметка и сборка
    sources.py               общий сетевой транспорт и provenance загрузки
    preparation.py           общая сборка semantic track с изоляцией GT
    _mars/                   приватные реализации Mastcam, не CLI-скрипты
    cahvor.py                низкоуровневая геометрия камер
  download_dataset.py        явная загрузка источников
  prepare_dataset.py         подготовка датасета; эта функция вызывается CLI
  models/                    registry, модели и архитектурный preprocessing
  inference/                 config и последовательный runner
  transforms/                общие операции изображений и manifest вариантов
  evaluation/                parser, метрики, reports и bootstrap
docs/protocols/               входы, GT, модели и протокол оценки
docs/research/                первичные источники и решения по сравнению моделей
data_rover/                  existing rover data: raw → interim → processed → manifests
data_orbital/                separate Mars/Moon orbital sources and annotations
outputs/                     runs, reports, prepared images
paper/                       будущий текст статьи и таблицы
tests/                       проверка счётчиков, границ данных и интеграции
```

Роли Codex: researcher (GPT-6 Luna), vlm-expert, data-eval. Общее владение и
порядок работы зафиксированы в `AGENTS.md`.

## Ограничения текущей версии

Free-form QA требует отдельной рубрики: сейчас evaluator показывает
`unsupported_scoring`. Bootstrap реализован для accuracy по одинаковым request IDs;
F1/FPR intervals и pairing clean/corrupt по parent IDs ещё предстоит добавить.
Параметры деградаций, GT mappings и splits замораживаются после development-пилота.
Occlusion visibility остаётся unknown до mask-aware проверки. Стерео Mastcam
работает как технический пилот: независимая точность и admission глубины ещё
не подтверждены. Normals и telemetry converter M2020 пока отсутствуют.
Текущее elapsed время первого запроса включает lazy загрузку модели, warmup
не выполняется; эти логи ещё не являются сопоставимым hardware benchmark.

Кандидатные реестры `configs/models.toml` и `configs/datasets.toml` описывают
admission перед научным benchmark. Run TOML предназначен для технического запуска;
он не доказывает, что checkpoint/dataset прошёл admission или leakage audit.

## Проверка реализации

Историческая диагностика SpaceLLaVA описана в
[FP16/NF4-контроле v12](docs/protocols/spacellava_precision_v1.md):
два прежних val-снимка, 16 запросов на режим, два последовательных процесса на
одной GPU. Нужна минимум 32 GiB свободной VRAM по preflight, не T4/16 GB.
Pinned native builder запускается без адаптера/runner, без offload/fallback.
Прежние исполняемые notebooks и ZIP-комплекты больше не поставляются.
Старый NF4 на T4 не заменяет новую ветку.
V7/v8 показали различающиеся описания, но зависимость закрытого ответа от первого
варианта; [v9](docs/protocols/spacellava_fresh_process_v1.md) воспроизвёл те же
четыре ответа при независимых загрузках без прошлой истории. Это наблюдения
в 4-bit конфигурации, не установленная причина. JSON-отчёты диагностики сохранены.
В v10 повторились ответы на разные изображения;
разные logits сами по себе не означают разные greedy-ответы и не устанавливают
причину. V11 воспроизвёл все 16 ответов/токенов без нашего адаптера; GPU-запуск
нового v12 ещё не выполнен. Научный benchmark не
допускается автоматически; основные вопросы, эталоны и результаты не менялись.

FastAPI backend с SQLModel/SQLite: [`инструкция`](src/api/README.md).
Запуск: `python -m uvicorn api.main:app --reload`.
Swagger доступен на `http://127.0.0.1:8000/docs`.

React-интерфейс: [`frontend/README.md`](frontend/README.md).
Из папки `frontend/`: `npm.cmd ci`, затем `npm.cmd run dev`.
Сайт: `http://127.0.0.1:5173`. Для обработки заданий запустите API с диспетчером
и два model services по [инструкции](deploy/README.md); графики используют
сохранённые сервером результаты. Старый `api.worker` больше не используется.

Полный pipeline SpaceLLaVA → тексты и NASA–IBM IMP → карты:
[`FULL_SYSTEMS.md`](src/planetary_vlm/inference/FULL_SYSTEMS.md).
Команда подготовки: `python -m planetary_vlm full-systems --prepare-only`.
Результаты каждого запуска сохраняются в `outputs/<название модели>/<run_id>/`.

```powershell
& $benchmarkPython -m unittest discover -s tests -v
```

Тесты не загружают веса. Они проверяют реальные ошибки счётчиков, смешивание GT,
отсутствующие ответы, полноту масок, устойчивость IDs и resume.
