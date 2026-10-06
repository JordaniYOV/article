# Evaluating Vision Models on Planetary Surfaces

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
Основной планируемый опыт теперь сравнивает **замороженные визуальные энкодеры**
[`remyxai/SpaceLLaVA`](https://huggingface.co/remyxai/SpaceLLaVA) и
[`NASA–IBM Lunar Foundation Model`](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model)
с отдельно обучаемыми пространственными головами одного типа. Первый пилот
использует лунные WAC-маски кратеров; его дизайн и ограничения описаны в
[`linear_head_approach/protocol.md`](linear_head_approach/protocol.md). Сравнение
полных систем и вопросы E1–E5 остаются отдельным анализом в
[`docs/protocols/orbital_study.md`](docs/protocols/orbital_study.md).
Команды для отдельного компьютера с RTX 5070 Ti находятся в
[`linear_head_approach/gpu_runbook.md`](linear_head_approach/gpu_runbook.md).
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

## Установка

Создано отдельное рабочее окружение `.venv-benchmark` (Python 3.12). В нём
проект установлен editable с Pillow/NumPy. Старое `.venv` остаётся неисправным.

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

FastAPI backend с SQLModel/SQLite: [`инструкция`](src/api/README.md).
Запуск: `python -m uvicorn api.main:app --reload`.
Swagger доступен на `http://127.0.0.1:8000/docs`.

React-интерфейс: [`frontend/README.md`](frontend/README.md).
Из папки `frontend/`: `npm.cmd ci`, затем `npm.cmd run dev`.
Сайт: `http://127.0.0.1:5173`. Для обработки заданий запустите API и отдельный
`python -m api.worker`; графики используют сохранённые сервером результаты.

Полный pipeline SpaceLLaVA → тексты и NASA–IBM IMP → карты:
[`FULL_SYSTEMS.md`](src/planetary_vlm/inference/FULL_SYSTEMS.md).
Команда подготовки: `python -m planetary_vlm full-systems --prepare-only`.
Результаты каждого запуска сохраняются в `outputs/<название модели>/<run_id>/`.

```powershell
& $benchmarkPython -m unittest discover -s tests -v
```

Тесты не загружают веса. Они проверяют реальные ошибки счётчиков, смешивание GT,
отсутствующие ответы, полноту масок, устойчивость IDs и resume.
