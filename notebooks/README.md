# Planetary vision models: Google Colab

## Текущий IBM: полный clean-прогон 100 снимков / 400 ответов

Обновлён **тот же** `ibm_val14_colab.ipynb`, но теперь он запускает full100/v2.
Использовать новый `outputs/colab/ibm_full100_v2.zip`, а не ZIP val14.
50 Moon + 50 Mars, только Q1–Q4 и clean, без искажений и обучения.
Порог, FP32, опубликованные веса и точные зависимости прежнего T4 пилота сохранены.

Загрузить ZIP в `MyDrive/planetary_benchmark/`, выполнять notebook сверху вниз;
на новой VM `DOWNLOAD_WEIGHTS=True`, после smoke — `RUN_BENCHMARK=True` в разделе 7.
Результат: `ibm_full100_v2_clean_results.zip`. Для resume тот же ZIP/RUN_ID,
`RESUME=True` и новый smoke. Полный GPU-прогон ещё не выполнен.

Выполненный val14 notebook сохранён побайтно в
`outputs/colab/ibm_full100_v2_checks/previous_ibm_val14_colab.executed.ipynb`.
Новый eval-kit `outputs/colab/ibm_full100_v2_eval/` остаётся локальным, вне Colab.
Разметка кандидатная; исходные val/test сохранены, pool включает development-сцены.
Подробности: [`ibm_full100_colab_v2.md`](../docs/protocols/ibm_full100_colab_v2.md).

## Предыдущий IBM clean-пилот: 14 val-сцен

Исторический ZIP: `outputs/colab/ibm_val14_v1.zip`; текущий notebook уже обновлён выше.
7 Moon + 7 Mars из отдельных исходных групп → **56 ответов Q1–Q4**.
Два прежних smoke-контроля запускаются отдельно и не входят в эти 14 сцен.
Только clean/val; пороги, FP32 и веса сохранены. Версии пакетов закреплены по
успешному T4 smoke. Новая VM: `DOWNLOAD_WEIGHTS=True`; затем ячейки по порядку,
smoke и `RUN_PILOT=True` в разделе 7, без повторного запуска parameters.
Прислать `ibm_val14_v1_clean_results.zip` из Drive.

Локальный eval-kit находится в `outputs/colab/ibm_val14_v1_eval/`, **не загружать
его в Colab**. Подготовка и CPU-проверки не означают, что новый GPU-пилот выполнен.
Подробности, resume и оценка:
[`ibm_val14_colab_v1.md`](../docs/protocols/ibm_val14_colab_v1.md).
Прежний выполненный notebook сохранён. Ниже — предыдущие варианты запуска.

## IBM: кратеры Q1–Q4 · отдельный запуск v1

`ibm_craters_colab.ipynb` + `outputs/colab/ibm_craters_v1.zip`.
Новая Colab VM с T4; не смешивать окружение с SpaceLLaVA. ZIP положить в
`MyDrive/planetary_benchmark/`. Для первой загрузки явно задать
`DOWNLOAD_WEIGHTS=True`, выполнять ячейки сверху вниз. Сначала 2 lunar val
изображения, 8 ответов и overlays; полный clean-прогон 100 изображений / 400
ответов включается отдельно после просмотра smoke. Сайт/сервисы не нужны.

Готовый опубликованный NAC crater detector, не IMP-голова и не LLM. Без обучения
и квантования, FP32; совместимость T4 и 12.67 GB RAM **не подтверждена**, пока
реальный smoke не выполнен. Источники/нормализация/ограничения/resume:
[`ibm_craters_colab_v1.md`](../docs/protocols/ibm_craters_colab_v1.md).
Ниже сохранены прежние SpaceLLaVA запуски.

## Текущий контроль: FP16 vs NF4 v12

`spacellava_language_colab.ipynb` и `outputs/colab/spacellava_precision_v12.zip`.
Две последовательные загрузки, по 16 тех же запросов на двух val-снимках.
Нужна минимум 32 GiB свободной VRAM по preflight; T4/16 GB не допускается.
При смене GPU оба режима запускаются на ней заново. ZIP результата v10 нужен
только для фиксированных входов/хэшей, не как готовая NF4-ветка.

Defaults: `RUN_PILOT=False`, `DOWNLOAD_WEIGHTS=False`. На новой VM разреши
загрузку весов явно, если локальных assets нет. Режимы не переключаются при OOM;
ошибка первой ветки останавливает вторую. Полный GPU-прогон v12 не выполнен
локально. Детали: `docs/protocols/spacellava_precision_v1.md`. Ниже — история.

## Текущая диагностика: standalone upstream v11 (17 октября 2026)

Используй обновлённый `spacellava_language_colab.ipynb` и
`outputs/colab/spacellava_reference_v11.zip`. Положи новый ZIP и прежний результат
`spacellava_evidence_v10_evidence_55164967_results.zip` в
`MyDrive/planetary_benchmark/`. По возможности сохрани прежнюю T4 VM, веса и
окружение. После подготовки явно включи `RUN_PILOT=True`.

Одна загрузка, те же два val-изображения и 16 запросов v10. Инференс через
upstream LLaVA, без нашего адаптера/runner; NF4 и входы проверяются на совпадение.
Это не другой движок: общая ошибка Torch/Transformers/LLaVA всё ещё возможна.
Реальный GPU-путь v11 здесь не выполнен. Детали и границы выводов:
`docs/protocols/spacellava_reference_v1.md`. Ниже сохранена история запусков.

## Текущий запуск Q1–Q4 (15 октября 2026)

Используй **`spacellava_q1_q4_colab.ipynb`**, а не старый notebook со свободными
описаниями. Пакет: `outputs/colab/spacellava_q1_q4_v5_clean.zip`. В нём 100 снимков
(50 Moon + 50 Mars) и 400 clean-запросов Q1–Q4. Эталоны/маски/высоты исключены.
Локально все 400 кандидатных ответов определены, но окончательный допуск scoring
GT ещё не выполнен. До просмотра предсказаний следует завершить фиксацию эталонов.

1. Загрузи ZIP в `MyDrive/planetary_benchmark/`, новый notebook — в Colab.
2. Выбери GPU и выполни ячейки до smoke включительно.
3. Если старая VM жива, ячейка `reuse-assets` подключит её weights/runtime без
   копирования. Иначе явно включи `DOWNLOAD_WEIGHTS=True`, когда готов к загрузке.
4. После успешного smoke отдельной ячейкой задай `RUN_BENCHMARK = True` и выполни
   ячейку `run`. Не перезапускай parameters без нового smoke (меняется SESSION_ID).
5. Результаты: `MyDrive/planetary_benchmark/runs/SpaceLLaVA/spacellava_q1_q4_v5_clean/`.

Новый workspace `/content/planetary_q1_q4_v5` не перезаписывает старые результаты.
Для resume сохраняй ZIP, RUN_ID, workspace и версии; восстанови assets и выполни
новый smoke. Облачный GPU-запуск здесь не выполнен: локальные проверки — только
CPU/schema/bootstrap, не доказательство CUDA/VRAM/Drive совместимости.

Повторный экспорт (существующий ZIP не перезаписывается):

```powershell
python -m planetary_vlm.colab export --collection data_orbital/orbital_300_v4 --crater-questions data_orbital/crater_annotations_v5 --clean-only --output outputs/colab/spacellava_q1_q4_v5_clean.zip
```

Замена Mars018 и неизменность всех масок/высот проверяются в
`mars_image_replacement_v1_audit.ipynb`. Старый `spacellava_colab.ipynb` с
пользовательскими outputs не изменён. Ниже сохранён прежний общий workflow.

Открой `spacellava_colab.ipynb` через Colab → File → Upload notebook, выбери GPU.
Сайт, Docker и отдельные HTTP-сервисы для этого прогона не нужны. Пока готовится
только SpaceLLaVA; IBM будет запускаться позже в другом окружении.

## Локальная подготовка (без GPU и загрузок)

Из корня проекта с установленными Pillow, NumPy и OpenCV:

```powershell
& .\.venv-benchmark\Scripts\python.exe -m planetary_vlm.colab export --output outputs/colab/spacellava_orbital_300_v3.zip
```

Экспорт читает только `data_orbital/orbital_300_v3/` и код проекта. Проверяет SHA-256,
фиксирует renderer/seed/параметры и создаёт переносимый ZIP с кодом, requirements,
image-only requests и source provenance. Повторный экспорт в тот же файл запрещён:
для другого протокола выбирай другое имя. Веса, `.env`, Git, masks, height arrays
и содержимое защищённого архива не упаковываются. Исходные файлы не меняются.

Коллекция содержит 300 исходных изображений, из них 100 publisher train исключены.
200 val/test × 6 условий дают 1200 запросов. Val — development, не final test;
related scenes и исходные splits остаются в provenance. Это 200 исходных снимков,
а не 1200 независимых наблюдений. Для быстрого CPU-экспорта есть `--clean-only`.

Текущие условия: clean, shear, radiation, lens_flare, hard_shadows, combined.
Это реальные существующие transformations проекта, не ещё не реализованные
blur/noise/contrast/occlusion из предлагаемого E2. Все PNG-байты создаются **один
раз локально**, поэтому последующее окружение IBM не будет перерендеривать их
другими NumPy/OpenCV версиями.

## Что перенести

1. Сам `.ipynb` открыть в Colab.
2. ZIP загрузить в `MyDrive/planetary_benchmark/spacellava_orbital_300_v3.zip`.
3. При другом расположении изменить только `PERSIST`/`BUNDLE_ZIP` в параметрах.

Notebook устанавливает изолированный Python 3.12, Torch 2.8.0 CUDA 12.8 и
`requirements/spacellava.txt`. Windows wheels пропускаются по platform markers.
Kernel Colab не используется как model environment; в него не надо импортировать
Torch. Отдельный Python provisioned через [uv](https://docs.astral.sh/uv/guides/install-python/).
[Официальная CUDA-wheel пара](https://pytorch.org/get-started/previous-versions/#v280)
— кандидат, не подтверждённая Colab/GPU конфигурация.

## Веса и запуск

`DOWNLOAD_WEIGHTS=False` и `RUN_BENCHMARK=False` по умолчанию. Для первой установки
поставь `DOWNLOAD_WEIGHTS=True` и выполни assets-ячейку: скачиваются опубликованные
SpaceLLaVA/CLIP веса по закреплённым revisions и LLaVA code по точному commit.
Исходные model datasets не скачиваются, обучение не запускается.

Полный merged checkpoint нужен и при NF4 inference: примерно 28 GB для SpaceLLaVA
и CLIP вместе, плюс Python/CUDA библиотеки и inputs. Helper требует 35 GiB свободного
локального диска перед скачиванием. Веса по умолчанию на временном локальном диске;
после потери VM их нужно загрузить снова. Бесплатного Drive 15 GB для их полного
кеша недостаточно. `WEIGHTS_SOURCE` может указывать на уже существующий кеш точных
revisions: он копируется локально, никогда не изменяется этим notebook.

После assets preflight запусти **1-image clean val smoke**. Сбой загрузки,
генерации, полностью восстановленного checkpoint или CPU/disk-offload запрещает
большой прогон. Mock не заменяет этот тест. Только после успеха включай
`RUN_BENCHMARK=True`. Каждый subprocess завершается и освобождает модель перед
следующим: не оставляй kernel model objects и не запускай два notebook одновременно.

Default prompt старого `spacellava_orbital_300_v3.zip` собирает свободные описания: это не реализованный benchmark
Q1–Q8, не автоматические accuracy/hallucination scores. Нет ответов, придуманных
из названия датасета, rock mask не объявляется физической опасностью, height
reference не объявляется model input. Для научных task metrics нужен отдельно
зафиксированный question/GT converter и task-specific evaluator.

### Подготовленные вопросы Q5–Q6, 14 октября 2026

`outputs/colab/spacellava_orbital_questions_v1.zip` — новый **clean-only** пакет:
100 height images (50 Moon + 50 Mars), 200 closed-form requests. Эталоны
остаются локально в `data_orbital/orbital_questions_v1/targets.jsonl`, не в ZIP.
Это пилот вопросов/эталонов, не допущенное парное сравнение с IBM. Q1–Q4 и
Q7–Q8 пока заблокированы. Точные правила и команды оценки:
[`orbital_questions_v1.md`](../docs/protocols/orbital_questions_v1.md).

Существующий notebook и сохранённые пользовательские outputs не изменены.
Чтобы использовать его с новым ZIP, загрузи ZIP в прежний `PERSIST`, затем
замени значения в первой ячейке:

```python
BUNDLE_ZIP = PERSIST / 'spacellava_orbital_questions_v1.zip'
WORKSPACE = Path('/content/planetary_questions_v1')
RUN_ID = 'spacellava_orbital_questions_v1_clean'
DOWNLOAD_WEIGHTS = False  # Use already downloaded assets if still present.
RUN_BENCHMARK = False
RESUME = False
```

Новая рабочая папка нужна потому, что безопасная распаковка намеренно не
заменяет изменённый код старого пакета. Если текущая Colab VM всё ещё содержит
загруженные weights/runtime в `/content/planetary_colab`, перед распаковкой
запусти следующую ячейку, чтобы переиспользовать их **без копирования 28 ГБ**:

```python
previous = Path('/content/planetary_colab')
WORKSPACE.mkdir(parents=True, exist_ok=True)
for name in ('weights', 'external'):
    source, target = previous / name, WORKSPACE / name
    if source.is_dir() and not target.exists():
        target.symlink_to(source, target_is_directory=True)
```

Если VM уже потеряна, assets отсутствуют; их загрузка требует снова явного
`DOWNLOAD_WEIGHTS=True`, а не происходит автоматически. Дальше последовательно
распаковка → зависимости → assets → **новый smoke с вопросом**. Старый smoke
с общим описанием не заменяет проверку нового пакета. Успех runtime smoke
не гарантирует точность ответа или допустимую долю invalid; сначала посмотри
ответ и его формат. Только затем принимай решение о `RUN_BENCHMARK=True`.

## Checkpoints, resume и результаты

Рабочий каталог стабилен: `/content/planetary_colab`. Результаты идут в его
`outputs/SpaceLLaVA/<run_id>` на локальном диске. На Drive копируются полностью
записанные JSON/JSONL-файлы через временное имя + rename с SHA-проверкой,
каждые 10 ответов и при обычном выходе/Interrupt. Не append на Drive/FUSE:
inference timing не включает checkpoint copying.

На Drive сохраняются predictions, requests, provenance, loading hashes,
bundle metadata, environment/GPU/driver и `pip-freeze.txt`. Жёсткий disconnect
может потерять последние ещё не checkpointed ответы; Google Drive операции
также могут завершиться ошибкой. Потерянные ответы будут заново вычислены.

После disconnect повтори setup с теми же ZIP/workspace/RUN_ID. Восстанови assets,
запусти новый smoke. Включи `RESUME=True` и `RUN_BENCHMARK=True`. В режиме resume
устанавливай зависимости из сохранённого `pip-freeze.txt`, а не заново разрешённых
диапазонов: версии, model hashes, input paths/bytes и GPU type должны совпасть.
Смена этих условий требует нового RUN_ID; не редактируй run.json для обхода.
Если checkpoint отсутствует, нужен RESUME=False. `MAX_REQUESTS` тоже фиксирует
протокол: расширить его при том же RUN_ID нельзя. Не сравнивай Colab latency с
RTX 5070 Ti как одну аппаратную конфигурацию.

IBM IMP-head применима только к допустимой lunar IMP подвыборке, не ко всем
200 изображениям/задачам. Она сможет переиспользовать те же допустимые PNG-байты
пакета, но её setup и преобразование native outputs в ответы — отдельный этап.

## Статус проверки

Локально: notebook schema/AST, реальный bootstrap на synthetic ZIP, SHA/GT/train
gates и core-runner checkpoint/restore/resume в CPU fixtures. **Не выполнено**:
top-to-bottom в Colab, CUDA driver/memory, NF4 model load, generation и реальные
Drive/FUSE recovery. Для закрытия gap: выполнить ячейки notebook последовательно
до successful smoke, затем batch и restore/resume в настоящей Colab-сессии.

Colab GPU, runtime lifetime и quota не гарантированы; не обходить ограничения
notebook UI/idle timeout. [Официальный FAQ](https://research.google.com/colaboratory/faq.html).
