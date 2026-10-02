# Полный pipeline SpaceLLaVA и NASA–IBM

Это отдельный технический pipeline для сравнения полных систем. Основной
эксперимент с одинаковыми обучаемыми probe heads остаётся в
`linear_head_approach/`. Здесь используется готовая IMP-голова IBM, обучение
не запускается. Подготовка входов и тесты не являются результатами моделей.

## Последовательность

1. Прочитать конфигурацию `configs/full_systems.toml`.
2. Выбрать lunar IMP val/test из `orbital_300_v3`, проверить SHA-256 снимков.
3. Создать clean, shear, radiation, lens_flare, hard_shadows и combined.
4. Сохранить image-only запросы и отдельный журнал происхождения данных.
5. Проверить локальный код и файлы каждой модели.
6. Загрузить SpaceLLaVA один раз, получить тексты, освободить память.
7. Загрузить IBM backbone и опубликованную IMP-голову один раз, получить карты.
8. Сохранить каждый ответ сразу; закрыть модель даже при исключении.

По умолчанию используются 21 снимок и 126 запросов на модель. Только IMP
соответствует выбранной готовой голове IBM; Mars, другие lunar классы и DEM
не включены в этот запуск. Веса, репозитории и датасеты автоматически не
скачиваются. Ground truth не читается и не передаётся в модели.

## Команды из корня проекта

```powershell
# Подготовка без загрузки моделей (уже выполнена).
& .\.venv-benchmark\Scripts\python.exe -m planetary_vlm full-systems --prepare-only

# Проверка локальных assets без GPU и импорта ML-библиотек.
& .\.venv-benchmark\Scripts\python.exe -m planetary_vlm full-systems --check-only

# После отдельного provision и разрешения на smoke test:
# Один исходный снимок × 6 условий; отдельные каталоги *_limit1.
& .\.venv-benchmark\Scripts\python.exe -m planetary_vlm full-systems --model SpaceLLaVA --limit 1
& .\.venv-benchmark\Scripts\python.exe -m planetary_vlm full-systems --model NASA-IBM-Lunar-Foundation-Model --limit 1

# После успешного smoke test и разрешения на полный запуск:
& .\.venv-benchmark\Scripts\python.exe -m planetary_vlm full-systems

# Продолжить прерванный запуск с неизменными config/input/checkpoint.
& .\.venv-benchmark\Scripts\python.exe -m planetary_vlm full-systems --resume
```

Существующий запуск без `--resume` не перезаписывается. Уже сохранённые
ошибки запросов также считаются обработанными: для повторения такого запроса
нужен новый `run_id`. Ошибка загрузки модели оставляет журнал запуска без
ответов; после устранения причины можно использовать `--resume`. Если один
набор assets отсутствует, другая система всё равно проверяется/запускается.

Для изменения параметров эксперимента задайте новый `prepared_dir` в общей
конфигурации и новый `run_id` в обеих конфигурациях моделей. Идентификаторы
изображение/условие должны интерпретироваться вместе с hash протокола. Нельзя
смешивать ответы разных протоколов только по совпадению `request_id`.

## Локальные файлы моделей

| Система | Ожидаемые assets |
|---|---|
| SpaceLLaVA | `external/LLaVA/`; полный merged checkpoint в `weights/SpaceLLaVA/`; локальный CLIP-336 checkpoint в `weights/clip-vit-large-patch14-336/` |
| IBM IMP | `external/NASA-IBM-Lunar-Foundation-Model/`; `weights/IBM-IMP/config.yaml` и `ni_lfm_ps8_frozen_s44.ckpt`; `weights/NASA-IBM-Lunar-Foundation-Model/backbone/config.yaml` и `checkpoint.pt` |

Точные commit/revision указаны в `configs/runs/*_orbital.toml`. Код проверяется
по Git HEAD и отсутствию изменений tracked files. Фактически загруженные
локальные файлы хешируются; declared HF revision сама по себе не доказывает
происхождение локальных весов. При provision необходимо сохранить соответствие
downloaded snapshot указанной revision.

SpaceLLaVA использует native LLaVA `LlavaLlamaForCausalLM`, шаблон `llava_v1`,
image token и CLIP processor из checkpoint. Vision tower создаётся до загрузки
полного checkpoint, чтобы восстановить именно tensors SpaceLLaVA.
Предусмотрена 4-bit NF4 загрузка LLM; vision tower/projector сохраняются вне
этого квантования. FP16 13B требует больше доступных 16 GB только для весов.
Ни quantization, ни CUDA на RTX 5070 Ti пока не проверены.

Нужны локально установленные PyTorch/CUDA, совместимые с GPU, Transformers,
Accelerate, bitsandbytes, Pillow и upstream LLaVA. Для IBM нужны upstream
NASA–IBM и его TerraTorch runtime, PyYAML, NumPy и Pillow. Старый LLaVA и
TerraTorch могут требовать разные dependency versions: совместимость одного
окружения не подтверждена. В таком случае запускайте команды `--model`
последовательно из отдельных окружений; общий input manifest остаётся тем же.
Полный последовательный запуск одной командой требует совместимого окружения.

## Файлы результатов

```text
outputs/
  SpaceLLaVA/full_systems_v1/
    run.json           config/options, input hashes, seed, dependency versions
    loading.json       фактические hashes весов, runtime и время загрузки
    requests.jsonl     копия model-visible запросов
    predictions.jsonl  raw_response: неизменённый сгенерированный текст
  NASA-IBM-Lunar-Foundation-Model/full_systems_v1/
    run.json
    loading.json
    requests.jsonl
    predictions.jsonl  raw_response: JSON с именами карт и hashes
    maps/
      <sha256(request_id)>.png  class IDs 0=Background, 1=IMP
      <sha256(request_id)>.npz  float32 logits и softmax probabilities (2,H,W)
```

Каждая строка ответа содержит `request_id`, `model_id`, status, время и
`error_message`. Журнал append-only с flush/fsync после каждого ответа.
Карты сохраняются через временные файлы и atomic replace до записи ответа;
при resume hashes уже сохранённых карт проверяются. PNG — class IDs, поэтому
обычный viewer отображает foreground почти чёрным; `.npz` хранит численные
результаты. Маски GT сюда не копируются.

## Нормализация и научные ограничения

Опубликованная IBM-голова принимает одноканальные NAC reflectance 256×256;
официальные mean/std: 0.027375 / 0.014783. В текущей коллекции остались PNG:
`round(clip(reflectance / 0.12, 0, 1) * 255)`; это подтверждено кодом
`scripts/prepare_orbital.py`, хотя итоговый manifest потерял этот display field.
Адаптер явно использует приблизительное восстановление `PNG / 255 * 0.12`.
Клиппинг, квантование и потеря исходной nodata необратимы. Этот PNG track
пригоден для технического пилота; воспроизведение publisher scores требует
исходных reflectance TIFF. Искажения применяются к общей PNG-репрезентации.

Текстовая QA accuracy и нативные segmentation metrics — разные величины.
Этот pipeline сохраняет ответы, не вычисляет общий рейтинг моделей. Для оценки
геометрически искажённых карт потребуется отдельно преобразовать GT тем же
полем и учесть новые invalid pixels. Подвыборка IMP была отобрана по positive
маскам; десять официальных test tiles не дают валидную оценку false positives.
Holdout/training overlap требует самостоятельного аудита до научных выводов.

## Источники интерфейсов

- [Remyx SpaceLLaVA](https://huggingface.co/remyxai/SpaceLLaVA).
- [Native LLaVA inference](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/eval/run_llava.py).
- [IBM IMP release и точный config](https://huggingface.co/nasa-ibm-ai4science/IMP-Segmentation-NASA-IBM-Lunar-Foundation-Model/tree/10a09ffae35ef5712c40c2a1fc41ec82d4881ea6).
- [NASA–IBM integration](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model/tree/d54c67aad513cb9daca444afa425cfb278e4fbf8/terratorch_integration).
