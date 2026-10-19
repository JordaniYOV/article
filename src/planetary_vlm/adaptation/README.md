# Mars crater adaptation — CLI pipelines

Новая основная задача: **бинарная семантическая сегментация каталожных кратеров
на орбитальных THEMIS thermal-IR снимках Марса**. Это не VLM QA и не rover navigation.
Старый `linear_head_approach/`, его конфигурации и опубликованные головы не меняются:
их код входит в исторические identities. Новые веса обучаются с нуля поверх
исходного foundation backbone, не продолжают прежние лунные головы.

## Структура

| Папка | Подход | Обучается |
|---|---|---|
| `linear_probe/` | A: замороженный Lunar-FM + 1×1 Conv + фиксированный bilinear resize | Только линейная голова |
| `frozen_decoder/` | B: замороженный Lunar-FM + лёгкий multilayer pyramid decoder | Только декодер |
| `lora_decoder/` | C: Lunar-FM с LoRA + **тот же** декодер, что в B | LoRA и декодер |
| `sam_baseline/` | D: frozen SAM ViT-B, automatic grid prompts, фиксированный геометрический фильтр | Ничего |
| `common/` | Данные, normalization, model adapter, cache, training, evaluation, отчёты и CLI | Общая реализация без копирования циклов |

В каждой папке метода есть свой `config.toml`, `pipeline.py`, `__main__.py` и README.
Запуск: `python -m planetary_vlm.adaptation <method> <stage> ...`.
Можно напрямую: `python -m planetary_vlm.adaptation.linear_probe <stage> ...`.

## Что уже проверено, а что ещё нет

Реализованы CPU-подготовка/аудит, plan, реальные model-smoke/extract/fit/resume/
evaluate/predict entrypoints и агрегирование трёх seeds. **Реальный pretrained
forward/backward на RTX 3060 Ti/T4 пока не проверен.** Все21 новых software-тестов
прошли в отдельной CPU-среде с torch2.8.0+cpu: gradients, frozen weights, checkpoint/
resume и повторная оценка. Это маленькие синтетические фикстуры, **не** реальные
IBM/SAM, не проверка загрузки pretrained NAC и не научное обучение. В API-среде
Torch отсутствует, поэтому6 Torch-тестов там явно пропускаются. Наличие runnable
кода не подтверждает VRAM, скорость или качество.
Никакого автоматического обучения, установки зависимостей, скачивания весов или данных нет.

Повторить software-проверки в выбранном ML-окружении:
`python -B -m unittest discover -s tests -p test_adaptation.py -v`.

## 1. Окружение и assets

Используй **новое** окружение Python 3.12, например `.venv-adaptation`; не обновляй
среду уже опубликованных голов. Из корня проекта, после установки подходящей
пары CUDA torch/torchvision для своего компьютера:

```powershell
$adaptPython = ".\.venv-adaptation\Scripts\python.exe"
& $adaptPython -m pip install -r requirements/adaptation.txt
& $adaptPython -m pip install --no-deps -e .
```

Первое окружение создаётся обычным Python 3.12: `py -3.12 -m venv .venv-adaptation`.
Для T4/Colab нужен отдельный набор CUDA wheels под Linux, не Windows-окружение.
`requirements/adaptation.txt` — диапазоны зависимостей, **не проверенный lock**.
Полный список фактически установленных версий сохраняется в identity каждого запуска.
После smoke сохрани своё окружение `python -m pip freeze` и не меняй его посреди run.

IBM source: `external/ibm-lfm`, commit `d54c67aad513cb9daca444afa425cfb278e4fbf8`.
Уже имеющиеся foundation `checkpoint.pt` и `config.yaml`:
`weights/encoder_heads/ibm/backbone/`. Проверяются точные известные SHA-256.
Файлы и external Git checkout не скачиваются/не исправляются автоматически.

Используется штатная pretrained **`nac` embedding, один канал**, не VIS5 и не
детектор, уже дообученный на кратерах. THEMIS не становится NAC-изображением:
это явно объявленная адаптация существующего grayscale-входа к другому сенсору.
Сторонние modality weights могут быть лишними в checkpoint; отсутствие **любого
активного** параметра/buffer блокирует загрузку, случайные веса не подставляются.

Бюджетный профиль общий для A/B/C: native PNG 512² → bilinear intensity 256²,
DN/255, mean/std только по resized train; patch16, блоки 2/5/8/11, native grid16².
Смена pretrained patch size обрабатывается опубликованным FlexiViT PI-resize
в IBM embedding loader; это фиксированная часть протокола, не обход shape-check.
Никаких выдуманных спектральных каналов, depth или новых GT нет.
Input/patch size можно изменить только новым config/output + новым smoke.

## 2. Безопасная проверка плана, без моделей

```powershell
& $adaptPython -m planetary_vlm.adaptation linear_probe plan
& $adaptPython -m planetary_vlm.adaptation frozen_decoder plan
& $adaptPython -m planetary_vlm.adaptation lora_decoder plan
& $adaptPython -m planetary_vlm.adaptation sam_baseline plan
```

`plan` ничего не пишет, не импортирует Torch и не загружает модели. `audit` проверяет
manifests; `audit --deep` проверяет hash/формат каждой image-mask пары на CPU.
`prepare` проходит только train для статистик и выбора двух smoke controls
(с пустой каталожной маской / с foreground). Test не используется для настройки.

## 3. Обучение A/B/C — последовательно

```powershell
& $adaptPython -m planetary_vlm.adaptation linear_probe train
& $adaptPython -m planetary_vlm.adaptation frozen_decoder train
& $adaptPython -m planetary_vlm.adaptation lora_decoder train
```

Каждая команда: prepare → реальный smoke на двух TRAIN controls → extract для
frozen-подходов → fit seeds13/37/73 → val каждого seed → mean/std val.
GPU-стадии идут отдельными процессами и используют тот же OS GPU-lock, что UI worker.
Останови benchmark до обучения; одновременно несколько GPU jobs не стартуют.
`train` **никогда не оценивает test**. Нет автоматического уменьшения resolution,
смены precision или OOM fallback. Ошибка требует отдельного нового протокола/smoke.

На этапе отладки вместо всей серии:

```powershell
& $adaptPython -m planetary_vlm.adaptation linear_probe prepare
& $adaptPython -m planetary_vlm.adaptation linear_probe smoke
& $adaptPython -m planetary_vlm.adaptation linear_probe extract
& $adaptPython -m planetary_vlm.adaptation linear_probe fit --seed 13
& $adaptPython -m planetary_vlm.adaptation linear_probe evaluate --seed 13 --split val
```

A/B используют один общий fp16 feature-cache для одинаковых data/preprocessing/
encoder/environment. Кэш четырёх taps для 4 500 train+val занимает примерно **6,59 GiB**
при 256/patch16; проверяется свободное место с запасом 1 GiB. Test в cache не попадает.
Маски/признаки читаются по одному batch, не накапливаются целиком в RAM.
C **не использует этот cache**: LoRA меняет представления при каждом optimizer step.

Общий бюджет: batch1, accumulation4, max40 epochs, patience8, AdamW + cosine,
gradient clipping, fp16 autocast/scaler на CUDA (CPU32), BCEWithLogits + .5 Dice.
Linear lr1e-3; B/C decoder lr3e-4; C LoRA lr1e-4. Positive weight из TRAIN, cap10.
Нет image/feature augmentation: поворот кэшированных features не эквивалентен
повторному encoding повёрнутого изображения с positional embeddings.
Все различия LR/параметров фиксируются, не выдаются за полностью одинаковый бюджет.
При C=768, decoder width64: A имеет769 обучаемых параметров; B492865;
LoRA rank4 добавляет147456, всего C640321. Это counts, не проценты от модели.
Для C rank4/alpha8 на **всей fused qkv projection** каждого из12 encoder blocks;
не только Q/V. Base weights, embeddings и norms заморожены; non-reentrant activation
checkpointing включён. Head B/C одна и та же, инициализация при одинаковом seed одинакова.

Loss использует area-resized soft targets на256²; оценка всегда на исходных512².
Threshold из [.2,.3,.4,.5,.6,.7,.8] и checkpoint выбираются только на val по scene-macro
IoU; tie → меньший threshold, более ранняя эпоха. Нельзя выбирать seed по test.

## 4. Resume и логи

```powershell
& $adaptPython -m planetary_vlm.adaptation linear_probe train --resume
& $adaptPython -m planetary_vlm.adaptation lora_decoder fit --seed 13 --resume
```

Сохраняются trainable weights, optimizer, scheduler, scaler, epoch, patience,
loader RNG, Python/NumPy/Torch/CUDA RNG. Прерванная эпоха повторяется от последней
завершённой. Старые различные identity не переписываются: dataset/config/code/
weights/environment/hardware должны совпадать; иначе новый `--output`.
Seeds контролируют случайность, но CUDA resize/SDPA **не объявляются побитно
детерминированными**. Checkpoints с RNG загружаются weights_only=False: только
собственные доверенные локальные `.pt`, не чужие вложения/URL.

`outputs/mars_adaptation_v1/<method>/seed_<seed>/`: `identity.json`, `best.pt`,
`last.pt`, `history.json`, `complete.json`, `events.log`, `evaluation/...`.
Логи идут и в консоль, и в файлы. Coordinator сохраняет вывод каждого процесса
в `<method>/logs/`. Признаки в общей `cache/`, статистики в `preparation/`.

## 5. Отдельная финальная оценка

```powershell
foreach ($seed in 13,37,73) {
    & $adaptPython -m planetary_vlm.adaptation linear_probe evaluate --seed $seed --split test --allow-test
}
& $adaptPython -m planetary_vlm.adaptation linear_probe summarize --split test
```

Повторить для `frozen_decoder` и `lora_decoder` после фиксации протокола. Test
берёт threshold из best.pt; не подбирает его снова. Сохраняются исходные 0/1 PNG
предсказаний, их hashes, per-image TP/FP/FN/IoU/Dice и общие метрики. Mean/std по
seeds — training variance, не ensemble и не CI по независимым сценам. Test содержит
лишь4 coarse mosaic groups; не считать900 тайлов900 независимыми географическими сценами.

Image-only inference, без чтения reference mask:

```powershell
& $adaptPython -m planetary_vlm.adaptation linear_probe predict --seed 13 --image data_orbital/mars_crater_binary_v1/test/images/thm_dir_N00_270_201.png --prediction predictions/example.png
```

Новый CLI выдаёт mask-контракт для будущего API-адаптера. **Существующий heads-worker
и UI пока не переключены на эти новые checkpoint/input protocols.** Старые deployment
JSON не подходят; не подменяй ими новые runs и не переименовывай новые веса в старые.

## 6. SAM

См. `sam_baseline/README.md`. Ниже не выполняется обучение или поиск фильтров:

```powershell
& $adaptPython -m planetary_vlm.adaptation sam_baseline smoke --config my_sam_config.toml
& $adaptPython -m planetary_vlm.adaptation sam_baseline evaluate --config my_sam_config.toml --split val
```

## Научные ограничения

- Первичные targets — опубликованные эллиптические каталожные области, а не
  точные геологические ободы и не исчерпывающий список видимых кратеров.
- Сохранены publisher splits. С прежними локальными выборками пересекаются
  14val/62test; train8,44% crater pixels, val8,86%, test1,62%. Полный test не новый blind holdout.
- 94 пустые маски допустимы; фон не доказывает физическое отсутствие кратеров.
  Сохраняем empty-target breakdown; не называем все FP физическими hallucinations.
- A/B/C измеряют способы адаптации одной модели. Для вывода, что именно lunar
  pretraining помогает Марсу, нужен отдельный контролируемый pretraining baseline.
- Переход Moon NAC→Mars THEMIS меняет планету, спектр, сенсор и масштаб вместе.
- SAM class-agnostic, геометрический фильтр не превращает его в обученный crater
  classifier. Нет доказательства отсутствия космических снимков в pretraining SAM.
- Эта реализация — фиксированный воспроизводимый baseline, не гарантия лучшей
  архитектуры или точности. Улучшения выбираются на val отдельными версиями протокола.

Источники: [IBM publisher source](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model),
[SAM code](https://github.com/facebookresearch/segment-anything/tree/6fdee8f2727f4506cfbbe553e23b895e27956588),
[Mars-Bench source](https://huggingface.co/datasets/Mirali33/mb-crater_binary_seg).
