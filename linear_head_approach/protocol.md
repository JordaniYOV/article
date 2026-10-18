# Frozen crater heads v2 — protocol

Роль: supporting feature-utility experiment. Основной `research_plan.md` не меняется.
Это не обещание оптимальной головы: небольшой baseline + predeclared ablation,
который можно воспроизвести на ограниченной памяти.

## Data and target

Источник: [SomBench WAC crater detection](https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-WAC-Crater-Detection),
revision `20f800becb64a0c7ad314652a683f97f0281fe67`, source card license claim CC-BY-4.0.
Полная атрибуция, исходная выборка и hashes в `data_orbital/head_training_v2/provenance.json`.
120 ранее отобранных WAC изображений 512×512. Publisher polygon masks сохранены
побайтно; 50 дополнительных масок ранее rasterized из pinned publisher COCO.
Сейчас никакой новой ручной разметки и никакой подмены изображения не производится.

50 train / 39 val / 31 source-test; 36/39/31 записанных scene groups. Это не
гарантия отсутствия spatial overlap разных scene IDs — geographic footprint
аудит и pretraining-provenance аудит остаются ограничениями. Не объявлять set
независимым от pretraining IBM, использующего lunar corpora.

`inputs.jsonl` не содержит targets/masks/height/answers. `targets.jsonl` — отдельный
контракт для loss, evaluator и проверки целостности файлов в audit. Входные изображения и маски разделены
логически, а не как security boundary: оператор имеет доступ к обеим папкам.
Все assets проходят SHA-256; неизвестные семантические классы блокируют сборку.

## Encoders

**SpaceLLaVA:** `remyxai/SpaceLLaVA`, revision
`5a69f71bda9dea3e4d8113c1047c07fd3da483f8`; actual 391 vision-tower tensors из
`model-00006-of-00006.safetensors`. CLIP ViT-L/14-336 architecture config revision
`ce19dc912ca5cd21c8a653c79e251e808ccabcd1`; hidden layer -2, CLS excluded.
RGB square 512→official 336 processor, native spatial feature grid 24×24×1024.
Мультимодальный projector и LLM исключены. Tower vs original CLIP equality не
проверена; Remyx SpaceLLaVA не является OSUPCVLab LLaVA-LE.
Checkpoint names `vision_model.*` загружаются с точным полным соответствием
model.state_dict: legacy wrapped layout либо flat layout
[Transformers5.15.1 CLIPVisionModel](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/clip/modeling_clip.py).
Меняются только имена ключей, не тензоры; strict=True обязателен для всех391
параметров. Нельзя игнорировать missing/unexpected keys через strict=False.

**IBM:** original `NASA-IBM-Lunar-Foundation-Model` revision
`b657bdfe6fc4c0445d918f77a2d7b919805f3bce`, backbone checkpoint/config known hashes.
Не NAC crater-finetuned detector. Runtime revision
`d54c67aad513cb9daca444afa425cfb278e4fbf8`, официальный `LunarBackbone`, VIS5,
512×512, patch8, block11, official per-band means/std, 64×64×768→bilinear24×24.
Недостающие активные checkpoint parameters запрещены (никаких silent random weights).

Оба encoder FP32, eval, requires_grad=False, inference_mode, batch1. Отдельные
процессы. Нет автоматического OOM fallback, quantization или смены input size.
Кэш float32 CHW24×24, incremental atomic per-image save; resume проверяет code,
environment, input hashes, model hashes и preprocessing. Перед full extraction
нужен smoke двух разных изображений с текущим session ID, проверка конечности,
различия features и повторяемости при повторном вызове.

**Сравнение native-input**, не equal-information: IBM5band vs CLIP RGB. Изменение
общей сетки или modalities — отдельный protocol/run, не patch после просмотра test.

## Heads and training

Predeclared families: linear 1×1 и spatial 1×1→64/GN/GELU→3×3/GN/GELU→1×1.
Одинаковый topology/budget, не одинаковое число parameters; все counts публикуются.
Признаки channel-standardized по train pixels only. Encoder не загружается в
training process. По умолчанию нет feature-space augmentation: flip features не
эквивалентен повторному encoding перевёрнутого изображения с positional embeddings.

AdamW lr1e-3, wd.01, batch4, cosine schedule max80 epochs, patience12,
gradient norm clip1. FP32, deterministic algorithms, фиксированные seeds13/37/73.
Каждый снимок weight ∝1/число train tiles той же source group.
Loss на128²: positive-weight BCE + .5 soft Dice. Positive weight считается train-only,
с учётом групп и cap10. Маски area-downsampled: сохраняется fractional coverage,
не используется nearest с потерей всех маленьких polygons. Нет test-derived weights.

Windows/RTX3060Ti update: IEEE FP32, TF32 disabled in both extraction/inference,
strict deterministic algorithms. Head bilinear upsampling is implemented by fixed
separable matrices (same align_corners=False geometry), not CUDA interpolate backward:
the latter is [documented as nondeterministic](https://docs.pytorch.org/docs/2.12/generated/torch.use_deterministic_algorithms.html).
No silent warn_only fallback. CPU equivalence/backward tests and GPU smoke are separate.

Validation на512²: fixed thresholds .2,.3,.4,.5,.6,.7,.8, выбор максимального
scene-macro IoU. При равенстве — меньший threshold, затем более ранняя эпоха.
Это совместный val-подбор порога и эпохи в заранее фиксированном бюджете, не
независимая оценка качества. Для test используется best.pt и сохранённый threshold.

## Resume and reporting

`last.pt`: head, optimizer, scheduler, shuffle generator, Python/NumPy/CPU/CUDA RNG,
history, wait counter, epoch, identity. Запись atomic rename на том же filesystem.
Сохраняется после полной эпохи; прерванная эпоха повторяется из предыдущего состояния.
Exact identity mismatch закрывает resume. Нельзя загружать сторонние untrusted .pt:
checkpoints содержат Python RNG objects и читаются weights_only=False.

Для завершённых legacy runs предусмотрена узкая export/inference compatibility
policy (`compatibility_revisions.json`): точные manifests revisions
`b6577c17b0a39df6b42a15a027a5d85310474a35` (только IBM) и
`a0b6d57c97f86e231fe6aa0999e72657b0f272d9` (IBM/SpaceLLaVA). Между ними изменён
только CLIP loader; IBM encoder/preprocessing и head training неизменны. LF/CRLF
учтены как варианты точных исторических hashes. Current runtime тоже проверяется
полностью. Identity/checkpoint не модифицируются: original training identity и
отдельная compatibility attestation сохраняются в deployment и проверяются
worker. Это не универсальный bypass, не разрешение resume и не отмена проверки
окружения, весов, targets и threshold. Любые неизвестные изменения блокируются.

CLI-оценка test закрыта отдельным `--allow-test`; notebook требует завершить обе
модели до оценки. UI запускает test только при явном выборе этого split после
экспорта фиксированной головы. Audit проверяет целостность всех splits, включая
test; это не оценка предсказаний. Test не используется для выбора головы, seed,
эпохи, normalization или threshold.
Основной metric: среднее per-image IoU внутри группы, затем mean по source groups.
Дополнительно Dice, image-macro IoU, all-foreground/all-background baseline.
При empty prediction+GT IoU/Dice=1; пустые GT помечены в per-image records.
Не сворачивать все pixels в один pooled IoU как единственный показатель.

Парное сравнение: одна и та же семья, seeds и sample/group IDs. Сначала среднее по
seeds для каждого изображения, затем paired scene-group bootstrap2000, seed2026.
CI отражает сценовую неопределённость при выбранных seeds, не всю training variance;
seed-specific metrics сохраняются отдельно. Нельзя выбрать только лучший seed.

Каталожная маска — объединённые polygons, не отдельные instances. Нельзя из этой
маски корректно получить число кратеров или Q2–Q4 instance relations без нового
контракта. Crater mask не depth/height и не физическая hazard/cost map. Route planning
требует отдельной постановки и GT. Не представлять software smoke как научный результат.
