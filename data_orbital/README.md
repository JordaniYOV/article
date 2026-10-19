# Orbital 300 v4

## Mars binary crater segmentation

`mars_crater_binary_v1/` — отдельная полная бинарная выборка Mars-Bench:
3 600 train / 900 val / 900 test, исходные орбитальные THEMIS PNG 512×512
и маски `0 = фон`, `1 = кратер`. Размеры, пиксели, имена и publisher splits
сохраняются. Это thermal IR, не видимые RGB-снимки. Фон означает отсутствие
каталожной разметки, а не доказанное отсутствие всех видимых кратеров.

Сборка/возобновление: `python -m planetary_vlm dataset mars-binary-build --allow-download`.
Проверка без сети: `python -m planetary_vlm dataset mars-binary-verify`.
Зависимости без моделей: `python -m pip install -e ".[datasets]"`.
Готовность подтверждается `metadata.json` и успешной полной проверкой; частичные
файлы загрузки сами по себе не означают готовый датасет. `audit.json` фиксирует
дубликаты и пересечения с прежними локальными коллекциями; эти test-снимки
нельзя автоматически объявлять новым слепым holdout. Старые коллекции не меняются.
Этот сборщик не запускает обучение и не подтверждает совместимость IBM с THEMIS.

Результат аудита полной v1: 5 400 пар, без точных/близких дубликатов (dHash ≤2)
и без общих filename-групп мозаики между splits. С прежними локальными
коллекциями пересекаются 14 val и 62 test; train-пересечений не найдено.
Список 838 остальных test-снимков — лишь кандидаты без обнаруженного локального
пересечения, а не доказательство отсутствия в предобучении моделей.
Доля crater-пикселей: train 8,44%, val 8,86%, test 1,62%; поэтому pixel accuracy
не подходит как основной показатель — нужны crater IoU/Dice и precision/recall.
Все 94 пустые каталожные маски сохранены (80 train, 14 test); их нельзя трактовать
как проверенное отсутствие всех визуально различимых кратеров.

## Очистка 19 октября 2026

Сохранены `head_training_v2/` (120 WAC сцен: 50 train / 39 val / 31 test),
`crater_seg_100_v1/` (50 Moon + 50 Mars), `orbital_300_v4/` (300 сцен с масками
и отдельными картами высот), ручная разметка `crater_annotations_v5/` и исходные
аннотации/доказательства замены изображения. Изображения, маски, DEM, manifests
и identities этих коллекций не изменены.

Удалены снимки старой v3, версии разметки v1–v4, черновая crater-выборка,
копии CVAT-изображений и старый generated QA-набор. Небольшие исходные v3
манифесты и единственный отличающийся снимок сохранены в
`provenance/orbital_300_v3/`: они нужны для hash-bound проверки пересечений и
происхождения исправленного снимка. API-каталог использует v4.
Исторические контексты v1/v2, панели buried-crater audit и дополнительные ручные
маски сгруппированы в `provenance/`, без удаления их содержимого.

CVAT ZIP распакованы в `crater_annotations_v5/exports/<planet_split>/annotations.xml`.
Проверка сравнивает XML с исходным запечатанным SHA-256; прежние hashes ZIP
остаются историческими записями, но отсутствующие ZIP-байты не объявляются
проверенными. 26 исходных Mars CSV сохранены по hashes из v5 samples в
`crater_source_annotations_v1/mars_crater_metadata/`. Полный publisher ZIP больше
не хранится: историческая пересборка через `SourceAnnotations` потребует его
явного восстановления, а не автоматического скачивания. Это не влияет на
обучение голов, проверку сохранённой v5 или текущие crater-коллекции.

Удалённое отправлено в Корзину Windows. Журнал — `outputs/cleanup_20261019/`.
Среды, веса, checkpoints, результаты и защищённый архив не очищались.

## Additional crater-only collection

`crater_seg_100_v1/` is a separate 50 Moon + 50 Mars publisher-test collection
requested on 18 October 2026. It has binary crater masks, SHA-256 manifests and
overlap audits against the old v3/v4 images. Lunar native VIS5 is retained; Mars
does not provide VIS5. Shared Mars parent mosaics are documented, not presented
as new independent regions. It does not replace the original 300-image tracks
or automatically become compatible with existing model deployments. See
[`crater_seg_100_v1.md`](../docs/research/crater_seg_100_v1.md).
Native Mars mosaic strips and four empty publisher masks are documented in
`crater_seg_100_v1/visual_qa.json`; image/mask identity verification is not a
claim that every image is artifact-free or every visible crater is annotated.

The UI button **Подключить crater_seg_100_v1** / API
`POST /datasets/crater-segmentation/register` now connects the existing local
collection as separate Moon/Mars test datasets for unchanged frozen heads.
The external clean-input protocol removes the old deployment image allowlist:
CLIP takes RGB on both planets; the current IBM VIS5 encoder can use the lunar
part only. No channel replication, training, downloads or source edits occur.
See [`API guide`](../src/api/README.md) for restart/new-run instructions.

## Original orbital collection

The active collection is [`orbital_300_v4/`](orbital_300_v4/). It has **300 distinct real orbital images**: 100 Moon and 100 Mars evaluation images, plus 50 Moon and 50 Mars train images. Each planet's evaluation set has 50 image/pixel-mask pairs and a separate 50 image/elevation-map pairs. The train images have both a publisher mask and a height array. `audit.json` and six JSONL manifests list exact files, source IDs, original splits, revisions, checksums, label semantics, and display transforms.

Only `mars_seg_018/image.png` changed from v3: a real THEMIS v11 crop was
resampled and image-feature-registered to the original 512×512 annotation grid.
All 799 other data assets, including every mask/height array, are byte-identical.
The v3 image copy and older candidate builds were retired during cleanup;
their required provenance and the current v5 annotations are retained.

| Planet | Segmentation eval | Elevation eval | Train | Segmentation test/val |
|---|---:|---:|---:|---:|
| Mars | 50 | 50 | 50 | 33 / 17 |
| Moon | 50 | 50 | 50 | 17 / 33 |

Question assignment selected on 14 October 2026: Q1–Q4 target each planet's
50 semantic-track images; Q5–Q8 target its separate 50 height-track images.
Four tasks per image, subject to reference/model admission. This does not make
non-crater masks into crater labels or admit routing from height arrays alone.
See `docs/protocols/orbital_study.md`. The prepared `orbital_questions_v1`
remains a partial clean Q5/Q6 pilot, not the full 800-request design.

Q1–Q4 preparation, 15 October 2026: the current separate candidate annotation
build is [`crater_annotations_v5/`](crater_annotations_v5/). It preserves the four
native CVAT exports and recovered publisher objects for all 100 semantic images,
with 400 invariant candidate answers and 0 unknown references out of 400.
The corrected Moon val ZIP removes ten human-confirmed non-craters, confirms
three formerly uncertain objects and refines one truncated bbox. The human
confirmed the other three exports and three specifically queried rim flags;
the previously obscured region in mars_seg_018 is visible in the registered replacement.
All 26 Mars crater masks are exactly reproduced from their 1,101 catalog IDs
using a frozen reverse-engineered raster recipe selected on source-val and
checked without refitting on source-test. This verifies raster alignment, not
physical map scale or visual rim correctness. **No scoring targets are accepted
yet**; explicit source-instance/identity second review remains required. Versions v1–v4 are historical
snapshots, not the current candidate build. See
[`crater_annotations_v1.md`](../docs/protocols/crater_annotations_v1.md) for the
format/converter protocol; its version is separate from the data-folder revision.

The earlier retain-band decision in `source_artifacts_v1.json` is historical,
superseded by the explicit image-only replacement request. The different mosaic
versions are not a controlled single-corruption pair. Replacement provenance is
in `orbital_300_v4/image_replacement.json`; its separate image-license review is
pending. The old Colab ZIP distribution was removed; current source images and
candidate references remain available locally.

All 100 segmentation images retain a publisher **val/test** split. The 100 height images also retain their source val/test strata. Train images are excluded from evaluation. The Mars height images are at least three MMLSv2 grid tiles (Manhattan distance) from the selected train tiles; this is a local adjacency check, not a proof of geographic independence across the parent mosaic. Lunar WAC train and evaluation images have disjoint product IDs. The source region overlap for Mars ConeQuest val/test and any model pretraining overlap remain open audit limitations.

## Segmentation classes

Masks use **source-local class IDs**, not a common cross-dataset taxonomy. Do not score unlike masks as one segmentation task.

| Planet / source | Selected | Pixel values and meaning |
|---|---:|---|
| Mars [Mars-Bench crater type](https://huggingface.co/datasets/Mirali33/mb-crater_multi_seg) | 12 | 0 Background, 1 Other crater, 2 Layered, 3 Buried, 4 Secondary. All four foreground types occur in all 12 selected masks. |
| Mars [Mars-Bench binary crater](https://huggingface.co/datasets/Mirali33/mb-crater_binary_seg) | 14 | 0 Background, 1 Crater. |
| Mars [ConeQuest](https://huggingface.co/datasets/Mirali33/mb-conequest_seg) | 20 | 0 Background, 1 Cone. |
| Mars [Mars-Bench boulder](https://huggingface.co/datasets/Mirali33/mb-boulder_seg) | 4 | 0 Background, 1 Boulder. |
| Moon [LROC NAC debris and voids](https://huggingface.co/datasets/F1nnSBK/lunar-debris-and-voids) | 9 | 0 Background, 1 Pit, 2 Stone, 3 Crater. Pits occur in 9 masks; stones and craters in 6 each. Source COCO polygons were rasterized without inventing labels. |
| Moon [SomBench IMP](https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-IMP-Segmentation) | 21 | 0 Background, 1 Irregular Mare Patch. |
| Moon [SomBench WAC crater](https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-WAC-Crater-Detection) | 20 | 0 Background, 1 Crater. Publisher COCO crater polygons were rasterized on the WAC grid. |

This diversity spans distinct instruments and annotation tasks; class names alone do not establish a shared pixel taxonomy. WAC crater polygons derive from a catalog and should be interpreted as catalog coverage, not exhaustive small-crater labels. The NAC pit/stone/crater annotation uses MobileSAM-assisted polygons with manual fallback, as described by its dataset card. Rock and crater masks do not measure traversability.

## Height and image format

- Mars elevation: 50 [MMLSv2](https://huggingface.co/datasets/MarsLS/MMLSv2) DEM arrays, `height_map.npy`, source-scaled `[0,1]`; physical units are not provided in this release. Image size is 128×128.
- Moon elevation: 50 [SomBench WAC](https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-WAC-Crater-Detection) DTM arrays in meters, bilinearly aligned from the publisher's 60 m DTM to the 512×512 WAC image grid (100 m/pixel). Each has `height_valid_mask.npy`.
- The 50 train images per planet come from the respective MMLSv2 and WAC train partitions and carry source masks plus elevation arrays. The lunar WAC crater masks now have a task-matched encoder-probe target defined below; no model head has been trained. The Mars train masks still differ from the detailed Mars segmentation evaluation tasks.
- Multi-band MMLSv2 and WAC rasters were rendered to three-channel PNG using visible bands 0–2 and a per-tile 2nd–98th percentile stretch. The manifest preserves the rendering method and SHA-256 of the source raster. These PNGs are inference images. Masks and height arrays are targets or controlled E4 inputs only; do not leak them into ordinary image-only inference.

## Derived WAC encoder-probe targets

`python -m linear_head_approach.prepare` builds a separate lunar WAC crater probe
from the curated set. In addition to the 20 WAC segmentation examples, the
50 WAC height-track images have publisher COCO crater polygons in the pinned
SomBench val/test indices. Their binary masks are rasterized into
`encoder_probe_wac_v1/derived_masks/` without using elevation as an input.
The probe has 50 train, 39 val and 31 test images; its input and target
manifests are separated in `linear_head_approach/outputs/encoder_probe_wac_v1/`. The original
`orbital_300_v3/` collection is unchanged. See
[`linear_head_approach/protocol.md`](../linear_head_approach/protocol.md).

## Layout and verification

`orbital_300_v4/{mars,moon}/{segmentation,height,train}/001/` contains assets. Each planet has `{segmentation,height,train}_manifest.jsonl`; paths in these manifests are relative to the repository root. Height arrays are NumPy `.npy` files. `audit.json` records aggregate counts. To verify every path, checksum, shape, mask value, split, and group separation:

```powershell
& .\.venv-benchmark\Scripts\python.exe scripts\verify_orbital_300.py --collection data_orbital/orbital_300_v4
```

Source caches and previous selections were removed after verification. [`scripts/build_orbital_300.py`](../scripts/build_orbital_300.py) records the selection logic; rerunning it requires reacquiring the pinned source subsets first. No model weights, paid APIs, training, or GPU jobs were used. The protected `data_rover/` archive was not accessed.
