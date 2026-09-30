# Orbital 300 v3

The active collection is [`orbital_300_v3/`](orbital_300_v3/). It has **300 distinct real orbital images**: 100 Moon and 100 Mars evaluation images, plus 50 Moon and 50 Mars train images. Each planet's evaluation set has 50 image/pixel-mask pairs and a separate 50 image/elevation-map pairs. The train images have both a publisher mask and a height array. `audit.json` and six JSONL manifests list exact files, source IDs, original splits, revisions, checksums, label semantics, and display transforms.

| Planet | Segmentation eval | Elevation eval | Train | Segmentation test/val |
|---|---:|---:|---:|---:|
| Mars | 50 | 50 | 50 | 33 / 17 |
| Moon | 50 | 50 | 50 | 17 / 33 |

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
- The 50 train images per planet come from the respective MMLSv2 and WAC train partitions and carry source masks plus elevation arrays. Their mask labels differ from the detailed segmentation evaluation tasks; training an evaluator on them is a separate, unperformed step and needs an explicit target definition.
- Multi-band MMLSv2 and WAC rasters were rendered to three-channel PNG using visible bands 0–2 and a per-tile 2nd–98th percentile stretch. The manifest preserves the rendering method and SHA-256 of the source raster. These PNGs are inference images. Masks and height arrays are targets or controlled E4 inputs only; do not leak them into ordinary image-only inference.

## Layout and verification

`orbital_300_v3/{mars,moon}/{segmentation,height,train}/001/` contains assets. Each planet has `{segmentation,height,train}_manifest.jsonl`; paths in these manifests are relative to the repository root. Height arrays are NumPy `.npy` files. `audit.json` records aggregate counts. To verify every path, checksum, shape, mask value, split, and group separation:

```powershell
& .\.venv-benchmark\Scripts\python.exe scripts\verify_orbital_300.py
```

Source caches and previous selections were removed after verification. [`scripts/build_orbital_300.py`](../scripts/build_orbital_300.py) records the selection logic; rerunning it requires reacquiring the pinned source subsets first. No model weights, paid APIs, training, or GPU jobs were used. The protected `data_rover/` archive was not accessed.
