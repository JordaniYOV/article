# D — Frozen SAM baseline

Здесь **нет обучения и трёх training seeds**. Automatic point grid32×32, prompts
не берутся из target masks/bboxes. Из class-agnostic proposals выбираются маски
по заранее объявленным площади, bbox-axis ratio и grid-perimeter circularity;
итог — их union на исходных512². `filter.py` не принимает GT.
Фильтр — геометрическая эвристика, не classifier, rim verifier или доказательство
crater presence. Если его параметры подбираются на val, это val-calibrated pipeline,
а не parameter-free zero-shot. Никакой test-подбор не реализован.

1. Установи отдельное окружение из `requirements/adaptation.txt`. SAM source
   зафиксирован официальным commit `6fdee8f2727f4506cfbbe553e23b895e27956588`.
2. Отдельно получи доверенный официальный ViT-B checkpoint из
   [официального репозитория](https://github.com/facebookresearch/segment-anything).
   Код не скачивает его. Нельзя передавать непроверенный чужой .pth: publisher loader
   использует torch serialization.
3. Создай свой TOML override, например `my_sam_config.toml`:

```toml
method = "sam_baseline"
[sam]
checkpoint = "weights/sam/sam_vit_b_01ec64.pth"
sha256 = "FULL_64_CHARACTER_SHA256_OF_TRUSTED_OFFICIAL_FILE"
```

Проверить локальный hash в PowerShell: `Get-FileHash -Algorithm SHA256 <path>`.
Это проверяет идентичность локального файла, но само по себе не доказывает
официальное происхождение: сначала проверь доверенный источник загрузки.

```powershell
python -m planetary_vlm.adaptation.sam_baseline plan
python -m planetary_vlm.adaptation.sam_baseline smoke --config my_sam_config.toml
python -m planetary_vlm.adaptation.sam_baseline evaluate --config my_sam_config.toml --split val
python -m planetary_vlm.adaptation.sam_baseline evaluate --config my_sam_config.toml --split test --allow-test
```

Вход — исходная intensity-картинка, повторённая в3RGB slots только ради SAM API;
новая спектральная информация не появляется. SAM использует свой штатный image
processor и не имеет того же вычислительного бюджета, что A/B/C. Сохраняются masks,
proposal counts, общие segmentation metrics, checkpoint/runtime hashes и логи.
Гарантий качества, отсутствия космических данных в pretraining и VRAM-fit нет.
[Общая инструкция](../README.md).
