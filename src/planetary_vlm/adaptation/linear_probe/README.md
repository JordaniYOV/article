# A — Linear probe

`model.py`: **только Conv2d(C,1,1)** и фиксированное bilinear увеличение logits.
Нет hidden layers, GELU, spatial convolutions или trainable upsampler. Исходный
Lunar-FM заморожен; используются только признаки последнего слоя из общего cache.
Подход оценивает доступность бинарной информации линейному классификатору, а не
языковые/геологические рассуждения VLM. Старые trained lunar heads не используются.

`pipeline.py` / `__main__.py`: свой entrypoint в общий train/eval pipeline.
`config.toml`: method, seeds13/37/73, head lr1e-3. Общие настройки разрешаются в
`common/config.py` и целиком сохраняются в identity.

```powershell
python -m planetary_vlm.adaptation.linear_probe plan
python -m planetary_vlm.adaptation.linear_probe train
python -m planetary_vlm.adaptation.linear_probe train --resume
python -m planetary_vlm.adaptation.linear_probe evaluate --seed 13 --split test --allow-test
```

Стадии отдельно: `prepare`, `smoke`, `extract`, `fit --seed 13`, `evaluate`,
`predict --image ... --prediction ...`, `summarize --split val`.
Train запускает все3 seeds и val, но **не test**. Реальный GPU smoke обязателен
перед extraction/fit; совместимость пока не проверена. Логи в консоли и outputs.
Окружение/assets/общий протокол: [общая инструкция](../README.md).
