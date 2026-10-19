# B — Frozen encoder + trainable decoder

`model.py`: четыре depth taps [2,5,8,11] → отдельные1×1 projections → фиксированная
пирамида масштабов [4×,2×,1×,0.5×] → top-down sum/refinement → binary logits.
Width64, GroupNorm/GELU. Это lightweight multilayer pyramid decoder, **не полная
копия U-Net или DPT**. Encoder заморожен. Обучаются все параметры нового декодера.

`pipeline.py`: отдельный B entrypoint. `config.toml`: width64 и seeds13/37/73.
Общий default decoder lr3e-4. Frozen feature-cache такой же, как у A, и может
использоваться повторно без нового encoding, если identities совпадают.

```powershell
python -m planetary_vlm.adaptation.frozen_decoder plan
python -m planetary_vlm.adaptation.frozen_decoder train
python -m planetary_vlm.adaptation.frozen_decoder train --resume
python -m planetary_vlm.adaptation.frozen_decoder evaluate --seed 13 --split test --allow-test
```

Тот же smoke/fit/evaluate/resume/predict, что у A. Главный контроль: C импортирует
**именно этот класс** декодера; изменения модели/width требуют нового протокола
для обоих. GPU-память/скорость не проверены. [Общая инструкция](../README.md).
