# C — LoRA + decoder

`model.py` импортирует тот же `PyramidDecoder`, что B. Head initialization с тем же
seed совпадает с B. `pipeline.py` использует **online image batches**, не frozen cache.

В12 encoder blocks заменяется fused attention `qkv`:
`frozen_QKV(x) + (alpha/r) * B(A(x))`. Rank4, alpha8, B starts at zero, dropout0.
Обновляются Q/K/V совместно; embeddings/norms/base projections/MLP заморожены.
Обучаются только LoRA A/B и новый декодер. Non-reentrant checkpointing блоков
не требует градиентов по input pixels. Замороженный stochastic depth/dropout
остаётся eval; адаптеры имеют отдельный train mode.

```powershell
python -m planetary_vlm.adaptation.lora_decoder plan
python -m planetary_vlm.adaptation.lora_decoder train
python -m planetary_vlm.adaptation.lora_decoder fit --seed 13 --resume
python -m planetary_vlm.adaptation.lora_decoder evaluate --seed 13 --split test --allow-test
```

`extract` для C намеренно запрещён. Decoder lr3e-4, LoRA lr1e-4; остальной
бюджет/данные/val selection общие с B. `last.pt` содержит optimizer/scheduler/scaler
и RNG, `best.pt` — trainable weights и зафиксированный val-threshold. Base checkpoint
не дублируется и не изменяется. Нет гарантии, что C превзойдёт B.
Реальный GPU forward/backward/VRAM smoke ещё не выполнен. [Общая инструкция](../README.md).
