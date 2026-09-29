# Native Seleno manifest

`requests.jsonl.example` — шаблон, а не существующая выборка. После получения
официальных файлов укажите реальный каталог tile с inputs.pt и сохраните manifest
под путём из `configs/runs/seleno_native.toml`.

Seleno адаптер получает native modalities, не rover RGB. Free-form вопрос
сохраняет raw response; универсальный closed-form evaluator для него возвращает
unsupported_scoring до появления отдельной рубрики. meta.json содержит ответы
для upstream fact checks; адаптер не читает его.

Bundled demo выбран авторами среди успешно интерпретируемых tiles. Он подходит
для проверки загрузки/генерации, но не для оценки качества на независимом test.
Локальный source, bridge, MAE, tokenizers, legend и Qwen base должны соответствовать
документированным ревизиям. Наличие template не означает проверенную совместимость
с 16 GB VRAM.
