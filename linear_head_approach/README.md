# Frozen encoder heads · v2

Учебный и воспроизводимый **supporting experiment**: исходный IBM foundation
backbone и фактический visual tower SpaceLLaVA заморожены; для каждого отдельно
обучаются головы одной семьи. Не запускает языковую модель и не заменяет
основной full-system benchmark.

Текущий target: **Windows10 / RTX3060Ti 8GiB / RAM16GiB**, обучение через CLI,
benchmark через UI/FastAPI и отдельный inference-only heads-worker на8003.
Начни с [`gpu_runbook.md`](gpu_runbook.md). Colab notebook/ZIP оставлены как
зафиксированный прежний вариант, их код не распаковывать поверх этой версии.
**Новый Windows GPU smoke ещё не выполнен.**

## Файлы

| Файл | Ответственность |
|---|---|
| `config.toml` | Фиксированный протокол, бюджет, головы, seeds и val-пороги |
| `__main__.py` | Независимые CLI-стадии, ничего не обучает при импорте |
| `common.py` | SHA-256, безопасные пути, атомарные записи, identity/resume, логи |
| `data.py` | Контракт входов/targets, аудит splits, IoU/Dice и scene bootstrap |
| `encoders.py` | Pinned assets, frozen CLIP/IBM, двухкадровый smoke, incremental cache |
| `heads.py` | Линейная голова и компактная spatial head; BCE + Dice |
| `training.py` | Train-only нормализация, AdamW, validation selection, checkpoints, test |
| `reporting.py` | Кривые, фиксированные val-примеры, парное сравнение по сценам |
| `local.py` | Doctor, pinned source setup, последовательный CLI train coordinator |
| `deployment.py` | Экспорт обученной головы и JSON для импорта в UI, без test scoring |
| `compatibility.py`, `compatibility_revisions.json` | Точечная проверка исторического кода для export/inference, без изменения исходных identities/checkpoints |
| `setup_windows.ps1` | Явная установка отдельного Windows-окружения, без весов/обучения |
| `bundle.py` | Проверенная миграция прежней выборки и переносимый ZIP, без весов |
| `build_notebook.py` | Генерация учебного notebook через nbformat |
| `tests/` | CPU-контракты + torch smoke/resume-тесты при наличии torch |

Данные находятся **вне кода**, в `data_orbital/head_training_v2/`. Новое окружение:
`.venv-heads3060`, assets: `weights/encoder_heads`, runs: `outputs/heads_rtx3060ti_v1`.
Windows requirements: `requirements/encoder-heads-windows.txt`. Признаки,
логи и обученные головы — вне кода, в `outputs/` или Drive. Прежняя версия
целиком сохранена в `outputs/archives/linear_head_approach_before_neural_v2_20261018/`.
Ни один файл защищённого архива не использовался.

## Важные границы

- Задача v2 — binary **catalog crater polygon segmentation** на 120 WAC tiles,
  не instance counting, VLM QA, traversability или route safety.
- Сохранены 50 train / 39 val / 31 test и исходные scene IDs. Train: 36 групп.
- Высоты не являются входами, не предсказываются и в ZIP не включены.
- Вне publisher mask может находиться визуальный кратер, отсутствующий в каталоге.
  Это ограничивает интерпретацию false positives. Нужен отдельный аудит полноты.
- IBM получает native VIS5, CLIP — rendered RGB той же сцены: **информация разная**.
  Не заполняем недостающие каналы копированием RGB и не называем это fair RGB-only.
- Source-heldout **не доказывает** отсутствие в pretraining. Часть исходных сцен
  уже рассматривалась в прежних экспериментах; это не новый нетронутый test set.
- Параметров у голов разное количество из-за C=768/1024; семейство и бюджет одинаковы.
- Tensor identity SpaceLLaVA tower vs base CLIP пока не проверена. Вывод об эффекте
  специализации SpaceLLaVA только по этому опыту недопустим.

Подробности: [`protocol.md`](protocol.md), запуск: [`gpu_runbook.md`](gpu_runbook.md).

## Статус проверки

Локально разрешены только подготовка и CPU software checks. Реальные признаки,
loss curves, качество и T4 ETA появятся после явного запуска notebook. ZIP не
содержит pretrained weights или заранее созданных «результатов обучения».
