# Локальные данные

Каталоги создаются конвертерами по необходимости; содержимое не входит в git:

- `raw/<dataset>/` — оригинальные архивы/изображения/labels без изменений.
- `interim/<dataset>/` — распаковка и нормализация исходных структур.
- `processed/<dataset>/<version>/` — подготовленные изображения/варианты.
- `manifests/<version>/` — samples/questions и **отдельно** ground truth.

Каждая запись сохраняет source ID, dataset version, original split, group ID,
provenance и license. Не пересоздавать held-out split случайно по отдельным кадрам.
Пути в переносимых manifests задаются относительно объявленного data root.
Manifest и вопросы сначала версионируются как спецификация; пользовательские
данные и сами manifests не публикуются автоматически.
