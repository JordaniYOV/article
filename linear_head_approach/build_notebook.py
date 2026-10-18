"""Generate the educational Colab notebook with nbformat (no GPU execution)."""
from pathlib import Path
import textwrap

import nbformat as nbf


def build(path):
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""
    # Frozen encoder → trainable crater head · v2

    ## Goal

    Учимся обучать **только пространственную голову**, сравнивая признаки двух
    замороженных энкодеров. LLM, проектор SpaceLLaVA и готовая IBM crater head
    здесь не загружаются. Это отдельный supporting experiment, не VLM Q1–Q4.

    Первый трек: лунные WAC **каталожные полигоны кратеров**. 120 изображений:
    50 train (36 source groups), 39 val, 31 test. Все исходные splits сохранены.
    Ноль вне маски означает «не входит в каталог», а не «кратеров точно нет».

    **Статус:** notebook подготовлен; запуск новых адаптеров и обучения на T4
    ещё не проверен. Сначала двухкадровый encoder smoke и synthetic head smoke.
    Никаких метрик качества до реального запуска здесь нет.

    ### Key assumptions

    - IBM: исходный foundation checkpoint, native WAC VIS из 5 каналов.
    - SpaceLLaVA: фактические CLIP tower tensors из её shard, rendered RGB.
    - Это **не equal-information** сравнение; эффект нельзя приписывать только
      planetary specialization. Совпадение SpaceLLaVA tower с base CLIP ещё не проверено.
    - Нет доказанной независимости от pretraining. Source-heldout ≠ unseen by model.
    - Никакие Mars train masks не смешиваются с лунными crater targets.
    """)
    md("""
    ## Setup

    ### 1. Параметры и предохранители

    Colab → Change runtime type → T4 GPU. Для первого запуска оставь `ENCODER='ibm'`.
    После него запусти тот же notebook с `ENCODER='spacellava'`: GPU-модели идут
    последовательно, в отдельных процессах. Для скачивания весов и обучения
    нужно явно включить соответствующие флаги. Для resume **не меняй** RUN_ID,
    bundle, конфигурацию, зависимости или папку сохранённых результатов.
    """)
    code("""
    from pathlib import Path
    import os, sys, json, hashlib, shutil, subprocess, uuid, zipfile, stat

    ENCODER = 'ibm'                 # затем 'spacellava'
    RUN_ID = 'frozen_crater_heads_v2'
    DOWNLOAD_WEIGHTS = False        # True — явное разрешение скачивания pinned весов
    RUN_EXTRACT = False             # True — smoke + полный кэш признаков
    RUN_TRAIN = False               # True — обе головы × 3 seed, после smoke
    OPEN_TEST = False               # True только после фиксации протокола
    RESUME = True                   # повторить незавершённую эпоху из last.pt
    EXPECTED_BUNDLE_SHA256 = ''      # опционально скопировать из .json рядом с ZIP
    WORKSPACE = Path('/content/planetary_encoder_heads_v2')
    VENV = Path('/content/venv_encoder_heads_v2')
    SESSION_ID = uuid.uuid4().hex    # не переиспользовать smoke прошлой Colab-сессии
    assert ENCODER in ('ibm', 'spacellava')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    print('Free /content GiB:', round(shutil.disk_usage('/content').free / 2**30, 1))
    print('Plan for >= 15–20 GiB free with environment + assets; actual size depends on cached wheels.')
    """)
    md("""
    ### 2. Drive и загрузка ZIP

    Загрузи `encoder_heads_v2_colab.zip` **и** одноимённый `.json` с SHA-256,
    либо укажи путь к уже сохранённым файлам на Drive. В ZIP есть только
    выбранные 120 изображений, native TIFF, маски, код и notebook; весов нет.
    """)
    code("""
    from google.colab import drive, files
    drive.mount('/content/drive')
    PERSIST = Path('/content/drive/MyDrive/planetary_benchmark/encoder_heads_v2')
    PERSIST.mkdir(parents=True, exist_ok=True)
    BUNDLE = Path('/content/encoder_heads_v2_colab.zip')
    if not BUNDLE.is_file():
        print('Upload the ZIP and its JSON sidecar:')
        uploaded = files.upload()
        for name, payload in uploaded.items():
            assert Path(name).name == name
            (Path('/content') / name).write_bytes(payload)
        del uploaded
    SIDECAR = BUNDLE.with_suffix('.json')
    if not EXPECTED_BUNDLE_SHA256:
        assert SIDECAR.is_file(), 'Upload the JSON sidecar or set EXPECTED_BUNDLE_SHA256.'
        EXPECTED_BUNDLE_SHA256 = json.loads(SIDECAR.read_text())['sha256']
    ASSETS = PERSIST / 'assets'
    EXPERIMENT = PERSIST / RUN_ID
    CACHE = EXPERIMENT / 'features' / ENCODER
    RUNS = EXPERIMENT / 'runs'
    """)
    md("""
    ### 3. Проверка целостности и распаковка

    Сначала проверяются SHA-256 всего ZIP и всех файлов, затем безопасные пути.
    Изменённый workspace не перезаписывается. Это также обнаружит повреждённый
    снимок или маску до начала дорогого вычисления.
    """)
    code("""
    def file_sha(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    assert file_sha(BUNDLE) == EXPECTED_BUNDLE_SHA256, 'ZIP SHA-256 mismatch'
    with zipfile.ZipFile(BUNDLE) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP entries'
        manifest = json.loads(archive.read('bundle.json'))
        assert set(names) == {'bundle.json', *manifest['files']}
        for item in archive.infolist():
            name = Path(item.filename)
            assert not name.is_absolute() and '..' not in name.parts
            assert chr(92) not in item.filename and ':' not in item.filename
            assert not stat.S_ISLNK(item.external_attr >> 16)
        for name, expected in manifest['files'].items():
            payload = archive.read(name)
            assert hashlib.sha256(payload).hexdigest() == expected, name
            target = WORKSPACE / name
            assert not any(p.is_symlink() for p in [target, *target.parents]), 'Symlink target'
            if target.exists():
                assert file_sha(target) == expected, f'Workspace differs: {target}'
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        saved_manifest = WORKSPACE / 'bundle.json'
        if saved_manifest.exists():
            assert json.loads(saved_manifest.read_text()) == manifest
        else:
            saved_manifest.write_text(json.dumps(manifest, indent=2))
    print('Verified files:', len(manifest['files']), '| weights included:', manifest['weights_included'])
    """)
    md("""
    ### 4. Изолированный Python 3.12

    Не устанавливаем старые зависимости в Python 3.13 kernel. Для обоих
    энкодеров используем отдельный venv с lock-файлом рабочего IBM T4 окружения.
    Сам **новый** CLIP/backbone adapter этим ещё не проверен — ниже нужен smoke.
    Установка может занять несколько минут. Если CUDA несовместима с драйвером,
    остановись на диагностике, не меняй пакеты в середине эксперимента.
    """)
    code("""
    def checked(command, **kwargs):
        subprocess.run([str(x) for x in command], check=True, **kwargs)

    PYTHON = VENV / 'bin/python'
    if not PYTHON.is_file():
        checked([sys.executable, '-m', 'pip', 'install', 'uv'])
        checked([sys.executable, '-m', 'uv', 'venv', '--python', '3.12.14', str(VENV)])
    checked([PYTHON, '--version'])
    checked([PYTHON, '-c', 'import sys; assert sys.version_info[:3] == (3,12,14), sys.version'])
    checked([PYTHON, '-m', 'ensurepip', '--upgrade'])
    profile = WORKSPACE / 'requirements/encoder-heads-colab.txt'
    assert profile.is_file(), f'Missing requirements: {profile}'
    checked([PYTHON, '-m', 'pip', 'install', '--extra-index-url',
             'https://download.pytorch.org/whl/cu130', '--only-binary=tokenizers', '-r', profile])
    checked([PYTHON, '-m', 'pip', 'check'])
    checked([PYTHON, '-c', 'import torch; print(torch.__version__, torch.version.cuda); '
             'assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'])
    """)
    md("""
    ## Steps

    ### 5. Общий запуск стадий и аудит данных

    Каждая стадия — новый процесс, поэтому GPU-память освобождается после неё.
    Файлы результатов пишутся на Drive. Логи идут и в консоль, и в файл.
    """)
    code("""
    def step(*arguments):
        command = [str(PYTHON), '-u', '-B', '-m', 'linear_head_approach', *map(str, arguments)]
        env = dict(os.environ, PYTHONPATH=str(WORKSPACE), CUBLAS_WORKSPACE_CONFIG=':4096:8')
        logs = EXPERIMENT / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / (str(arguments[0]) + '_' + SESSION_ID[:8] + '.log')
        with log_path.open('a') as log:
            with subprocess.Popen(command, cwd=WORKSPACE, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                try:
                    for line in process.stdout:
                        print(line, end='')
                        log.write(line)
                        log.flush()
                    code = process.wait()
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    raise
        print('Log:', log_path)

    step('audit')
    checked([PYTHON, '-B', '-m', 'unittest', 'discover', '-s', 'linear_head_approach/tests', '-v'], cwd=WORKSPACE)
    """)
    md("""
    ### 6. Веса: только выбранный энкодер

    По умолчанию сеть запрещена на этой стадии. При первом запуске разреши
    `DOWNLOAD_WEIGHTS=True`. Скачаются исходный IBM backbone **или** один shard
    SpaceLLaVA с CLIP-тензорами и конфиги. Не 13B LLM, не task detector.
    Ревизии зафиксированы, hashes записываются; IBM известные hashes проверяются.
    Полные прежние SpaceLLaVA-веса автоматически не переиспользуются без provenance.
    """)
    code("""
    if DOWNLOAD_WEIGHTS:
        step('assets', '--encoder', ENCODER, '--assets', ASSETS, '--allow-download')
    else:
        print('No download requested. Next stage requires previously verified assets:', ASSETS)
    """)
    md("""
    ### 7. Замороженный энкодер → признаки (без масок)

    `requires_grad_(False)` запрещает градиенты; `eval()` отключает training-mode
    поведение; `inference_mode()` не строит граф. В encoder передаётся только
    словарь clean input. Маски, высоты и ответы в этот процесс не попадают.

    Smoke заново извлекает признаки двух разных кадров и проверяет, что они
    различаются, конечны и повторяемы. Потом кэшируется весь набор с сеткой 24×24:
    CLIP native 24×24; IBM native 64×64 → declared bilinear 24×24.
    Из-за этой общей сетки мелкие структуры могут теряться — это ограничение probe.
    В логах появятся **измеренные** время и peak VRAM. При OOM нельзя молча менять
    разрешение только одной модели: нужен новый протокол и новый RUN_ID.
    """)
    code("""
    if RUN_EXTRACT:
        step('extract', '--encoder', ENCODER, '--assets', ASSETS, '--output', CACHE,
             '--session-id', SESSION_ID, '--smoke')
        step('extract', '--encoder', ENCODER, '--assets', ASSETS, '--output', CACHE,
             '--session-id', SESSION_ID)
    else:
        print('Extraction disabled. Set RUN_EXTRACT=True after reviewing setup.')
    """)
    md("""
    ### 8. Что именно обучаем

    **Linear:** Conv1×1(C→1). **Spatial:** Conv1×1(C→64), GroupNorm, GELU,
    Conv3×3(64→64), GroupNorm, GELU, Conv1×1(64→1). Затем bilinear upsample logits.
    Семья голов одинакова, но C=768 у IBM и C=1024 у CLIP → число параметров различно
    и сохраняется в отчёте. Нельзя утверждать, что capacity строго одинаковая.

    Статистики стандартизации и положительный вес BCE считаются **только train**.
    Маски area-downsample до 128×128 для loss (soft coverage). Loss = weighted BCE
    + 0.5·soft Dice. Веса снимков выравнивают вклад source groups.

    Внутри каждой итерации:
    1. Берём feature tensor X и train mask Y, нормализуем X.
    2. `logits = head(X)` → сырые числа до sigmoid.
    3. `loss(logits,Y)` измеряет ошибку.
    4. `loss.backward()` считает производные **только головы**.
    5. `optimizer.step()` меняет её веса (AdamW); gradient clipping ограничивает скачки.

    Encoder в этом процессе отсутствует. Кэш не позволяет end-to-end fine-tuning.
    Train loss не равен качеству на новых изображениях; следим за validation IoU.
    """)
    code("""
    # Показываем реальный код из того же bundle, а не отдельный toy training loop.
    from IPython.display import display, Code
    display(Code(filename=str(WORKSPACE / 'linear_head_approach/heads.py'), language='python'))
    training_source = (WORKSPACE / 'linear_head_approach/training.py').read_text()
    batch_start = training_source.index('        for x, y, indices in loader:')
    batch_end = training_source.index('        validation = predict', batch_start)
    display(Code(training_source[batch_start:batch_end], language='python'))

    # Software smoke: случайные features и синтетическая маска, НЕ научный результат.
    if RUN_TRAIN:
        step('head-smoke', '--output', EXPERIMENT / 'software_checks')
    else:
        print('Training disabled; no optimizer updates requested.')
    """)
    md("""
    ### 9. Обучение, validation и resume

    По умолчанию 2 головы × 3 seed = 6 небольших head runs на один encoder.
    Max 80 эпох, early stopping 12 эпох без улучшения. Порог из фиксированного
    набора 0.2…0.8 и лучшая эпоха выбираются по **scene-macro val IoU**.
    Test не участвует. `best.pt` — выбранная голова, `last.pt` — optimizer,
    scheduler, RNG, история и номер завершённой эпохи. При разрыве повторится
    только незавершённая эпоха; изменение кода/зависимостей/данных блокирует resume.

    Не сравнивай лучший seed одной модели со средним другой. Сохраняем все seeds.
    Это заранее заданный baseline, не гарантия «лучшего возможного» decoder.
    """)
    code("""
    import tomllib
    with (WORKSPACE / 'linear_head_approach/config.toml').open('rb') as stream:
        protocol = tomllib.load(stream)
    if RUN_TRAIN:
        for family in protocol['heads']:
            for seed in protocol['seeds']:
                run = RUNS / ENCODER / family / f'seed_{seed}'
                args = ['fit', '--cache', CACHE, '--output', run, '--family', family, '--seed', seed]
                if RESUME:
                    args.append('--resume')
                step(*args)
                step('evaluate', '--cache', CACHE, '--output', run, '--split', 'val')
        print('Finished training + validation for', ENCODER)
    """)
    md("""
    ## Checks

    ### 10. Посмотреть loss, validation и маски

    Первые три val-примера выбираются по фиксированному порядку, не по качеству.
    Чёрное = фон каталога, белое = crater polygon. Это не bounding boxes и
    не готовые ответы Q1–Q4; объединённая маска не сохраняет instance counts.
    """)
    code("""
    from IPython.display import display, Image as DisplayImage
    example = RUNS / ENCODER / 'spatial' / 'seed_13'
    if (example / 'best.pt').is_file():
        step('preview', '--cache', CACHE, '--output', example)
        display(DisplayImage(filename=str(example / 'learning_curves.png')))
        display(DisplayImage(filename=str(example / 'validation_preview.png')))
        print(json.loads((example / 'fit_summary.json').read_text()))
    else:
        print('No trained head yet. Execute stages 7–9 explicitly.')
    """)
    md("""
    ### 11. Test — открыть только после фиксации протокола

    Включи `OPEN_TEST=True` только после проверки обеих моделей на val.
    На test применяются **уже выбранные** веса и val-порог. Если после test
    менять модель по её результату, это уже exploratory evaluation, не новый holdout.
    Отчёт содержит IoU/Dice, all-background/all-foreground baselines и bootstrap
    по scene groups (не по миллионам пикселей).
    """)
    code("""
    if OPEN_TEST:
        missing = [RUNS / enc / family / f'seed_{seed}' / 'best.pt'
                   for enc in protocol['encoders'] for family in protocol['heads'] for seed in protocol['seeds']
                   if not (RUNS / enc / family / f'seed_{seed}' / 'best.pt').is_file()]
        assert not missing, f'Complete BOTH encoders before opening test: {missing[:2]}'
        for enc in protocol['encoders']:
            for family in protocol['heads']:
                for seed in protocol['seeds']:
                    step('evaluate', '--cache', EXPERIMENT / 'features' / enc,
                         '--output', RUNS / enc / family / f'seed_{seed}', '--split', 'test', '--allow-test')
        step('compare', '--output', RUNS, '--split', 'test')
    else:
        print('Test remains sealed.')
    """)
    md("""
    ## Next Steps

    ### 12. Сохранить результаты

    ZIP результатов включает обученные головы, историю, идентичности окружений,
    логи и отчёты; исключает pretrained weights и feature cache. Кэш остаётся на
    Drive для resume. Не публикуй лучший seed как единственный результат.

    После пилота: проверить feature utility, изучить ошибки каталога, сравнить
    native-input и отдельный equal-information protocol. Новый Mars/terrain/height
    task требует собственного контракта разметки, а не переименования crater masks.
    Полученная маска сама по себе не даёт надёжный маршрут или физическую опасность.
    """)
    code("""
    if RUNS.exists():
        export = PERSIST / f'{RUN_ID}_results_{SESSION_ID[:8]}.zip'
        with zipfile.ZipFile(export, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for base in (RUNS, EXPERIMENT / 'logs', EXPERIMENT / 'software_checks'):
                if base.exists():
                    for path in base.rglob('*'):
                        if path.is_file():
                            archive.write(path, path.relative_to(EXPERIMENT).as_posix())
            for enc in protocol['encoders']:
                for name in ('identity.json', 'smoke.json', 'features.jsonl'):
                    path = EXPERIMENT / 'features' / enc / name
                    if path.is_file():
                        archive.write(path, path.relative_to(EXPERIMENT).as_posix())
            archive.write(WORKSPACE / 'linear_head_approach/config.toml', 'protocol.toml')
            archive.write(WORKSPACE / 'bundle.json', 'bundle.json')
        print('Results:', export, '| MiB:', round(export.stat().st_size/2**20, 1), '| SHA-256:', file_sha(export))
    """)
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "encoder_heads_colab.ipynb"},
        "validation": {"status": "structure_and_offline_checks_only", "gpu_execution": "not_run"}})
    nbf.validate(notebook)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    return path


if __name__ == "__main__":
    print(build(Path(__file__).resolve().parents[1] / "notebooks/encoder_heads_colab.ipynb"))
