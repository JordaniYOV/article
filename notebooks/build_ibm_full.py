"""CPU-only full100 v2 export; update the current notebook with a byte backup.

Original images, source splits, candidate statuses and model settings stay fixed.
Local eval files are deliberately excluded from the inference ZIP.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from planetary_vlm.colab import digest, export_bundle
from planetary_vlm.datasets.crater_reference import TASKS, VERSION
from planetary_vlm.io import read_jsonl, write_json, write_jsonl
from build_ibm_colab import notebook as base_notebook
from build_ibm_pilot import FREEZE_SHA256

VERSION_ID = "ibm_full100_v2"
LOCK = "requirements/ibm-val14-colab-lock.txt"


def notebook(checksum):
    import nbformat as nb
    template = base_notebook(checksum)
    original = {c.id:c.source for c in template.cells if c.cell_type=="code"}
    cells=[]
    def md(text):
        cells.append(nb.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(id,text):
        cell=nb.v4.new_code_cell(textwrap.dedent(text).strip()+"\n")
        cell.id=id
        cells.append(cell)
    md("""
        # IBM · полный clean-прогон · 100 снимков · v2
        ## Цель
        50 Moon + 50 Mars → **100 forwards детектора, 400 ответов Q1–Q4**.
        Только исходные изображения, без искажений, depth, обучения или квантования.
        Опубликованный NAC Faster R-CNN head → фиксированный перевод рамок в ответы.
        FP32, score ≥0.5 и `crater-reference-v1` не менялись после val-пилота.

        **Загрузить новый `ibm_full100_v2.zip` в `MyDrive/planetary_benchmark/`.**
        Выполнять сверху вниз на GPU T4. На новой VM установить `DOWNLOAD_WEIGHTS=True`.
        После успешного smoke включить `RUN_BENCHMARK=True` только в разделе 7.
        Старый ZIP пилота сюда не подходит. Текущий notebook обновлён с val14 на full100;
        выполненная пилотная версия сохранена локально в архиве подготовки.

        ### Что уже проверено и что остаётся ограничением
        Пилот 14 сцен реально выполнен на T4 в прежнем зафиксированном окружении.
        **Этот полный GPU-прогон ещё не выполнен.** Низкая точность пилота не блокирует
        исполнение. Разметка остаётся кандидатной и требует независимой проверки;
        результаты не становятся финальными научными метриками автоматически.
        Исходные val/test сохранены; в 100 входят старые smoke и val-пилот.
        Всего 70 исходных групп; три группы встречаются и в source-val, и в source-test.
        Эти пересечения отмечаются локально, а не объявляются чистым holdout.
        Ни эталоны, ни маски, ни рамки GT в ZIP или вход модели не передаются.
        Некоторые исходные `.png` имеют TIFF-содержимое: байты сохранены, формат
        определяется декодером, а не расширением. Нормализация и тензоры не меняются.
    """)
    md("""
        ## Настройка
        ### 1. Параметры и Drive
        Не повторять эту ячейку перед run: она создаёт SESSION_ID.
        При resume оставить тот же RUN_ID и ZIP, поставить RESUME=True,
        выполнить setup/assets и новый smoke. Старые run IDs пилота не использовать.
    """)
    parameters=original["parameters"].replace("ibm_craters_v1.zip",VERSION_ID+".zip")
    parameters=parameters.replace("planetary_ibm_craters_v1","planetary_"+VERSION_ID)
    parameters=parameters.replace("venv_ibm_craters_v1","venv_"+VERSION_ID)
    parameters=parameters.replace("'ibm_craters_v1_clean'",repr(VERSION_ID+"_clean"))
    parameters=parameters.replace("RUN_BENCHMARK = False\n","")
    parameters=parameters.replace("PYTHON = str(VENV/'bin/python')", """# На той же VM повторно используем прежний IBM venv; не SpaceLLaVA.
for previous_env in (Path('/content/venv_ibm_val14_v1'), Path('/content/venv_ibm_craters_v1')):
    if (previous_env/'bin/python').is_file():
        VENV = previous_env
        break
PYTHON = str(VENV/'bin/python')""")
    code("parameters",parameters+"\nREUSE_OLD_ASSETS = True\n")
    md("### 2. Хеши ZIP и распаковка\nСтарые workspaces и прогоны не перезаписываются. Повтор этой ячейки допустим только для того же ZIP.")
    code("unpack",original["unpack"].replace("requirements/ibm-crater-colab.txt",LOCK))
    md("""
        ### 3. Изолированное окружение Python 3.12
        Те же 156 закреплённых версий, что в успешном T4 val-пилоте.
        Если прежний IBM venv существует, используется он; иначе создаётся новый.
        Kernel Colab не импортирует Torch. На новой VM нужны место для библиотек,
        cache и ≈4.18 GB весов; желательно ≥25 GiB свободно перед установкой.
    """)
    code("install",original["install"].replace("requirements/ibm-crater-colab.txt",LOCK))
    md("### 4. Проверка состава, окружения и CUDA\nОжидается 100 снимков / 400 ответов. Логи одновременно в консоли и Drive. Проверяется CUDA NMS.")
    commands=original["commands"].replace("planetary_vlm.colab_ibm'","planetary_vlm.colab_ibm_full'")
    commands += "\nrecorded_env = json.loads((WORKSPACE/'ibm_environment.json').read_text())\n"
    commands += f"assert recorded_env['pip_freeze_sha256'] == {FREEZE_SHA256!r}, 'Environment differs from frozen pilot; do not run/resume'\n"
    code("commands",commands)
    md("""
        ### 5. Повторное использование или явная загрузка весов
        На той же VM берём проверенные assets прежнего val14 или первого IBM workspace
        через ссылки — без второй копии. Хеши и ревизии проверяются повторно.
        Если assets нет и DOWNLOAD_WEIGHTS=False, команда остановится без скачивания.
    """)
    code("assets","""
        if REUSE_OLD_ASSETS:
            for previous in (Path('/content/planetary_ibm_val14_v1'), Path('/content/planetary_ibm_craters_v1')):
                if previous.is_dir():
                    for relative in ('external/IBM-Lunar', 'weights/IBM-Craters', 'weights/IBM-Lunar'):
                        source, target = previous/relative, WORKSPACE/relative
                        if source.is_dir() and not target.exists() and not target.is_symlink():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.symlink_to(source.resolve(), target_is_directory=True)
                            print('Reused asset directory:', relative)
        step('assets', *(['--download'] if DOWNLOAD_WEIGHTS else []))
    """)
    md("""
        ## Проверки
        ### 6. Технический smoke в текущей сессии
        Два прежних lunar val-контроля, отдельный smoke ZIP. Это gate исполнения,
        а не требование высокой точности. Старые ложные находки не надо исправлять
        изменением порога или разметки. После smoke модель выгружается.
        Оба изображения также входят в последующий полный набор, но отдельные smoke
        ответы не добавляются к его 400 ответам и не становятся независимыми тестами.
    """)
    code("smoke",original["smoke"])
    md("""
        ## Запуск
        ### 7. Полный clean-прогон
        Поставить RUN_BENCHMARK=True **здесь**, не повторяя parameters.
        Одна загрузка IBM, по одному forward на каждый из 100 снимков.
        Прогресс сохраняется в Drive после каждого изображения. При обрыве тот же
        RUN_ID/ZIP + RESUME=True, setup, новый smoke и эта ячейка: готовые кадры пропускаются.
        Результаты пилота сюда не импортируются — full100 имеет новую воспроизводимую идентичность.
    """)
    code("run","""
        RUN_BENCHMARK = False  # True после технически успешного smoke
        if RUN_BENCHMARK:
            try:
                step('run', '--persist', PERSIST, '--run-id', RUN_ID, '--session-id', SESSION_ID,
                     *(['--resume'] if RESUME else []))
            finally:
                print('Result ZIP (complete or partial):', PERSIST/(RUN_ID+'_results.zip'))
                print('Drive checkpoints:', PERSIST/'runs/IBM-Craters'/RUN_ID)
        else:
            print('Set RUN_BENCHMARK=True in this cell after smoke; do not rerun parameters.')
    """)
    md("### 8. Результаты и полнота\nПроверка 100 снимков / 400 уникальных ответов; отображаем только несколько overlays, не все 100. Это не вычисляет accuracy без эталонов.")
    code("review","""
        result_folder = WORKSPACE/'outputs/IBM-Craters'/RUN_ID
        if (result_folder/'summary.json').is_file():
            summary = json.loads((result_folder/'summary.json').read_text())
            predictions = [json.loads(line) for line in (result_folder/'predictions.jsonl').read_text().splitlines() if line.strip()]
            print(summary)
            assert summary['success'] is True and summary['images'] == 100 and summary['answers'] == 400
            assert len(predictions) == len({row['request_id'] for row in predictions}) == 400
            from IPython.display import display
            from PIL import Image
            for path in sorted((result_folder/'overlays').glob('*.png'))[:4]:
                record = json.loads((result_folder/'detections'/(path.stem+'.json')).read_text())
                print(path.stem, record['source_split'], record['source_dataset'], record['answers'])
                display(Image.open(path))
        else:
            print('No completed full run. Keep RUN_ID and ZIP; use logs/checkpoints for resume.')
        print('Send:', PERSIST/(RUN_ID+'_results.zip'))
    """)
    md("""
        ## Следующий шаг
        Пришли `ibm_full100_v2_clean_results.zip` из MyDrive/planetary_benchmark.
        При сбое — частичный ZIP и соответствующий log. Не менять score threshold,
        вопросы или разметку по ошибкам test. Оценка выполняется отдельно локальным
        eval-kit, не загружаемым в Colab; кандидатные ответы не являются принятым GT.
        Новый full100 notebook проверяется CPU-тестами, но полный GPU-путь ещё не выполнен.
    """)
    document=nb.v4.new_notebook(cells=cells,metadata=template.metadata)
    nb.validate(document)
    return document


def main():
    import nbformat as nb
    output=ROOT/"outputs/colab"/(VERSION_ID+".zip")
    refs=ROOT/"outputs/colab"/(VERSION_ID+"_eval")
    checks=ROOT/"outputs/colab"/(VERSION_ID+"_checks")
    destination=ROOT/"notebooks/ibm_val14_colab.ipynb"
    if output.exists() or refs.exists() or checks.exists():
        raise FileExistsError("Full100 artifacts already exist; use a new version, never silently replace")
    locked=[r for r in (ROOT/LOCK).read_text().splitlines() if r and not r.startswith("#")]
    if len(locked)!=156 or digest(sorted(locked))!=FREEZE_SHA256:
        raise ValueError("Frozen IBM environment lock changed")
    ann=ROOT/"data_orbital/crater_annotations_v5"
    metadata=json.loads((ann/"metadata.json").read_text(encoding="utf8"))
    for name,expected in metadata["files_sha256"].items():
        if sha256((ann/name).read_bytes()).hexdigest()!=expected:
            raise ValueError("Source annotation build changed: "+name)
    samples=read_jsonl(ann/"samples.jsonl")
    questions={r["request_id"]:r for r in read_jsonl(ann/"questions.jsonl")}
    if (len(samples)!=100 or len({r["sample_id"] for r in samples})!=100
        or Counter(r["planet"] for r in samples)!={"moon":50,"mars":50}
        or any(r["source_split"] not in {"val","test"} or not r["evaluation_eligible"] for r in samples)):
        raise ValueError("Expected original 50 Moon + 50 Mars semantic evaluation images")
    samples.sort(key=lambda r:r["sample_id"])
    for row in samples:
        for task in TASKS:
            q=questions[row["sample_id"]+"__"+task+"__clean"]
            if (q["candidate_answer"] is None or q["reference_status"]!="CANDIDATE_REFERENCE"
                or q["annotation_fingerprint"]!=row["annotation_fingerprint"]):
                raise ValueError("Expected immutable known candidate references, not new GT admission")
    checks.mkdir()
    if destination.exists():
        backup=checks/"previous_ibm_val14_colab.executed.ipynb"
        shutil.copy2(destination,backup)
        write_json(checks/"previous_notebook.json",{"file":backup.name,"sha256":sha256(backup.read_bytes()).hexdigest()})
    result=export_bundle(ROOT,output,collection="data_orbital/orbital_300_v4",
        crater_questions="data_orbital/crater_annotations_v5",conditions=("clean",),
        requirements_profiles=(LOCK,),source_ids=[r["sample_id"] for r in samples])
    checksum=sha256(output.read_bytes()).hexdigest()
    nb.write(notebook(checksum),destination)
    refs.mkdir()
    with zipfile.ZipFile(output) as archive:
        requests={r["request_id"]:r for r in (json.loads(line) for line in archive.read("data_orbital/colab_inputs/requests.jsonl").splitlines())}
    prior=json.loads((ROOT/"outputs/colab/ibm_val14_v1_eval/reference_manifest.json").read_text())
    from planetary_vlm.colab_ibm import SMOKE_IDS
    exposed=set(prior["pilot_sample_ids"])|set(SMOKE_IDS)
    group_key=lambda row:(row["planet"],row["source_dataset"],row["scene_group_id"])
    exposed_groups={group_key(row) for row in samples if row["sample_id"] in exposed}
    grouped=defaultdict(list)
    for row in samples:
        grouped[group_key(row)].append(row)
    mixed_groups={key for key,values in grouped.items() if len({r["source_split"] for r in values})>1}
    eval_requests,targets,candidates=[],[],[]
    for row in samples:
        relative=Path(os.path.relpath(ROOT/row["image"],refs)).as_posix()
        for task in TASKS:
            rid=row["sample_id"]+"__"+task+"__clean"
            q=questions[rid]
            eval_requests.append({**requests[rid],"image_paths":[relative]})
            targets.append({"request_id":rid,"source_sample_id":row["sample_id"],
                "group_id":f"{row['planet']}:{row['source_dataset']}:{row['scene_group_id']}",
                "derivation_version":VERSION,"task_id":task,"condition_id":"clean",
                "answer":q["candidate_answer"],"reference_status":"CANDIDATE_REFERENCE",
                "planet":row["planet"],"source_dataset":row["source_dataset"],"source_split":row["source_split"],
                "prior_ibm_development_scene":row["sample_id"] in exposed,
                "prior_ibm_development_group":group_key(row) in exposed_groups,
                "cross_source_split_group":group_key(row) in mixed_groups})
            candidates.append(q)
    for name,rows in (("requests.jsonl",eval_requests),("candidate_targets.jsonl",targets),("source_candidates.jsonl",candidates)):
        write_jsonl(refs/name,rows)
    seal={"schema":"ibm-full100-local-references-v2","input_bundle_sha256":result["bundle_sha256"],
        "input_zip_sha256":checksum,"sample_ids":[r["sample_id"] for r in samples],
        "image_sha256":{r["sample_id"]:r["image_sha256"] for r in samples},
        "environment_pip_freeze_sha256":FREEZE_SHA256,
        "files_sha256":{name:sha256((refs/name).read_bytes()).hexdigest() for name in
            ("requests.jsonl","candidate_targets.jsonl","source_candidates.jsonl")},"scientific_result":False}
    write_json(refs/"reference_manifest.json",{**seal,"manifest_sha256":digest(seal)})
    write_json(refs/"selection_audit.json",{"scope":"Complete existing 100-image semantic pool, no model-output filtering or changed labels",
        "counts":dict(Counter(r["planet"]+"/"+r["source_split"] for r in samples)),
        "unique_source_groups":len({(r["planet"],r["source_dataset"],r["scene_group_id"]) for r in samples}),
        "cross_source_split_groups":[{"group":list(key),"sample_ids":[r["sample_id"] for r in grouped[key]],
            "source_splits":sorted({r["source_split"] for r in grouped[key]})} for key in sorted(mixed_groups)],
        "prior_development_ids":sorted(exposed),"source_annotation_build_sha256":metadata["bundle_sha256"],
        "records":[{k:r[k] for k in ("sample_id","planet","source_dataset","source_revision","source_split","scene_group_id","image_sha256","annotation_fingerprint")} for r in samples],
        "limitations":["Candidate references pending second review; full pool includes development scenes.",
            "Unique source groups do not prove independence or training holdout."]})
    receipt={**result,"zip_sha256":checksum,"zip_bytes":output.stat().st_size,"notebook":str(destination),
        "notebook_sha256":sha256(destination.read_bytes()).hexdigest(),"local_eval_kit":str(refs),
        "gpu_executed":False,"scientific_result":False}
    write_json(checks/"build.json",receipt)
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
