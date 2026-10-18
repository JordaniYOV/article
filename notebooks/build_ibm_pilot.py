"""Build a frozen 14-scene val pilot + two controls, without model execution.

Inference ZIP is image-only. Candidate labels stay in a separate local eval kit.
No existing run notebook, source dataset or previous ZIP is overwritten.
"""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from planetary_vlm.colab import digest, export_bundle
from planetary_vlm.colab_ibm import SMOKE_IDS
from planetary_vlm.colab_ibm_pilot import SCHEMA
from planetary_vlm.datasets.crater_reference import TASKS, VERSION
from planetary_vlm.io import read_jsonl, write_json, write_jsonl
from build_ibm_colab import notebook as base_notebook

SELECTION_VERSION = "ibm-val14-source-group-v1"
SEED = 20261017
FREEZE_SHA256 = "bf9c86747575d29a18bb384422c07cebbd0f77bb1990ac1a7bb787834d76ec0b"


def rank(row):
    return sha256(f"{SELECTION_VERSION}:{SEED}:{row['sample_id']}".encode()).hexdigest()


def select(root):
    folder = root/"data_orbital/crater_annotations_v5"
    metadata = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    for name, expected in metadata["files_sha256"].items():
        if sha256((folder/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Changed annotation build: " + name)
    samples = read_jsonl(folder/"samples.jsonl")
    questions = {row["request_id"]: row for row in read_jsonl(folder/"questions.jsonl")}
    sample_index = {row["sample_id"]: row for row in samples}
    group = lambda row: (row["planet"], row["source_dataset"], row["scene_group_id"])
    smoke_groups = {group(sample_index[sid]) for sid in SMOKE_IDS}
    eligible = [row for row in samples if row["source_split"] == "val"
        and row["evaluation_eligible"] is True and row["sample_id"] not in SMOKE_IDS
        and group(row) not in smoke_groups]
    selected = []
    # Predeclared source/reference strata, never IBM-output strata. These labels
    # and per-image strata remain local; they are not exported to inference.
    quotas = (
        ("F1nnSBK/lunar-debris-and-voids", "YES", 1),
        ("F1nnSBK/lunar-debris-and-voids", "NO", 1),
        ("nasa-ibm-ai4science/Sombench-IMP-Segmentation", "YES", 1),
        ("nasa-ibm-ai4science/Sombench-IMP-Segmentation", "NO", 2),
        ("nasa-ibm-ai4science/Sombench-WAC-Crater-Detection", "YES", 2),
    )
    for source, label, count in quotas:
        candidates = sorted((row for row in eligible if row["planet"] == "moon"
            and row["source_dataset"] == source
            and questions[row["sample_id"]+"__crater_presence_v1__clean"]["candidate_answer"] == label), key=rank)
        used = {group(row) for row in selected}
        picked = []
        for row in candidates:
            if group(row) not in used:
                picked.append(row)
                used.add(group(row))
            if len(picked) == count:
                break
        if len(picked) != count:
            raise ValueError("Insufficient distinct val groups for frozen lunar strata")
        selected.extend(picked)
    mars_groups = {}
    for row in eligible:
        if row["planet"] == "mars":
            mars_groups.setdefault(group(row), []).append(row)
    if len(mars_groups) != 7:
        raise ValueError("The frozen val pilot expects exactly seven available Mars groups")
    selected.extend(min(rows, key=rank) for _, rows in sorted(mars_groups.items()))
    selected.sort(key=lambda row: row["sample_id"])
    if len(selected) != 14 or len({group(row) for row in selected}) != 14:
        raise ValueError("Expected 14 distinct source/scene groups")
    for row in selected:
        for task in TASKS:
            question = questions[row["sample_id"]+"__"+task+"__clean"]
            if (question["candidate_answer"] is None or question["reference_status"] != "CANDIDATE_REFERENCE"
                or question["annotation_fingerprint"] != row["annotation_fingerprint"]):
                raise ValueError("Pilot requires known invariant, still-provisional reference answers")
    cohort = {"schema": SCHEMA, "selection_version": SELECTION_VERSION,
        "pilot_sample_ids": [row["sample_id"] for row in selected], "smoke_sample_ids": list(SMOKE_IDS)}
    return selected, questions, cohort, metadata


def notebook(checksum):
    import nbformat as nb
    template = base_notebook(checksum)
    sources = {cell.id: cell.source for cell in template.cells if cell.cell_type == "code"}
    cells = []
    def markdown(text):
        cells.append(nb.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(name, source):
        cell = nb.v4.new_code_cell(textwrap.dedent(source).strip()+"\n")
        cell.id = name
        cells.append(cell)
    markdown("""
        # IBM · clean val-пилот · 14 сцен
        ## Цель
        Проверить перенос готового **NAC crater detector** на 7 Moon + 7 Mars val-сценах.
        Каждая сцена из отдельной исходной группы. Два прежних smoke-контроля запускаются
        отдельно и **не входят** в 14 сцен / 56 ответов пилота. Test/train и искажения исключены.

        Загрузить `ibm_val14_v1.zip` в `MyDrive/planetary_benchmark/`, выбрать GPU T4.
        Выполнять сверху вниз. Для новой VM включить `DOWNLOAD_WEIGHTS=True`.
        На прежней VM веса можно использовать повторно — их хэши всё равно проверяются.
        Результаты старого smoke и выполненный старый notebook не изменяются.

        ### Зафиксированные настройки
        FP32, без квантования, исходные checkpoint/revisions, score ≥0.5 и `crater-reference-v1`.
        Один detector forward на изображение, затем четыре детерминированных ответа из рамок.
        Никаких масок, рамок GT или эталонных ответов в ZIP/входе модели. Отдельная локальная
        оценка по текущим кандидатным ответам возможна после получения результатов.
        Это development-пилот, **не итоговый benchmark** и не проверенный native NAC-протокол.
        Две прежние сцены работали на T4; этот новый пилот ещё не выполнен.
    """)
    markdown("## Настройка\n### 1. Drive и параметры\nНе перезапускай эту ячейку перед pilot: она создаёт новый SESSION_ID. На новой VM smoke обязателен заново.")
    parameters = sources["parameters"].replace("ibm_craters_v1.zip", "ibm_val14_v1.zip")
    parameters = parameters.replace("planetary_ibm_craters_v1", "planetary_ibm_val14_v1")
    parameters = parameters.replace("venv_ibm_craters_v1", "venv_ibm_val14_v1")
    parameters = parameters.replace("'ibm_craters_v1_clean'", "'ibm_val14_v1_clean'")
    parameters = parameters.replace("RUN_BENCHMARK = False", "RUN_PILOT = False")
    code("parameters", parameters+"\nREUSE_OLD_ASSETS = True  # Только из прежнего IBM workspace, без копирования весов\n")
    markdown("### 2. Проверка ZIP и распаковка\nПроверяются SHA-256 и все файлы; существующий workspace не перезаписывается.")
    code("unpack", sources["unpack"].replace("requirements/ibm-crater-colab.txt", "requirements/ibm-val14-colab-lock.txt"))
    markdown("### 3. Отдельное окружение Python 3.12\nТочные 156 версий из успешного IBM T4 smoke, включая Torch/torchvision CUDA 13.0. В kernel Colab Torch не импортируется. Установка и GPU-проверки требуют Colab; здесь они не выполнены.")
    code("install", sources["install"].replace("requirements/ibm-crater-colab.txt", "requirements/ibm-val14-colab-lock.txt"))
    markdown("### 4. Проверка ограниченной выборки и CUDA\nЛоги одновременно в консоли и Drive. verify должен показать 14 pilot + 2 smoke, только val.")
    code("commands", sources["commands"].replace("planetary_vlm.colab_ibm'", "planetary_vlm.colab_ibm_pilot'"))
    markdown("### 5. Веса: повторное использование или явная загрузка\nНа той же VM можно использовать прежние веса/репозиторий через ссылки. Ничего не удаляется. При отсутствии assets и DOWNLOAD_WEIGHTS=False команда остановится без скачивания.")
    code("assets", """
        previous = Path('/content/planetary_ibm_craters_v1')
        if REUSE_OLD_ASSETS and previous.is_dir():
            for relative in ('external/IBM-Lunar', 'weights/IBM-Craters', 'weights/IBM-Lunar'):
                source = previous/relative
                target = WORKSPACE/relative
                if source.is_dir() and not target.exists() and not target.is_symlink():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(source.resolve(), target_is_directory=True)
                    print('Reused asset directory:', relative)
        step('assets', *(['--download'] if DOWNLOAD_WEIGHTS else []))
    """)
    markdown("## Проверки\n### 6. Smoke: два прежних lunar val-контроля\nЭто проверка исполнения, не порог качества. Известные ложные детекции не означают, что smoke технически упал. Результаты этих двух снимков не смешиваются с новым пилотом.")
    code("smoke", sources["smoke"].replace("benchmark", "pilot"))
    markdown("""
        ## Запуск пилота
        ### 7. Включить только 14 новых val-сцен
        Ниже поставить `RUN_PILOT=True` **после успешного исполнения smoke**, не повторяя parameters.
        Будут ровно 14 detector forwards и 56 ответов, не 100 изображений и не test.
        Сохранение в Drive после каждого кадра. При обрыве: сохранить RUN_ID/ZIP, поставить
        RESUME=True в parameters, выполнить setup/assets, новый smoke и затем pilot.
        Изменение зависимостей/ZIP/протокола запрещает resume; нужен новый RUN_ID.
    """)
    code("pilot", """
        RUN_PILOT = False  # Изменить здесь на True после технически успешного smoke
        if RUN_PILOT:
            step('pilot', '--persist', PERSIST, '--run-id', RUN_ID, '--session-id', SESSION_ID,
                 *(['--resume'] if RESUME else []))
        else:
            print('Pilot disabled. Set RUN_PILOT=True here after smoke; do not rerun parameters.')
        print('Pilot result ZIP:', PERSIST/(RUN_ID+'_results.zip'))
    """)
    markdown("### 8. Просмотр результатов\nТолько предсказания, без эталонов. ZIP автоматически сохраняется в корень папки planetary_benchmark; пришли его после завершения.")
    code("review", """
        result_folder = WORKSPACE/'outputs/IBM-Craters'/RUN_ID
        if (result_folder/'summary.json').is_file():
            summary = json.loads((result_folder/'summary.json').read_text())
            print(summary)
            assert summary['images'] == 14 and summary['answers'] == 56
            from IPython.display import display
            from PIL import Image
            for path in sorted((result_folder/'overlays').glob('*.png')):
                record = json.loads((result_folder/'detections'/(path.stem+'.json')).read_text())
                print(path.stem, record['source_dataset'], record['answers'])
                display(Image.open(path))
        else:
            print('No completed pilot yet. Use the step logs if the run was interrupted.')
    """)
    markdown("""
        ## Следующий шаг
        Пришли `ibm_val14_v1_clean_results.zip` из `MyDrive/planetary_benchmark/`.
        При сбое — соответствующий log и частичный ZIP. Не меняй порог/разметку по увиденным ошибкам.
        Локальный eval-kit не загружается в Colab; он позволяет отдельно сравнить все 56 ответов
        с кандидатной разметкой, явно включив предварительную оценку.
    """)
    document = nb.v4.new_notebook(cells=cells, metadata=template.metadata)
    nb.validate(document)
    return document


def main():
    import nbformat as nb
    output = ROOT/"outputs/colab/ibm_val14_v1.zip"
    destination = ROOT/"notebooks/ibm_val14_colab.ipynb"
    references = ROOT/"outputs/colab/ibm_val14_v1_eval"
    if output.exists() or destination.exists() or references.exists():
        raise ValueError("Pilot artifacts already exist; preserve them and use a new version")
    locked = [line for line in (ROOT/"requirements/ibm-val14-colab-lock.txt").read_text().splitlines()
        if line and not line.startswith("#")]
    if len(locked) != 156 or digest(sorted(locked)) != FREEZE_SHA256:
        raise ValueError("Colab lock differs from the recorded successful T4 environment")
    selected, questions, cohort, metadata = select(ROOT)
    sample_ids = cohort["pilot_sample_ids"] + cohort["smoke_sample_ids"]
    result = export_bundle(ROOT, output, collection="data_orbital/orbital_300_v4",
        crater_questions="data_orbital/crater_annotations_v5", conditions=("clean",),
        requirements_profiles=("requirements/ibm-val14-colab-lock.txt",),
        source_ids=sample_ids, inference_cohort=cohort)
    checksum = sha256(output.read_bytes()).hexdigest()
    document = notebook(checksum)
    with destination.open("x", encoding="utf-8") as stream:
        nb.write(document, stream)
    references.mkdir()
    with zipfile.ZipFile(output) as archive:
        requests = [json.loads(line) for line in archive.read("data_orbital/colab_inputs/requests.jsonl").splitlines()]
    request_index = {row["request_id"]: row for row in requests}
    eval_requests, targets, candidates = [], [], []
    for row in selected:
        relative = Path(os.path.relpath(ROOT/row["image"], references)).as_posix()
        for task in TASKS:
            rid = row["sample_id"]+"__"+task+"__clean"
            question = questions[rid]
            eval_requests.append({**request_index[rid], "image_paths": [relative]})
            targets.append({"request_id": rid, "source_sample_id": row["sample_id"],
                "group_id": f"{row['planet']}:{row['source_dataset']}:{row['scene_group_id']}",
                "derivation_version": VERSION, "task_id": task, "condition_id": "clean",
                "answer": question["candidate_answer"], "reference_status": "CANDIDATE_REFERENCE"})
            candidates.append(question)
    for name, rows in (("requests.jsonl", eval_requests), ("candidate_targets.jsonl", targets), ("source_candidates.jsonl", candidates)):
        write_jsonl(references/name, rows)
    ref_seal = {"schema": "ibm-val14-local-references-v1", "input_bundle_sha256": result["bundle_sha256"],
        "input_zip_sha256": checksum, "pilot_sample_ids": cohort["pilot_sample_ids"],
        "image_sha256": {row["sample_id"]: row["image_sha256"] for row in selected},
        "files_sha256": {name: sha256((references/name).read_bytes()).hexdigest() for name in (
            "requests.jsonl", "candidate_targets.jsonl", "source_candidates.jsonl")}, "scientific_result": False}
    write_json(references/"reference_manifest.json", {**ref_seal, "manifest_sha256": digest(ref_seal)})
    write_json(references/"selection_audit.json", {"selection_version": SELECTION_VERSION, "seed": SEED,
        "policy": "Outcome-independent SHA-256 rank within predeclared lunar source/candidate-presence quotas; one ranked frame per Mars source/scene group. The prior two smoke groups are excluded.",
        "model_predictions_used_for_selection": False, "scope": "val development only",
        "source_annotation_build_sha256": metadata["bundle_sha256"],
        "pilot_images": 14, "pilot_answers": 56, "separate_smoke_images": 2,
        "group_counts": {planet: len({row["scene_group_id"] for row in selected if row["planet"] == planet}) for planet in ("moon", "mars")},
        "local_candidate_presence_counts": dict(Counter(row["planet"]+"/"+questions[row["sample_id"]+"__crater_presence_v1__clean"]["candidate_answer"] for row in selected)),
        "records": [{key: row[key] for key in ("sample_id", "planet", "source_dataset", "source_revision",
            "source_split", "scene_group_id", "image_sha256", "annotation_fingerprint")} for row in selected],
        "limitations": ["Candidate labels and stratification counts stay local, never in the inference ZIP.",
            "All 14 groups are distinct; the sample is stratified and not an unbiased estimate of population accuracy.",
            "Training-overlap/native-NAC validation is not established; published source splits are preserved."]})
    print(json.dumps({**result, "zip_sha256": checksum, "zip_bytes": output.stat().st_size,
        "notebook": str(destination), "local_eval_kit": str(references), "cohort": cohort}, indent=2))


if __name__ == "__main__":
    main()
