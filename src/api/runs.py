"""One-model run planning and immutable source selections."""

from uuid import uuid4

from fastapi import HTTPException
from sqlmodel import select
from sqlalchemy import update as sql_update

from . import settings as paths
from .datasets import digest, file_hash, sample_rows, safe_path
from .models import DatasetConfig, ModelConfig, Run, utc_now
from .transforms import variants

KINDS = {"vlm": {"spacellava", "mock"}, "vit": {"ibm_imp"}}
TERMINAL = {"completed", "cancelled", "interrupted", "blocked", "failed"}


def find_run(session, run_id):
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


def create_run(payload, kind, session):
    model = session.exec(select(ModelConfig).where(ModelConfig.name == payload.model_name,
        ModelConfig.version == payload.model_version)).first()
    dataset = session.get(DatasetConfig, payload.dataset_id)
    if model is None or dataset is None:
        raise HTTPException(404, "Model configuration or dataset not found")
    if model.backend not in KINDS[kind]:
        raise HTTPException(422, "Model backend does not match this endpoint; ViT needs a supported published head")
    if not model.enabled:
        raise HTTPException(409, "Enable the configured model before launching a run")
    if model.options.get("local_files_only", True) is not True:
        raise HTTPException(422, "Implicit weight downloads are disabled")
    if payload.split not in dataset.splits:
        raise HTTPException(422, "Selected split is absent from the dataset")
    if dataset.targets_path:
        raise HTTPException(422, "API scoring uses answers/masks in the source manifest; convert separate targets first")
    try:
        sources = sample_rows(dataset)
        sources = [row for row in sources if row["source_split"] == payload.split]
        sources.sort(key=lambda row: digest([payload.seed, row["sample_id"]]))
        if len(sources) < payload.image_count:
            raise ValueError(f"Only {len(sources)} eligible images in this split")
        selected = sources[:payload.image_count]
        protocol = payload.model_dump(exclude={"model_name", "model_version", "dataset_id", "reuse_run_id"})
        protocol.update({"version": "orbital-api-v1", "dataset_config_sha256": dataset.config_sha256,
            "manifest_sha256": file_hash(dataset.manifest_path), "task_id": dataset.task_id,
            "planet": dataset.planet, "preprocessing": dataset.preprocessing,
            "max_new_tokens": model.max_new_tokens})
        # The selection includes source hashes, groups, splits and target provenance.
        protocol["selection_sha256"] = digest(selected)
        protocol["model_config_sha256"] = model.config_sha256
        comparable = {key: value for key, value in protocol.items() if key != "model_config_sha256"}
        protocol_hash = digest(comparable)
        if payload.reuse_run_id:
            other = find_run(session, payload.reuse_run_id)
            if other.dataset_config_id != dataset.id or other.protocol_sha256 != protocol_hash:
                raise ValueError("Reuse requires the same dataset, split, count, seed, prompt and distortion protocol")
            selected = other.samples
        if kind == "vit":
            expected = "curated_imp_png_reflectance_0_0.12_v1"
            if (dataset.planet != "moon" or dataset.task_id != "imp_segmentation"
                or dataset.preprocessing.get("input_protocol") != expected
                or model.options.get("input_protocol") != expected):
                raise ValueError("IBM IMP head requires the declared lunar IMP PNG input protocol")
            from planetary_vlm.models.ibm_imp import restore_imp_png
            for row in selected:
                restore_imp_png(row["image"])
        run_id = uuid4().hex
        # Model names may be display labels, never arbitrary filesystem paths.
        if (model.name in {".", ".."} or model.name.endswith((".", " "))
            or any(ord(char) < 32 or char in '/\\:<>|?*"' for char in model.name)
            or model.name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL",
                *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}):
            raise ValueError("Model name cannot be an output directory name")
        directory = safe_path(model.name + "/" + run_id, paths.WORKSPACE_ROOT / "outputs")
        run = Run(id=run_id, model_config_id=model.id, dataset_config_id=dataset.id, kind=kind,
            protocol=protocol, protocol_sha256=protocol_hash, samples=selected,
            total=len(selected) * len(variants(protocol)), output_dir=str(directory))
        session.add(run)
        session.commit()
        session.refresh(run)
        return run
    except (ValueError, OSError, KeyError) as error:
        session.rollback()
        raise HTTPException(422, str(error)) from error


def control_run(session, run_id, action):
    run = find_run(session, run_id)
    expected_status = run.status
    values = {"updated_at": utc_now()}
    if action == "cancel":
        if run.status == "queued":
            values.update(status="cancelled", phase="stopped", cancel_requested=True)
        elif run.status == "running":
            values.update(cancel_requested=True, phase="cancelling")
        elif run.status != "cancelled":
            raise HTTPException(409, "Run is not queued or running")
    else:
        if run.status not in {"cancelled", "interrupted", "blocked", "failed"}:
            raise HTTPException(409, "Only a stopped or blocked run can be resumed")
        values.update(status="queued", phase="waiting", cancel_requested=False, error_message=None)
    changed = session.execute(sql_update(Run).where(Run.id == run_id, Run.status == expected_status).values(**values))
    session.commit()
    if changed.rowcount != 1:
        raise HTTPException(409, "Run status changed during this request; refresh its status")
    session.refresh(run)
    return run
