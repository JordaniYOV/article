"""One-model run planning and immutable source selections."""

from uuid import uuid4

from fastapi import HTTPException
from sqlmodel import select
from sqlalchemy import update as sql_update

from benchmark_services.common import settings as paths
from .datasets import sample_rows
from benchmark_services.common.files import digest, file_hash, safe_path
from benchmark_services.common.store import DatasetConfig, ModelConfig, Run, utc_now
from benchmark_services.common.transforms import variants
from benchmark_services.common.preparation import renderer_provenance
from benchmark_services.common.execution import request_cancellation
from .logging import get_logger

LOGGER = get_logger("api.runs")

KINDS = {"vlm": {"spacellava", "mock"}, "vit": {"ibm_imp", "frozen_crater"}}
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
        protocol.update({"version": "orbital-services-v2", "renderer": renderer_provenance(), "dataset_config_sha256": dataset.config_sha256,
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
        if model.backend == "frozen_crater":
            from planetary_vlm.heads_contract import TASK, INPUT_PROTOCOL, TRANSFER_PROTOCOL, read_deployment, pixel_hash, transfer_input
            input_protocol = dataset.preprocessing.get("input_protocol")
            if (dataset.task_id != TASK or input_protocol not in {INPUT_PROTOCOL, TRANSFER_PROTOCOL}
                or model.options.get("input_protocol") != INPUT_PROTOCOL or payload.distortions or not payload.include_clean or payload.allowed_answers):
                raise ValueError("Trained heads require clean binary crater segmentation without question answers/corruptions. "
                                 "Connect the external crater collection or a declared frozen-head dataset")
            deployment_path, deployment = read_deployment(model.options["deployment_path"], model.options["deployment_sha256"], paths.WORKSPACE_ROOT)
            if input_protocol == TRANSFER_PROTOCOL:
                from planetary_vlm.heads_contract import reject_training_overlap, CRATER_SEMANTICS, CLASSES
                expected_manifest = dataset.provenance.get("manifest_sha256")
                if expected_manifest and file_hash(dataset.manifest_path) != expected_manifest:
                    raise ValueError("Registered external crater manifest changed; create a new dataset version")
                reject_training_overlap(selected, paths.WORKSPACE_ROOT,
                    deployment.get("training_identity", {}).get("feature_identity", {}).get("inputs_sha256"))
                if any(row.get("mask_semantics") not in CRATER_SEMANTICS or
                       row.get("classes") not in [CLASSES, ["Background", "Crater"]] for row in selected):
                    raise ValueError("External heads need declared binary crater masks, not IMP/terrain labels")
            from pathlib import Path
            # Deployment index is in outputs, not a dataset manifest. It contains inputs only.
            import json
            index_path = deployment_path.parent / "encoder_inputs.jsonl"
            admitted = {r["sample_id"]: r for r in map(json.loads, index_path.read_text(encoding="utf-8").splitlines()) if r}
            for row in selected:
                if input_protocol == TRANSFER_PROTOCOL:
                    transfer_input(row, paths.WORKSPACE_ROOT, deployment["encoder"])
                    continue
                known = admitted.get(row["sample_id"])
                if not known or known["split"] != payload.split or known["scene_group_id"] != row["scene_group_id"]:
                    raise ValueError("Head dataset must preserve admitted held-out IDs/groups/splits")
                if pixel_hash(row["image"]) != known["rgb_pixels_sha256"] or row.get("native_vis_sha256") != known["native_vis_sha256"]:
                    raise ValueError("Head source image/native VIS differs from the deployment")
        elif kind == "vit":
            expected = "curated_imp_png_reflectance_0_0.12_v1"
            if (dataset.planet != "moon" or dataset.task_id != "imp_segmentation"
                or dataset.preprocessing.get("input_protocol") != expected
                or model.options.get("input_protocol") != expected):
                raise ValueError("IBM IMP head requires the declared lunar IMP PNG input protocol")
            from benchmark_services.common.segmentation import restore_imp_png
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
        LOGGER.info("run_queued", run_id=run.id, backend=model.backend, status=run.status, total=run.total)
        return run
    except (ValueError, OSError, KeyError) as error:
        session.rollback()
        raise HTTPException(422, str(error)) from error


def control_run(session, run_id, action):
    run = find_run(session, run_id)
    expected_status = run.status
    values = {"updated_at": utc_now()}
    if action == "cancel":
        # The worker may claim the run after find_run(). The shared atomic
        # transition handles either current queued/running state, not that snapshot.
        changed = request_cancellation(session, run_id)
        session.refresh(run)
        if not changed and run.status != "cancelled":
            LOGGER.warning("run_cancel_rejected", run_id=run.id, status=run.status)
            raise HTTPException(409, "Run is not queued or running")
        LOGGER.info("run_cancel_requested" if changed else "run_cancel_already_applied",
                    run_id=run.id, status=run.status, phase=run.phase)
        return run
    elif action == "resume":
        if run.status not in {"cancelled", "interrupted", "blocked", "failed"}:
            raise HTTPException(409, "Only a stopped or blocked run can be resumed")
        values.update(status="queued", phase="waiting", cancel_requested=False, error_message=None)
    else:
        raise HTTPException(422, "Unknown run action")
    changed = session.execute(sql_update(Run).where(Run.id == run_id, Run.status == expected_status).values(**values))
    session.commit()
    if changed.rowcount != 1:
        raise HTTPException(409, "Run status changed during this request; refresh its status")
    session.refresh(run)
    LOGGER.info("run_resumed", run_id=run.id, status=run.status)
    return run
