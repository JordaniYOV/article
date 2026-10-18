"""Shared durable execution; model services run only explicitly selected jobs."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time

from sqlmodel import Session, SQLModel, select
from sqlalchemy import case, update as sql_update

from api.logging import get_logger

from planetary_vlm.contracts import ModelResponse
from planetary_vlm.inference.runner import dependency_versions
from planetary_vlm.models.registry import create_adapter
from . import settings as paths
from .files import digest, file_hash, safe_path, atomic_json
from .locks import gpu_lock
from .store import DatasetConfig, ModelConfig, ModelResult, ModelResultCreate, Run, utc_now
from .preparation import load_bundle, prepare_bundle, PreparationCancelled


def request_cancellation(session, run_id) -> bool:
    """Atomically cancel queued work or flag a running job without stale ORM writes."""
    changed = session.execute(sql_update(Run).where(
        Run.id == run_id, Run.status.in_(("queued", "running"))).values(
        cancel_requested=True,
        status=case((Run.status == "queued", "cancelled"), else_=Run.status),
        phase=case((Run.status == "queued", "stopped"), else_="cancelling"),
        updated_at=utc_now()).execution_options(synchronize_session=False))
    session.commit()
    session.expire_all()
    return changed.rowcount == 1


def finish(engine, run_id, completed):
    """Cancellation racing with the last response wins until completion is committed."""
    with Session(engine) as session:
        session.execute(sql_update(Run).where(Run.id == run_id, Run.status == "running").values(
            status=case((Run.cancel_requested.is_(True), "cancelled"), else_="completed"),
            phase=case((Run.cancel_requested.is_(True), "stopped"), else_="finished"),
            completed=completed, updated_at=utc_now()))
        session.commit()


def stop_cancelled(engine, run_id):
    """Stop only this still-running cancellation, never a newly resumed queue item."""
    with Session(engine) as session:
        session.execute(sql_update(Run).where(Run.id == run_id, Run.status == "running",
            Run.cancel_requested.is_(True)).values(status="cancelled", phase="stopped",
            error_message=None, updated_at=utc_now()))
        session.commit()


def record_failure(engine, run_id, message, *, cleanup=False):
    """Commit failure atomically, retaining cancellation and any new queued attempt."""
    eligible = ("running", "completed", "blocked", "failed", "interrupted") if cleanup else ("running",)
    with Session(engine) as session:
        session.execute(sql_update(Run).where(Run.id == run_id, Run.status.in_(eligible)).values(
            status=case((Run.cancel_requested.is_(True), "cancelled"), else_="failed" if cleanup else "blocked"),
            phase="stopped", error_message=case((Run.cancel_requested.is_(True), None), else_=message),
            updated_at=utc_now()))
        session.commit()
        return session.get(Run, run_id).status


def update(engine, run_id, **values):
    with Session(engine) as session:
        # A progress write must not hide cancellation that committed concurrently.
        phase = values.get("phase")
        if phase in {"preparing", "loading", "inference"}:
            values["phase"] = case((Run.cancel_requested.is_(True), "cancelling"), else_=phase)
        session.execute(sql_update(Run).where(Run.id == run_id).values(**values, updated_at=utc_now()))
        session.commit()


def cancelled(engine, run_id):
    with Session(engine) as session:
        return session.get(Run, run_id).cancel_requested


def recover(engine):
    # Called only while holding the project GPU lock; no live worker can own these.
    with Session(engine) as session:
        session.execute(sql_update(Run).where(Run.status == "running").values(
            status=case((Run.cancel_requested.is_(True), "cancelled"), else_="interrupted"),
            phase="stopped", error_message=case((Run.cancel_requested.is_(True), None),
                else_="Worker stopped; resume to continue from saved responses"), updated_at=utc_now()))
        session.commit()


def export_predictions(engine, run):
    with Session(engine) as session:
        results = session.exec(select(ModelResult).where(ModelResult.run_id == run.id).order_by(ModelResult.id)).all()
        atomic_json(Path(run.output_dir) / "predictions.jsonl", [row.model_dump() for row in results], lines=True)


def map_artifacts(raw, folder):
    from .segmentation import verify_saved_maps
    artifacts = json.loads(raw)
    if Path(artifacts["map_directory"]).resolve() != (folder / "maps").resolve():
        raise ValueError("Segmentation maps must belong to this run")
    from planetary_vlm.heads_contract import CLASSES, TASK
    imp = artifacts["output_type"] == "imp_segmentation" and artifacts["classes"] == ["Background", "IMP"] and artifacts["shape"] == [256, 256]
    crater = (artifacts["output_type"] == TASK and artifacts["classes"] == CLASSES and artifacts["shape"] == [512, 512]
              and artifacts.get("decision") == "sigmoid_ge_validation_threshold" and 0 < artifacts.get("threshold", 0) < 1)
    if (not (imp or crater)
        or any(artifacts[key] not in artifacts["files_sha256"] for key in ("label_png", "arrays_npz"))):
        raise ValueError("Invalid segmentation artifact contract")
    verify_saved_maps(raw)
    return artifacts


def execute(engine, run_id, *, mock_mode=False, prepared_only=False):
    adapter = None
    cleanup_failed = None
    started = time.perf_counter()
    logger = get_logger("worker.execution", run_id=run_id)
    try:
        with Session(engine) as session:
            run = session.get(Run, run_id)
            model, dataset = session.get(ModelConfig, run.model_config_id), session.get(DatasetConfig, run.dataset_config_id)
            prior = session.exec(select(ModelResult).where(ModelResult.run_id == run.id)).all()
        logger = logger.bind(backend=model.backend, mock=mock_mode or model.backend == "mock")
        logger.info("execution_started")
        if cancelled(engine, run_id):
            stop_cancelled(engine, run_id)
            logger.info("execution_cancelled", reason="before_preparation")
            return
        folder = safe_path(run.output_dir, paths.WORKSPACE_ROOT / "outputs")
        protocol = run.protocol
        if not model.enabled or model.config_sha256 != protocol["model_config_sha256"]:
            raise ValueError("Model configuration changed or was disabled")
        if dataset.config_sha256 != protocol["dataset_config_sha256"] or file_hash(dataset.manifest_path) != protocol["manifest_sha256"]:
            raise ValueError("Dataset configuration or manifest changed")
        if digest(run.samples) != protocol["selection_sha256"]:
            raise ValueError("Source selection changed")
        import numpy as np
        from PIL import Image
        update(engine, run.id, phase="preparing")
        if prepared_only:
            requests, records, prepared_hash = load_bundle(run)
        else:
            requests, records, prepared_hash = prepare_bundle(run, lambda: cancelled(engine, run.id))
        for row in prior:
            if row.request_id not in records or row.run_metadata["input_sha256"] != records[row.request_id]["input_sha256"]:
                raise ValueError("Saved response input differs from prepared input")
            if row.output_type in {"imp_segmentation", "catalog_crater_segmentation"} and row.status == "ok":
                map_artifacts(row.raw_response, folder)
        done = {row.request_id for row in prior}
        export_predictions(engine, run)
        metadata_path = folder / "run.json"
        versions = dependency_versions()
        is_mock = mock_mode or model.backend == "mock"
        execution_mode = "mock" if is_mock else "real"
        pipeline_files = {name: file_hash(Path(__file__).with_name(name))
            for name in ("execution.py", "worker.py", "files.py", "preparation.py", "transforms.py",
                         "segmentation.py", "fixtures.py", "local_runtime.py")}
        adapter_package = {"spacellava": "spacellava", "ibm_imp": "ibm", "frozen_crater": "heads"}.get(model.backend)
        if adapter_package:
            pipeline_files["adapter.py"] = file_hash(Path(__file__).resolve().parents[1] / adapter_package / "adapter.py")
        if model.backend == "frozen_crater":
            pipeline_files["heads_contract.py"] = file_hash(Path(__file__).resolve().parents[2] / "planetary_vlm/heads_contract.py")
            for name in ("encoders.py", "heads.py", "common.py", "compatibility.py", "compatibility_revisions.json"):
                pipeline_files[f"frozen/{name}"] = file_hash(Path(__file__).resolve().parents[3] / "linear_head_approach" / name)
        pipeline_hash = digest(pipeline_files)
        if metadata_path.exists():
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (saved_metadata["protocol_sha256"] != run.protocol_sha256
                or saved_metadata["model"]["config_sha256"] != model.config_sha256
                or saved_metadata["dependencies"] != versions
                or saved_metadata.get("execution_mode") != execution_mode
                or saved_metadata.get("pipeline_sha256") != pipeline_hash
                or saved_metadata.get("prepared_sha256") != prepared_hash):
                raise ValueError("Run metadata or dependency versions changed; resume refused")
        elif prior:
            raise ValueError("Saved responses have no run metadata; resume refused")
        else:
            atomic_json(metadata_path, {**run.model_dump(), "model": model.model_dump(),
                "dependencies": versions, "mock": is_mock, "execution_mode": execution_mode,
                "pipeline_sha256": pipeline_hash, "prepared_sha256": prepared_hash,
                "python_executable": sys.executable,
                "timing_protocol": "request_wall_seconds_excludes_explicit_initialization_includes_map_save_no_warmup"})
        atomic_json(folder / "requests.jsonl", [asdict(request) for request in requests], lines=True)
        # Evaluation-only source metadata lives separately from model requests.
        atomic_json(folder / "selection.json", run.samples)
        pending = [request for request in requests if request.request_id not in done]
        if cancelled(engine, run.id):
            stop_cancelled(engine, run.id)
            logger.info("execution_cancelled", reason="before_loading")
            return
        if pending:
            update(engine, run.id, loading={**run.loading, "mock": is_mock,
                "execution_mode": execution_mode, "pipeline_sha256": pipeline_hash})
            update(engine, run.id, phase="loading", completed=len(done))
            logger.info("model_loading_started", completed=len(done), total=len(requests), pending=len(pending))
            if cancelled(engine, run.id):
                stop_cancelled(engine, run.id)
                logger.info("execution_cancelled", reason="before_adapter_initialization")
                return
            random.seed(protocol["seed"])
            np.random.seed(protocol["seed"])
            options = {**model.options, "seed": protocol["seed"], "local_files_only": True}
            if model.backend == "frozen_crater":
                from planetary_vlm.heads_contract import TRANSFER_PROTOCOL, transfer_input
                # Run-owned input provenance must never come from model config.
                options.pop("transfer_protocol", None)
                options.pop("transfer_inputs", None)
                if protocol.get("preprocessing", {}).get("input_protocol") == TRANSFER_PROTOCOL:
                    options["transfer_protocol"] = TRANSFER_PROTOCOL
                    options["transfer_inputs"] = [transfer_input(row, paths.WORKSPACE_ROOT) for row in run.samples]
            if model.backend in {"ibm_imp", "frozen_crater"}:
                options["artifact_dir"] = str(folder / "maps")
            if mock_mode and model.backend != "mock":
                from .fixtures import ServiceFixture
                adapter = ServiceFixture(model.backend, options)
            else:
                adapter = create_adapter(model.backend, options)
            if hasattr(adapter, "initialize"):
                start = time.perf_counter()
                provenance = adapter.initialize()
                if run.loading.get("provenance") is not None and run.loading["provenance"] != provenance:
                    raise ValueError("Loaded checkpoint provenance changed; resume refused")
                loading = {"elapsed_seconds": time.perf_counter() - start, "provenance": provenance,
                    "mock": is_mock, "execution_mode": execution_mode, "pipeline_sha256": pipeline_hash}
                update(engine, run.id, loading=loading)
                atomic_json(folder / "loading.json", loading)
                logger.info("model_loading_completed", elapsed_seconds=loading["elapsed_seconds"])
            update(engine, run.id, phase="inference")
        for request in pending:
            if cancelled(engine, run.id):
                stop_cancelled(engine, run.id)
                logger.info("execution_cancelled", reason="before_request")
                return
            start = time.perf_counter()
            artifacts = {}
            try:
                response = adapter.predict(request)
                if not isinstance(response, ModelResponse) or response.request_id != request.request_id:
                    raise ValueError("Invalid model response identity")
                if model.backend in {"ibm_imp", "frozen_crater"} and response.status == "ok":
                    artifacts = map_artifacts(response.raw_response, folder)
                    expected_type = "catalog_crater_segmentation" if model.backend == "frozen_crater" else "imp_segmentation"
                    if artifacts["output_type"] != expected_type:
                        raise ValueError("Adapter returned another task's map contract")
                # Validate the entire adapter response before persisting it.
                ModelResultCreate(model_config_id=model.id, dataset_config_id=dataset.id,
                    run_id=run.id, request_id=request.request_id, status=response.status,
                    raw_response=response.raw_response, elapsed_seconds=response.elapsed_seconds)
            except Exception as error:
                logger.error("request_failed", request_id=request.request_id, error_type=type(error).__name__)
                response = ModelResponse(request.request_id, "", "error", time.perf_counter() - start,
                    f"{type(error).__name__}: {error}")
                artifacts = {}
            source = records[request.request_id]
            row = ModelResult.model_validate(ModelResultCreate(model_config_id=model.id,
                dataset_config_id=dataset.id, run_id=run.id, request_id=request.request_id,
                source_sample_id=source["source_sample_id"], scene_group_id=source["scene_group_id"],
                source_split=source["source_split"], condition_id=source["condition_id"],
                image_paths=list(request.image_paths), prompt=request.prompt,
                output_type=("catalog_crater_segmentation" if model.backend == "frozen_crater" else "imp_segmentation" if run.kind == "vit" else "text"),
                status=response.status, raw_response=response.raw_response,
                elapsed_seconds=response.elapsed_seconds, error_message=response.error_message,
                artifacts=artifacts, run_metadata=source, mock=is_mock))
            with Session(engine) as session:
                current = session.get(Run, run.id)
                current.completed += 1
                current.updated_at = utc_now()
                session.add(row)
                session.add(current)
                session.commit()
                completed = current.completed
            logger.info("request_completed", request_id=request.request_id, status=response.status,
                completed=completed, total=len(requests), elapsed_seconds=time.perf_counter() - start)
            # SQLite is authoritative; rebuilding this file also repairs crash gaps.
            export_predictions(engine, run)
        finish(engine, run.id, len(requests))
        with Session(engine) as session:
            final_status = session.get(Run, run.id).status
        logger.info("execution_cancelled" if final_status == "cancelled" else "execution_completed",
            status=final_status, completed=len(requests), total=len(requests),
            elapsed_seconds=time.perf_counter() - started)
    except PreparationCancelled:
        stop_cancelled(engine, run_id)
        logger.info("execution_cancelled", reason="during_preparation")
    except Exception as error:
        failure_status = record_failure(engine, run_id, f"{type(error).__name__}: {error}")
        logger.error("execution_cancelled" if failure_status == "cancelled" else "execution_blocked",
            status=failure_status, error_type=type(error).__name__)
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                cleanup_failed = error
                record_failure(engine, run_id, f"Resource cleanup failed: {error}", cleanup=True)
                logger.error("resource_cleanup_failed", error_type=type(error).__name__)
            else:
                logger.info("resource_cleanup_completed")
        with Session(engine) as session:
            run = session.get(Run, run_id)
            folder = safe_path(run.output_dir, paths.WORKSPACE_ROOT / "outputs")
            atomic_json(folder / "status.json", run.model_dump())
        if cleanup_failed is not None:
            raise RuntimeError("Worker stopped because model resources could not be released") from cleanup_failed


def process_next(engine):
    with Session(engine) as session:
        run = session.exec(select(Run).where(Run.status == "queued", Run.cancel_requested.is_(False))
            .order_by(Run.created_at, Run.id).limit(1)).first()
        if run is None:
            return False
        run_id = run.id
        claimed = session.execute(sql_update(Run).where(Run.id == run_id, Run.status == "queued",
            Run.cancel_requested.is_(False)).values(
            status="running", phase="preparing", updated_at=utc_now()))
        session.commit()
        if claimed.rowcount != 1:
            return True  # Cancellation won the race; check the next queue item.
    execute(engine, run_id)
    return True


def run_once(settings):
    """In-process regression-test helper; production uses run_selected only."""
    engine = settings.create_engine()
    try:
        SQLModel.metadata.create_all(engine)
        with gpu_lock(settings):
            recover(engine)
            return process_next(engine)
    finally:
        engine.dispose()


def run_selected(settings, run_id, backends, *, mock_mode=False):
    """Claim this exact queued job under the GPU lock; never consume a different model."""
    engine = settings.create_engine()
    try:
        SQLModel.metadata.create_all(engine)
        with gpu_lock(settings):
            with Session(engine) as session:
                selected = session.get(Run, run_id)
                if selected is None:
                    raise ValueError("Run not found")
                model = session.get(ModelConfig, selected.model_config_id)
                if model.backend not in backends:
                    raise ValueError("Run belongs to a different model service")
                if selected.status != "queued" or selected.cancel_requested:
                    return False
            recover(engine)
            with Session(engine) as session:
                claimed = session.execute(sql_update(Run).where(Run.id == run_id, Run.status == "queued",
                    Run.cancel_requested.is_(False)).values(
                    status="running", phase="preparing", error_message=None, updated_at=utc_now()))
                session.commit()
                if claimed.rowcount != 1:
                    return False
            get_logger("worker.execution", run_id=run_id, mock=mock_mode, backend=model.backend).info("job_claimed")
            attempt_id = os.environ.get("PLANETARY_SERVICE_ATTEMPT_ID")
            if attempt_id:
                folder = safe_path(selected.output_dir, paths.WORKSPACE_ROOT / "outputs")
                atomic_json(folder / "service-attempt.json", {"attempt_id": attempt_id, "claimed": True})
            execute(engine, run_id, mock_mode=mock_mode, prepared_only=True)
            return True
    finally:
        engine.dispose()
