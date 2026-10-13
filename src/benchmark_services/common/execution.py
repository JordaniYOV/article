"""Shared durable execution; model services run only explicitly selected jobs."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time

from sqlmodel import Session, SQLModel, select
from sqlalchemy import update as sql_update

from planetary_vlm.contracts import ModelResponse
from planetary_vlm.inference.runner import dependency_versions
from planetary_vlm.models.registry import create_adapter
from . import settings as paths
from .files import digest, file_hash, safe_path, atomic_json
from .locks import gpu_lock
from .store import DatasetConfig, ModelConfig, ModelResult, ModelResultCreate, Run, utc_now
from .preparation import load_bundle, prepare_bundle, PreparationCancelled



def update(engine, run_id, **values):
    with Session(engine) as session:
        run = session.get(Run, run_id)
        run.sqlmodel_update({**values, "updated_at": utc_now()})
        session.add(run)
        session.commit()


def cancelled(engine, run_id):
    with Session(engine) as session:
        return session.get(Run, run_id).cancel_requested


def recover(engine):
    # Called only while holding the project GPU lock; no live worker can own these.
    with Session(engine) as session:
        for run in session.exec(select(Run).where(Run.status == "running")):
            run.status, run.phase = "interrupted", "stopped"
            run.error_message = "Worker stopped; resume to continue from saved responses"
            run.updated_at = utc_now()
            session.add(run)
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
    if (artifacts["output_type"] != "imp_segmentation" or artifacts["classes"] != ["Background", "IMP"]
        or artifacts["shape"] != [256, 256]
        or any(artifacts[key] not in artifacts["files_sha256"] for key in ("label_png", "arrays_npz"))):
        raise ValueError("Invalid segmentation artifact contract")
    verify_saved_maps(raw)
    return artifacts


def execute(engine, run_id, *, mock_mode=False, prepared_only=False):
    adapter = None
    cleanup_failed = None
    try:
        with Session(engine) as session:
            run = session.get(Run, run_id)
            model, dataset = session.get(ModelConfig, run.model_config_id), session.get(DatasetConfig, run.dataset_config_id)
            prior = session.exec(select(ModelResult).where(ModelResult.run_id == run.id)).all()
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
            if row.output_type == "imp_segmentation" and row.status == "ok":
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
        adapter_package = {"spacellava": "spacellava", "ibm_imp": "ibm"}.get(model.backend)
        if adapter_package:
            pipeline_files["adapter.py"] = file_hash(Path(__file__).resolve().parents[1] / adapter_package / "adapter.py")
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
            update(engine, run.id, status="cancelled", phase="stopped")
            return
        if pending:
            update(engine, run.id, loading={**run.loading, "mock": is_mock,
                "execution_mode": execution_mode, "pipeline_sha256": pipeline_hash})
            update(engine, run.id, phase="loading", completed=len(done))
            random.seed(protocol["seed"])
            np.random.seed(protocol["seed"])
            options = {**model.options, "seed": protocol["seed"], "local_files_only": True}
            if model.backend == "ibm_imp":
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
            update(engine, run.id, phase="inference")
        for request in pending:
            if cancelled(engine, run.id):
                update(engine, run.id, status="cancelled", phase="stopped")
                return
            start = time.perf_counter()
            artifacts = {}
            try:
                response = adapter.predict(request)
                if not isinstance(response, ModelResponse) or response.request_id != request.request_id:
                    raise ValueError("Invalid model response identity")
                if model.backend == "ibm_imp" and response.status == "ok":
                    artifacts = map_artifacts(response.raw_response, folder)
                # Validate the entire adapter response before persisting it.
                ModelResultCreate(model_config_id=model.id, dataset_config_id=dataset.id,
                    run_id=run.id, request_id=request.request_id, status=response.status,
                    raw_response=response.raw_response, elapsed_seconds=response.elapsed_seconds)
            except Exception as error:
                response = ModelResponse(request.request_id, "", "error", time.perf_counter() - start,
                    f"{type(error).__name__}: {error}")
                artifacts = {}
            source = records[request.request_id]
            row = ModelResult.model_validate(ModelResultCreate(model_config_id=model.id,
                dataset_config_id=dataset.id, run_id=run.id, request_id=request.request_id,
                source_sample_id=source["source_sample_id"], scene_group_id=source["scene_group_id"],
                source_split=source["source_split"], condition_id=source["condition_id"],
                image_paths=list(request.image_paths), prompt=request.prompt,
                output_type="imp_segmentation" if run.kind == "vit" else "text",
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
            # SQLite is authoritative; rebuilding this file also repairs crash gaps.
            export_predictions(engine, run)
        update(engine, run.id, status="cancelled" if cancelled(engine, run.id) else "completed",
            phase="finished", completed=len(requests))
    except PreparationCancelled:
        update(engine, run_id, status="cancelled", phase="stopped")
    except Exception as error:
        update(engine, run_id, status="blocked", phase="stopped", error_message=f"{type(error).__name__}: {error}")
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                cleanup_failed = error
                update(engine, run_id, status="failed", phase="stopped", error_message=f"Resource cleanup failed: {error}")
        with Session(engine) as session:
            run = session.get(Run, run_id)
            folder = safe_path(run.output_dir, paths.WORKSPACE_ROOT / "outputs")
            atomic_json(folder / "status.json", run.model_dump())
        if cleanup_failed is not None:
            raise RuntimeError("Worker stopped because model resources could not be released") from cleanup_failed


def process_next(engine):
    with Session(engine) as session:
        run = session.exec(select(Run).where(Run.status == "queued").order_by(Run.created_at, Run.id).limit(1)).first()
        if run is None:
            return False
        run_id = run.id
        claimed = session.execute(sql_update(Run).where(Run.id == run_id, Run.status == "queued").values(
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
                if selected.status != "queued":
                    return False
            recover(engine)
            with Session(engine) as session:
                claimed = session.execute(sql_update(Run).where(Run.id == run_id, Run.status == "queued").values(
                    status="running", phase="preparing", error_message=None, updated_at=utc_now()))
                session.commit()
                if claimed.rowcount != 1:
                    return False
            attempt_id = os.environ.get("PLANETARY_SERVICE_ATTEMPT_ID")
            if attempt_id:
                folder = safe_path(selected.output_dir, paths.WORKSPACE_ROOT / "outputs")
                atomic_json(folder / "service-attempt.json", {"attempt_id": attempt_id, "claimed": True})
            execute(engine, run_id, mock_mode=mock_mode, prepared_only=True)
            return True
    finally:
        engine.dispose()
