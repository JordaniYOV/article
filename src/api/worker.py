"""Durable sequential worker. Run with python -m api.worker or this file."""

from contextlib import contextmanager
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

# Direct file/IDE launches and python -m src.api.worker do not expose the
# sibling planetary_vlm package. Delegate to the canonical module before
# importing application code, using this checkout's src tree.
if __name__ == "__main__" and __package__ != "api":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from api.worker import main as worker_main

    worker_main()
    raise SystemExit

from sqlmodel import Session, SQLModel, select
from sqlalchemy import update as sql_update

from planetary_vlm.contracts import ModelRequest, ModelResponse
from planetary_vlm.inference.runner import dependency_versions
from planetary_vlm.models.registry import create_adapter
from . import settings as paths
from .datasets import digest, file_hash, safe_path
from .models import DatasetConfig, ModelConfig, ModelResult, ModelResultCreate, Run, utc_now
from .transforms import apply_chain, variants


@contextmanager
def gpu_lock(settings):
    """OS lock is released on crash; one GPU worker per project across databases."""
    lock_path = paths.WORKSPACE_ROOT / ".state" / "api-worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
    finally:
        stream.close()


def atomic_json(path, value, *, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        if lines:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, default=str, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


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
    from planetary_vlm.models.ibm_imp import verify_saved_maps
    artifacts = json.loads(raw)
    if Path(artifacts["map_directory"]).resolve() != (folder / "maps").resolve():
        raise ValueError("Segmentation maps must belong to this run")
    if (artifacts["output_type"] != "imp_segmentation" or artifacts["classes"] != ["Background", "IMP"]
        or artifacts["shape"] != [256, 256]
        or any(artifacts[key] not in artifacts["files_sha256"] for key in ("label_png", "arrays_npz"))):
        raise ValueError("Invalid segmentation artifact contract")
    verify_saved_maps(raw)
    return artifacts


def execute(engine, run_id):
    adapter = None
    cleanup_failed = None
    try:
        with Session(engine) as session:
            run = session.get(Run, run_id)
            model, dataset = session.get(ModelConfig, run.model_config_id), session.get(DatasetConfig, run.dataset_config_id)
            prior = session.exec(select(ModelResult).where(ModelResult.run_id == run.id)).all()
        folder = safe_path(run.output_dir, paths.WORKSPACE_ROOT / "outputs")
        input_folder = safe_path(run.id, paths.WORKSPACE_ROOT / "data_orbital" / "inference" / "api")
        protocol = run.protocol
        if not model.enabled or model.config_sha256 != protocol["model_config_sha256"]:
            raise ValueError("Model configuration changed or was disabled")
        if dataset.config_sha256 != protocol["dataset_config_sha256"] or file_hash(dataset.manifest_path) != protocol["manifest_sha256"]:
            raise ValueError("Dataset configuration or manifest changed")
        if digest(run.samples) != protocol["selection_sha256"]:
            raise ValueError("Source selection changed")
        import numpy as np
        from PIL import Image
        requests, records = [], {}
        update(engine, run.id, phase="preparing")
        for source in run.samples:
            if cancelled(engine, run.id):
                update(engine, run.id, status="cancelled", phase="stopped")
                return
            image_path = paths.orbital_path(source["image"])
            if file_hash(image_path) != source["image_sha256"]:
                raise ValueError("Source image changed; resume refused")
            with Image.open(image_path) as image:
                pixels = np.array(image)
            for condition, chain in variants(protocol):
                request_id = digest([source["sample_id"], condition])
                destination = input_folder / "images" / (request_id + ".png")
                transformed, _ = apply_chain(pixels.copy(), chain, protocol["parameters"], protocol["seed"], source["sample_id"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                Image.fromarray(transformed).save(temporary, format="PNG")
                if destination.exists() and file_hash(destination) != file_hash(temporary):
                    temporary.unlink()
                    raise ValueError("Prepared image changed or transform implementation differs")
                os.replace(temporary, destination)
                request = ModelRequest(request_id, (str(destination),), protocol["prompt"],
                    tuple(protocol["allowed_answers"]), max_new_tokens=model.max_new_tokens)
                requests.append(request)
                records[request_id] = {"source_sample_id": source["sample_id"], "scene_group_id": source["scene_group_id"],
                    "source_split": source["source_split"], "condition_id": condition,
                    "input_sha256": file_hash(destination), "chain": chain}
        for row in prior:
            if row.request_id not in records or row.run_metadata["input_sha256"] != records[row.request_id]["input_sha256"]:
                raise ValueError("Saved response input differs from prepared input")
            if row.output_type == "imp_segmentation" and row.status == "ok":
                map_artifacts(row.raw_response, folder)
        done = {row.request_id for row in prior}
        export_predictions(engine, run)
        metadata_path = folder / "run.json"
        versions = dependency_versions()
        if metadata_path.exists():
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (saved_metadata["protocol_sha256"] != run.protocol_sha256
                or saved_metadata["model"]["config_sha256"] != model.config_sha256
                or saved_metadata["dependencies"] != versions):
                raise ValueError("Run metadata or dependency versions changed; resume refused")
        elif prior:
            raise ValueError("Saved responses have no run metadata; resume refused")
        else:
            atomic_json(metadata_path, {**run.model_dump(), "model": model.model_dump(),
                "dependencies": versions, "mock": model.backend == "mock",
                "timing_protocol": "request_wall_seconds_excludes_explicit_initialization_includes_map_save_no_warmup"})
        atomic_json(folder / "requests.jsonl", [asdict(request) for request in requests], lines=True)
        # Evaluation-only source metadata lives separately from model requests.
        atomic_json(folder / "selection.json", run.samples)
        pending = [request for request in requests if request.request_id not in done]
        if cancelled(engine, run.id):
            update(engine, run.id, status="cancelled", phase="stopped")
            return
        if pending:
            update(engine, run.id, phase="loading", completed=len(done))
            random.seed(protocol["seed"])
            np.random.seed(protocol["seed"])
            options = {**model.options, "seed": protocol["seed"], "local_files_only": True}
            if model.backend == "ibm_imp":
                options["artifact_dir"] = str(folder / "maps")
            adapter = create_adapter(model.backend, options)
            if hasattr(adapter, "initialize"):
                start = time.perf_counter()
                provenance = adapter.initialize()
                if run.loading and run.loading.get("provenance") != provenance:
                    raise ValueError("Loaded checkpoint provenance changed; resume refused")
                loading = {"elapsed_seconds": time.perf_counter() - start, "provenance": provenance}
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
                artifacts=artifacts, run_metadata=source, mock=model.backend == "mock"))
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
    """Process one queued job; useful for CLI --once and integration tests."""
    engine = settings.create_engine()
    try:
        SQLModel.metadata.create_all(engine)
        with gpu_lock(settings):
            recover(engine)
            return process_next(engine)
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    settings = paths.Settings()
    if args.once:
        run_once(settings)
        return
    engine = settings.create_engine()
    try:
        SQLModel.metadata.create_all(engine)
        with gpu_lock(settings):
            recover(engine)
            while True:
                if not process_next(engine):
                    time.sleep(settings.worker_poll_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
