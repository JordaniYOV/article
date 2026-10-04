"""FastAPI routes for configuration, saved results and metric analyses."""

from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
from typing import Annotated
from io import BytesIO
import http.client
from urllib.parse import urlsplit, urlunsplit
import zipfile
from PIL import Image

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, select

from .models import (
    DatasetConfig, DatasetConfigCreate, DatasetConfigRead,
    ImportRun, MetricAnalysis, MetricAnalysisCreate, MetricAnalysisRead,
    ModelConfig, ModelConfigCreate, ModelConfigRead,
    ModelResult, ModelResultCreate, ModelResultRead, Run, RunCreate, DatasetURL, ComparisonCreate, utc_now,
)
from .settings import Settings, WORKSPACE_ROOT
from .datasets import ingest_zip, download_zip, safe_path
from .runs import create_run, find_run, control_run
from .evaluation import calculate, report, compare


def fingerprint(value) -> str:
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)
    return hashlib.sha256(content.encode()).hexdigest()


def get_session(request: Request):
    with Session(request.app.state.engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
Limit = Annotated[int, Query(ge=1, le=500)]
Offset = Annotated[int, Query(ge=0)]


def get_record(session, table, record_id):
    record = session.get(table, record_id)
    if record is None:
        raise HTTPException(404, f"{table.__tablename__} record not found")
    return record


def commit(session, record):
    session.add(record)
    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(409, "Duplicate record or a referenced record prevents this operation") from error
    session.refresh(record)
    return record


def delete_record(session, record):
    session.delete(record)
    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(409, "Record is referenced by saved results or analyses") from error
    return Response(status_code=204)


def ensure_unused_config(session, record, field):
    for table in (ModelResult, MetricAnalysis, Run):
        used = session.exec(select(table.id).where(getattr(table, field) == record.id).limit(1)).first()
        if used is not None:
            raise HTTPException(409, "Configuration already has results; create a new version")


def validate_result_scope(session, payload):
    get_record(session, ModelConfig, payload.model_config_id)
    dataset = get_record(session, DatasetConfig, payload.dataset_config_id)
    if payload.condition_id not in dataset.conditions:
        raise HTTPException(422, "Result condition is outside the dataset configuration")
    if payload.source_split is not None and payload.source_split not in dataset.splits:
        raise HTTPException(422, "Result split is outside the dataset configuration")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application):
        engine = settings.create_engine()
        application.state.engine = engine
        application.state.settings = settings
        try:
            SQLModel.metadata.create_all(engine)
            yield
        finally:
            engine.dispose()

    application = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware, 
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"], 
        allow_headers=["Content-Type"])

    @application.get("/health", tags=["health"])
    def health(session: SessionDep):
        session.exec(select(1)).one()
        return {"status": "ok", "database": "sqlite"}

    @application.post("/model-configs", response_model=ModelConfigRead, status_code=201, tags=["model configs"])
    def create_model_config(payload: ModelConfigCreate, session: SessionDep):
        values = payload.model_dump()
        return commit(session, ModelConfig.model_validate({**values, "config_sha256": fingerprint(values)}))

    @application.get("/model-configs", response_model=list[ModelConfigRead], tags=["model configs"])
    def list_model_configs(session: SessionDep, offset: Offset = 0, limit: Limit = 100):
        return session.exec(select(ModelConfig).order_by(ModelConfig.id).offset(offset).limit(limit)).all()

    @application.get("/model-configs/{config_id}", response_model=ModelConfigRead, tags=["model configs"])
    def read_model_config(config_id: int, session: SessionDep):
        return get_record(session, ModelConfig, config_id)

    @application.put("/model-configs/{config_id}", response_model=ModelConfigRead, tags=["model configs"])
    def update_model_config(config_id: int, payload: ModelConfigCreate, session: SessionDep):
        record = get_record(session, ModelConfig, config_id)
        ensure_unused_config(session, record, "model_config_id")
        values = payload.model_dump()
        record.sqlmodel_update({**values, "config_sha256": fingerprint(values), "updated_at": utc_now()})
        return commit(session, record)

    @application.delete("/model-configs/{config_id}", status_code=204, tags=["model configs"])
    def delete_model_config(config_id: int, session: SessionDep):
        return delete_record(session, get_record(session, ModelConfig, config_id))

    @application.post("/dataset-configs", response_model=DatasetConfigRead, status_code=201, tags=["dataset configs"])
    def create_dataset_config(payload: DatasetConfigCreate, session: SessionDep):
        values = payload.model_dump()
        return commit(session, DatasetConfig.model_validate({**values, "config_sha256": fingerprint(values)}))

    @application.get("/dataset-configs", response_model=list[DatasetConfigRead], tags=["dataset configs"])
    def list_dataset_configs(session: SessionDep, offset: Offset = 0, limit: Limit = 100):
        return session.exec(select(DatasetConfig).order_by(DatasetConfig.id).offset(offset).limit(limit)).all()

    @application.get("/dataset-configs/{config_id}", response_model=DatasetConfigRead, tags=["dataset configs"])
    def read_dataset_config(config_id: int, session: SessionDep):
        return get_record(session, DatasetConfig, config_id)

    @application.put("/dataset-configs/{config_id}", response_model=DatasetConfigRead, tags=["dataset configs"])
    def update_dataset_config(config_id: int, payload: DatasetConfigCreate, session: SessionDep):
        record = get_record(session, DatasetConfig, config_id)
        ensure_unused_config(session, record, "dataset_config_id")
        values = payload.model_dump()
        record.sqlmodel_update({**values, "config_sha256": fingerprint(values), "updated_at": utc_now()})
        return commit(session, record)

    @application.delete("/dataset-configs/{config_id}", status_code=204, tags=["dataset configs"])
    def delete_dataset_config(config_id: int, session: SessionDep):
        return delete_record(session, get_record(session, DatasetConfig, config_id))

    @application.post("/model-results", response_model=ModelResultRead, status_code=201, tags=["model results"])
    def create_model_result(payload: ModelResultCreate, session: SessionDep):
        if session.get(Run, payload.run_id):
            raise HTTPException(409, "Queued-run responses are written only by the worker")
        validate_result_scope(session, payload)
        return commit(session, ModelResult.model_validate(payload))

    @application.get("/model-results", response_model=list[ModelResultRead], tags=["model results"])
    def list_model_results(session: SessionDep, model_config_id: int | None = None,
                           dataset_config_id: int | None = None, run_id: str | None = None,
                           condition_id: str | None = None, offset: Offset = 0, limit: Limit = 100):
        query = select(ModelResult)
        for name, value in (("model_config_id", model_config_id), ("dataset_config_id", dataset_config_id),
                            ("run_id", run_id), ("condition_id", condition_id)):
            if value is not None:
                query = query.where(getattr(ModelResult, name) == value)
        return session.exec(query.order_by(ModelResult.id).offset(offset).limit(limit)).all()

    @application.get("/model-results/{result_id}", response_model=ModelResultRead, tags=["model results"])
    def read_model_result(result_id: int, session: SessionDep):
        return get_record(session, ModelResult, result_id)

    @application.delete("/model-results/{result_id}", status_code=204, tags=["model results"])
    def delete_model_result(result_id: int, session: SessionDep):
        record = get_record(session, ModelResult, result_id)
        if session.get(Run, record.run_id):
            raise HTTPException(409, "Queued-run responses are immutable")
        analyses = session.exec(select(MetricAnalysis).where(
            MetricAnalysis.model_config_id == record.model_config_id,
            MetricAnalysis.dataset_config_id == record.dataset_config_id,
            MetricAnalysis.run_id == record.run_id)).all()
        if any(result_id in item.result_ids for item in analyses):
            raise HTTPException(409, "Result is included in a metric analysis")
        return delete_record(session, record)

    @application.post("/metric-analyses", response_model=MetricAnalysisRead, status_code=201, tags=["metric analyses"])
    def create_metric_analysis(payload: MetricAnalysisCreate, session: SessionDep):
        if session.get(Run, payload.run_id):
            raise HTTPException(409, "Use the run metrics endpoint to calculate these analyses")
        dataset = get_record(session, DatasetConfig, payload.dataset_config_id)
        get_record(session, ModelConfig, payload.model_config_id)
        if dataset.task_id != payload.task_id:
            raise HTTPException(422, "Analysis task does not match the dataset task")
        results = session.exec(select(ModelResult).where(ModelResult.id.in_(payload.result_ids)).order_by(ModelResult.id)).all()
        if len(results) != len(payload.result_ids):
            raise HTTPException(422, "Some result_ids do not exist")
        for result in results:
            if ((result.model_config_id, result.dataset_config_id, result.run_id) !=
                    (payload.model_config_id, payload.dataset_config_id, payload.run_id)
                    or (payload.condition_id is not None and result.condition_id != payload.condition_id)):
                raise HTTPException(422, "Analysis results must belong to the selected model/dataset/run/condition")
        if len({result.mock for result in results}) != 1:
            raise HTTPException(422, "Analysis cannot mix fixture and real model results")
        values = payload.model_dump()
        record = MetricAnalysis.model_validate({**values,
            "results_sha256": fingerprint([item.model_dump() for item in results]), "mock": results[0].mock})
        return commit(session, record)

    @application.get("/metric-analyses", response_model=list[MetricAnalysisRead], tags=["metric analyses"])
    def list_metric_analyses(session: SessionDep, model_config_id: int | None = None,
                            dataset_config_id: int | None = None, run_id: str | None = None,
                            offset: Offset = 0, limit: Limit = 100):
        query = select(MetricAnalysis)
        for name, value in (("model_config_id", model_config_id), ("dataset_config_id", dataset_config_id), ("run_id", run_id)):
            if value is not None:
                query = query.where(getattr(MetricAnalysis, name) == value)
        return session.exec(query.order_by(MetricAnalysis.id).offset(offset).limit(limit)).all()

    @application.get("/metric-analyses/{analysis_id}", response_model=MetricAnalysisRead, tags=["metric analyses"])
    def read_metric_analysis(analysis_id: int, session: SessionDep):
        return get_record(session, MetricAnalysis, analysis_id)

    @application.delete("/metric-analyses/{analysis_id}", status_code=204, tags=["metric analyses"])
    def delete_metric_analysis(analysis_id: int, session: SessionDep):
        return delete_record(session, get_record(session, MetricAnalysis, analysis_id))

    @application.post("/configurations/import-defaults", tags=["configuration import"])
    def import_defaults(session: SessionDep):
        from planetary_vlm.inference.full_systems import load_pipeline
        config, _, runs = load_pipeline(WORKSPACE_ROOT / "configs" / "full_systems.toml")
        records = []
        for name, run in runs.items():
            payload = ModelConfigCreate(name=name, model_id=run.model_id, backend=run.backend, seed=run.seed,
                revision=run.options.get("model_revision", run.options.get("head_revision")),
                options=run.options, max_new_tokens=config["generation"]["max_new_tokens"])
            records.append((ModelConfig, payload))
        dataset = config["dataset"]
        records.append((DatasetConfig, DatasetConfigCreate(name="orbital_imp_png", planet="moon",
            task_id="imp_segmentation", manifest_path=str(dataset["manifest"]),
            requests_path=str(dataset["prepared_dir"] / "requests.jsonl"), splits=dataset["splits"],
            conditions=dataset["conditions"], preprocessing={"generation": config["generation"],
                "distortion": config["distortion"], "input_protocol": "curated_imp_png_reflectance_0_0.12_v1"},
            provenance={"source_dataset": dataset["source_dataset"], "pipeline_config": "configs/full_systems.toml"})))
        saved = {"model_config_ids": [], "dataset_config_ids": []}
        for table, payload in records:
            values = payload.model_dump()
            digest = fingerprint(values)
            existing = session.exec(select(table).where(table.name == payload.name, table.version == payload.version)).first()
            if existing is not None and existing.config_sha256 != digest:
                session.rollback()
                raise HTTPException(409, "Existing default configuration differs; create a new version")
            record = existing or table.model_validate({**values, "config_sha256": digest})
            session.add(record)
            session.flush()
            saved["model_config_ids" if table is ModelConfig else "dataset_config_ids"].append(record.id)
        session.commit()
        return saved

    @application.post("/imports/results", tags=["result import"])
    def import_results(payload: ImportRun, session: SessionDep):
        if session.get(Run, payload.run_id):
            raise HTTPException(409, "This run is owned by the queue worker")
        return import_saved_run(payload, session)

    @application.post("/datasets/upload", response_model=DatasetConfigRead, status_code=201, tags=["datasets"])
    def upload_dataset(request: Request, session: SessionDep,
                       file: Annotated[UploadFile, File()], metadata: Annotated[str, Form()]):
        """ZIP file and a JSON metadata form field (name, planet, task, provenance)."""
        try:
            values = json.loads(metadata)
            payload = DatasetURL.model_validate({**values, "url": "uploaded-zip"})
            content = BytesIO()
            while chunk := file.file.read(1024 * 1024):
                content.write(chunk)
                if content.tell() > settings.max_upload_bytes:
                    raise HTTPException(413, "Upload exceeds size limit")
            content.seek(0)
            return ingest_zip(content, payload, request.app.state.settings, session)
        except HTTPException:
            raise
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile, Image.DecompressionBombError) as error:
            raise HTTPException(422, f"Cannot import dataset: {error}") from error
        finally:
            file.file.close()

    @application.post("/datasets/from-url", response_model=DatasetConfigRead, status_code=201, tags=["datasets"])
    def dataset_from_url(payload: DatasetURL, request: Request, session: SessionDep):
        """Import a bounded direct ZIP URL; repository pages are not archive URLs."""
        try:
            content = download_zip(payload.url, request.app.state.settings)
            source_url = urlsplit(payload.url)
            # Preserve source location without query tokens or credentials.
            payload.provenance = {**payload.provenance, "source_url": urlunsplit((
                source_url.scheme, source_url.netloc, source_url.path, "", ""))}
            return ingest_zip(content, payload, request.app.state.settings, session)
        except HTTPException:
            raise
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile,
                http.client.HTTPException, Image.DecompressionBombError) as error:
            raise HTTPException(422, f"Cannot import dataset: {error}") from error

    @application.get("/datasets", response_model=list[DatasetConfigRead], tags=["datasets"])
    def datasets(session: SessionDep, offset: Offset = 0, limit: Limit = 100):
        return session.exec(select(DatasetConfig).order_by(DatasetConfig.id).offset(offset).limit(limit)).all()

    @application.post("/vlm/runs", status_code=202, tags=["runs"])
    def start_vlm(payload: RunCreate, session: SessionDep):
        return run_view(create_run(payload, "vlm", session))

    @application.post("/vit/runs", status_code=202, tags=["runs"])
    def start_vit(payload: RunCreate, session: SessionDep):
        return run_view(create_run(payload, "vit", session))

    @application.get("/runs", tags=["runs"])
    def runs(session: SessionDep, status: str | None = None, model_config_id: int | None = None,
             offset: Offset = 0, limit: Limit = 100):
        query = select(Run)
        if status:
            query = query.where(Run.status == status)
        if model_config_id is not None:
            query = query.where(Run.model_config_id == model_config_id)
        return [run_view(row) for row in session.exec(query.order_by(Run.created_at.desc()).offset(offset).limit(limit))]

    @application.get("/runs/{run_id}", tags=["runs"])
    def read_run(run_id: str, session: SessionDep):
        return run_view(find_run(session, run_id))

    @application.post("/runs/{run_id}/cancel", tags=["runs"])
    def cancel_run(run_id: str, session: SessionDep):
        return run_view(control_run(session, run_id, "cancel"))

    @application.post("/runs/{run_id}/resume", status_code=202, tags=["runs"])
    def resume_run(run_id: str, session: SessionDep):
        return run_view(control_run(session, run_id, "resume"))

    @application.get("/runs/{run_id}/results", tags=["runs"])
    def results(run_id: str, session: SessionDep, condition: str | None = None,
                status: str | None = None, offset: Offset = 0, limit: Limit = 100):
        find_run(session, run_id)
        query = select(ModelResult).where(ModelResult.run_id == run_id)
        if condition is not None:
            query = query.where(ModelResult.condition_id == condition)
        if status is not None:
            query = query.where(ModelResult.status == status)
        rows = session.exec(query.order_by(ModelResult.id).offset(offset).limit(limit)).all()
        return [{**ModelResultRead.model_validate(row).model_dump(),
            "image_url": f"/runs/{run_id}/results/{row.id}/image",
            "map_url": f"/runs/{run_id}/results/{row.id}/map" if row.output_type == "imp_segmentation" and row.status == "ok" else None} for row in rows]

    @application.get("/runs/{run_id}/results/{result_id}/{asset}", tags=["runs"])
    def result_asset(run_id: str, result_id: int, asset: str, session: SessionDep):
        run = find_run(session, run_id)
        row = get_record(session, ModelResult, result_id)
        if row.run_id != run_id:
            raise HTTPException(404, "Result does not belong to the run")
        try:
            if asset == "image":
                from .settings import orbital_path
                path = orbital_path(row.image_paths[0])
            elif asset == "map" and row.output_type == "imp_segmentation" and row.status == "ok":
                from .worker import map_artifacts
                artifacts = map_artifacts(row.raw_response, Path(run.output_dir))
                path = safe_path(artifacts["label_png"], Path(run.output_dir) / "maps")
            else:
                raise HTTPException(404, "Asset not available")
            if not path.is_file():
                raise HTTPException(404, "Asset file is missing")
            return FileResponse(path)
        except (ValueError, KeyError, IndexError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    @application.post("/runs/{run_id}/metrics", response_model=MetricAnalysisRead, tags=["evaluation"])
    def metrics(run_id: str, session: SessionDep):
        return calculate(session, run_id)

    @application.get("/runs/{run_id}/report", tags=["evaluation"])
    def run_report(run_id: str, session: SessionDep):
        return report(session, run_id)

    @application.post("/comparisons", tags=["evaluation"])
    def comparison(payload: ComparisonCreate, session: SessionDep):
        return compare(session, payload.run_ids)

    return application


def run_view(run):
    return {**run.model_dump(exclude={"samples"}), "run_id": run.id,
        "source_sample_ids": [row["sample_id"] for row in run.samples],
        "progress": {"completed": run.completed, "total": run.total,
            "percent": round(100 * run.completed / run.total, 2)},
        "status_url": f"/runs/{run.id}", "results_url": f"/runs/{run.id}/results",
        "report_url": f"/runs/{run.id}/report"}


def import_saved_run(payload, session):
    """Import already saved responses atomically; never invoke inference."""
    from planetary_vlm.datasets.manifest import load_requests
    from planetary_vlm.inference.runner import file_hash
    from planetary_vlm.io import read_jsonl, records_fingerprint
    from dataclasses import asdict

    model = get_record(session, ModelConfig, payload.model_config_id)
    dataset = get_record(session, DatasetConfig, payload.dataset_config_id)
    if model.name not in {"SpaceLLaVA", "NASA-IBM-Lunar-Foundation-Model"}:
        raise HTTPException(422, "Saved run import requires one of the two configured output folders")
    output_root = (WORKSPACE_ROOT / "outputs" / model.name).resolve()
    folder = (output_root / payload.run_id).resolve()
    if not output_root.is_relative_to(WORKSPACE_ROOT / "outputs") or not folder.is_relative_to(output_root):
        raise HTTPException(422, "Invalid output directory")
    try:
        metadata = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        requests = load_requests(folder / "requests.jsonl")
        predictions = read_jsonl(folder / "predictions.jsonl")
        if (metadata["model_id"] != model.model_id or metadata["backend"] != model.backend
                or metadata["run_id"] != payload.run_id or metadata["seed"] != model.seed
                or metadata["options"] != model.options):
            raise ValueError("Saved run does not match the selected model configuration")
        if not dataset.requests_path:
            raise ValueError("Dataset requests_path is required for run import")
        configured_requests = load_requests(dataset.requests_path)
        if any(item.max_new_tokens != model.max_new_tokens for item in requests):
            raise ValueError("Generation limit does not match the selected model configuration")
        digest = records_fingerprint(asdict(item) for item in requests)
        if digest != records_fingerprint(asdict(item) for item in configured_requests) or digest != metadata["request_manifest_sha256"]:
            raise ValueError("Saved requests do not match the selected dataset configuration")
        loading_path = folder / "loading.json"
        loading = json.loads(loading_path.read_text(encoding="utf-8")) if loading_path.is_file() else None
        provenance_path = Path(dataset.requests_path).parent / "provenance.jsonl"
        provenance = {row["request_id"]: row for row in read_jsonl(provenance_path)} if provenance_path.is_file() else {}
        by_id = {item.request_id: item for item in requests}
        seen, inserted, skipped = set(), 0, 0
        for row in predictions:
            request_id = row["request_id"]
            if request_id in seen or request_id not in by_id:
                raise ValueError("Duplicate or unknown saved request ID")
            seen.add(request_id)
            if (row["run_id"] != payload.run_id or row["model_id"] != model.model_id
                    or row["request_manifest_sha256"] != digest or row["mock"] != metadata["mock"]):
                raise ValueError("Saved response has inconsistent run provenance")
            artifacts = {}
            if model.backend == "ibm_imp" and row["status"] == "ok":
                artifacts = json.loads(row["raw_response"])
                maps_root = Path(artifacts["map_directory"]).resolve()
                if maps_root != (folder / "maps").resolve():
                    raise ValueError("IBM maps must belong to this saved run")
                for name, expected in artifacts["files_sha256"].items():
                    path = (maps_root / name).resolve()
                    if not path.is_relative_to(folder) or file_hash(path) != expected:
                        raise ValueError("A saved IBM map is missing or has changed")
            request = by_id[request_id]
            source = provenance.get(request_id, {})
            result_payload = ModelResultCreate(model_config_id=model.id, dataset_config_id=dataset.id,
                run_id=payload.run_id, request_id=request_id, source_sample_id=source.get("source_sample_id"),
                scene_group_id=source.get("scene_group_id"), source_split=source.get("source_split"),
                condition_id=source.get("condition", request_id.rsplit("__", 1)[-1]),
                image_paths=list(request.image_paths), prompt=request.prompt,
                output_type="imp_segmentation" if model.backend == "ibm_imp" else "text",
                status=row["status"], raw_response=row["raw_response"], elapsed_seconds=row["elapsed_seconds"],
                error_message=row.get("error_message"), artifacts=artifacts, mock=row["mock"],
                run_metadata={"run": metadata, "loading": loading, "source": source})
            validate_result_scope(session, result_payload)
            existing = session.exec(select(ModelResult).where(
                ModelResult.model_config_id == model.id, ModelResult.dataset_config_id == dataset.id,
                ModelResult.run_id == payload.run_id, ModelResult.request_id == request_id)).first()
            if existing:
                if ModelResultRead.model_validate(existing).model_dump(exclude={"id", "created_at"}) != result_payload.model_dump():
                    raise ValueError("Previously imported result differs from the saved response")
                skipped += 1
            else:
                session.add(ModelResult.model_validate(result_payload))
                inserted += 1
        session.commit()
        return {"inserted": inserted, "skipped": skipped, "run_id": payload.run_id}
    except FileNotFoundError as error:
        session.rollback()
        raise HTTPException(404, "Saved run files are missing") from error
    except (ValueError, KeyError, TypeError, OSError, IntegrityError) as error:
        session.rollback()
        raise HTTPException(422, f"Cannot import saved run: {error}") from error


app = create_app()
