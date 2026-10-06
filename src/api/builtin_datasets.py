"""Persistent Mars/Moon catalog, local registration and background restoration."""
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import HTTPException
from sqlmodel import Session, select

from planetary_vlm.datasets.orbital import (
    COLLECTION, LABELS, checked_path, download_dataset, inspect_subset, validate_selection, verify_assets,
)
from . import settings as paths
from .datasets import digest
from .models import DatasetConfig, DatasetConfigCreate
from .models import utc_now

MUTEX = RLock()


def job_folder():
    return paths.WORKSPACE_ROOT / ".state/builtin-datasets"


def save_job(job):
    folder = job_folder()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{job['id']}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)


def read_job(job_id):
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise HTTPException(404, "Dataset preparation not found")
    target = job_folder() / f"{job_id}.json"
    if not target.is_file():
        raise HTTPException(404, "Dataset preparation not found")
    return json.loads(target.read_text(encoding="utf-8"))


def jobs():
    folder = job_folder()
    return sorted([json.loads(path.read_text(encoding="utf-8")) for path in folder.glob("*.json")], key=lambda j: j.get("created_at", "")) if folder.exists() else []


def recover_jobs():
    # Background downloads do not survive API restarts. Retrying reuses validated
    # assets; interrupted tasks must never look like completed preparations.
    for job in jobs():
        if job["status"] in {"queued", "downloading"}:
            job.update(status="failed", error="Сервер был перезапущен. Выберите датасет ещё раз для продолжения.")
            save_job(job)


def config_name(planet, subset):
    return f"{COLLECTION}_{planet}_{subset}"


def register(session, planet, subset):
    info = inspect_subset(paths.WORKSPACE_ROOT, planet, subset)
    if not info["ready"]:
        raise ValueError("Selected orbital subset is incomplete")
    rows = info["rows"]
    verify_assets(paths.WORKSPACE_ROOT, rows)
    normalized = [{**row, **({"dem": row["height_map"]} if row.get("height_map") else {})} for row in rows]
    checksum = digest(normalized)
    manifest = checked_path(paths.WORKSPACE_ROOT,
        f"data_orbital/inference/builtin/{planet}/{subset}/{checksum}/manifest.jsonl")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized)
    if manifest.exists() and manifest.read_text(encoding="utf-8") != content:
        raise ValueError("Registered orbital manifest has changed")
    if not manifest.exists():
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, manifest)
    payload = DatasetConfigCreate(name=config_name(planet, subset), version=f"v3-{checksum[:16]}",
        planet=planet, task_id="imp_segmentation" if subset == "imp" else "elevation_map_reading" if subset == "height" else "orbital_features",
        manifest_path=str(manifest), splits=sorted(info["split_counts"]),
        preprocessing={"input_protocol": "curated_imp_png_reflectance_0_0.12_v1" if subset == "imp" else "orbital_image_only_v1",
            "target_type": "height" if subset == "height" else "mask"},
        provenance={"builtin_collection": COLLECTION, "builtin_subset": subset,
            "source_manifest": info["manifest"].relative_to(paths.WORKSPACE_ROOT).as_posix(),
            "sample_count": info["sample_count"], "split_counts": info["split_counts"],
            "source_revisions": sorted({(r["source_dataset"], r["source_revision"]) for r in rows}),
            "license_claims": sorted({r.get("license_claim", "unknown") for r in rows}),
            "selection_sha256": checksum})
    values = payload.model_dump()
    existing = session.exec(select(DatasetConfig).where(DatasetConfig.name == payload.name, DatasetConfig.version == payload.version)).first()
    if existing:
        if existing.config_sha256 != digest(values):
            raise ValueError("Builtin configuration changed; its frozen version cannot be reused")
        return existing
    record = DatasetConfig.model_validate({**values, "config_sha256": digest(values)})
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def catalog(session):
    records = session.exec(select(DatasetConfig)).all()
    previous = jobs()
    result = []
    for planet, label in (("mars", "Марс"), ("moon", "Луна")):
        subsets = []
        for subset in ("segmentation", "height", *(["imp"] if planet == "moon" else [])):
            entry = {"id": subset, "label": LABELS[subset], "status": "missing", "dataset_id": None,
                "job_id": None, "completed": 0, "total": 0, "error": None, "sample_count": 0, "split_counts": {}}
            try:
                info = inspect_subset(paths.WORKSPACE_ROOT, planet, subset)
                checksum = digest([{**r, **({"dem": r["height_map"]} if r.get("height_map") else {})} for r in info["rows"]])
                entry.update(sample_count=info["sample_count"], split_counts=info["split_counts"], total=info["sample_count"],
                    status="ready" if info["ready"] else "missing")
                matching = next((r for r in records if r.name == config_name(planet, subset) and r.version == f"v3-{checksum[:16]}"), None)
                if matching and info["ready"]:
                    entry["dataset_id"] = matching.id
                active = next((j for j in reversed(previous) if j["planet"] == planet and j["subset"] == subset and j["status"] in {"queued", "downloading"}), None)
                if active:
                    entry.update(status=active["status"], job_id=active["id"], completed=active["completed"], total=active["total"])
                elif not info["ready"]:
                    failed = next((j for j in reversed(previous) if j["planet"] == planet and j["subset"] == subset and j["status"] == "failed"), None)
                    if failed:
                        entry.update(status="failed", error=failed["error"])
            except (ValueError, OSError, KeyError) as error:
                entry.update(status="failed", error=str(error))
            subsets.append(entry)
        result.append({"planet": planet, "label": label, "subsets": subsets})
    return result


def ensure(planet, subset, session, tasks, engine):
    try:
        validate_selection(planet, subset)
        with MUTEX:
            info = inspect_subset(paths.WORKSPACE_ROOT, planet, subset)
            if info["ready"]:
                record = register(session, planet, subset)
                return {"status": "ready", "dataset": record.model_dump(), "job_id": None}
            active = next((j for j in jobs() if j["planet"] == planet and j["status"] in {"queued", "downloading"}), None)
            if active:
                if active["subset"] != subset:
                    raise HTTPException(409, "Сначала дождитесь подготовки другой подвыборки этой планеты.")
                return {"status": active["status"], "dataset": None, "job_id": active["id"]}
            job = {"id": uuid4().hex, "planet": planet, "subset": subset, "status": "queued",
                "completed": 0, "total": info["sample_count"], "dataset_id": None, "error": None,
                "created_at": utc_now().isoformat()}
            save_job(job)
            tasks.add_task(prepare, job, engine)
            return {"status": "queued", "dataset": None, "job_id": job["id"]}
    except (ValueError, OSError, KeyError) as error:
        raise HTTPException(422, str(error)) from error


def prepare(job, engine):
    try:
        job.update(status="downloading")
        save_job(job)
        def progress(done, total):
            job.update(completed=done, total=total)
            save_job(job)
        download_dataset(paths.WORKSPACE_ROOT, job["planet"], job["subset"], progress)
        with Session(engine) as session:
            record = register(session, job["planet"], job["subset"])
            job.update(status="ready", dataset_id=record.id)
    except Exception as error:
        job.update(status="failed", error=f"Не удалось подготовить датасет: {error}")
    save_job(job)
