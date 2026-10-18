"""Register local external crater masks without changing trained deployments."""
from collections import Counter

from fastapi import HTTPException
from sqlmodel import select

from benchmark_services.common import settings as paths
from benchmark_services.common.files import file_hash
from benchmark_services.common.store import DatasetConfig, DatasetConfigCreate
from planetary_vlm.heads_contract import TASK, TRANSFER_PROTOCOL, CRATER_SEMANTICS, transfer_input
from .datasets import read_rows


def register_crater_collection(session, fingerprint):
    """Idempotent, atomic registration; no download, inference or target edits."""
    import numpy as np
    from PIL import Image
    saved = []
    try:
        for planet in ("moon", "mars"):
            manifest = paths.orbital_path(f"data_orbital/crater_seg_100_v1/{planet}/manifest.jsonl")
            records = read_rows(manifest)
            if not records or len({r["sample_id"] for r in records}) != len(records):
                raise ValueError("Crater collection must have nonempty unique sample IDs")
            for row in records:
                if row.get("planet") != planet or row.get("mask_semantics") not in CRATER_SEMANTICS:
                    raise ValueError("Expected source-declared crater masks for the selected planet")
                if row.get("classes") != ["Background", "Crater"]:
                    raise ValueError("Expected source classes Background/Crater")
                transfer_input(row, paths.WORKSPACE_ROOT)
                mask = paths.orbital_path(row["mask"])
                if file_hash(mask) != row["mask_sha256"]:
                    raise ValueError("Crater collection mask SHA-256 changed")
                with Image.open(mask) as image:
                    target = np.asarray(image)
                if target.shape != (512, 512) or not np.isin(target, [0, 1]).all():
                    raise ValueError("Expected a binary 512x512 crater mask")
            counts = dict(Counter(r["source_split"] for r in records))
            payload = DatasetConfigCreate(name=f"Crater_seg_100_v1_{planet}", version="v1", planet=planet,
                task_id=TASK, manifest_path=str(manifest), splits=sorted(counts), conditions=["clean"],
                preprocessing={"input_protocol": TRANSFER_PROTOCOL}, provenance={
                    "collection": "crater_seg_100_v1", "sample_count": len(records), "split_counts": counts,
                    "manifest_sha256": file_hash(manifest), "native_vis5_available": all(r.get("native_vis") for r in records),
                    "reference_scope": "publisher crater masks; not an exhaustive visible-crater census",
                    "warning": "Mars shares parent mosaics with orbital_300_v3; source strips remain; no regional independence claimed"})
            values = payload.model_dump()
            hashed = fingerprint(values)
            existing = session.exec(select(DatasetConfig).where(DatasetConfig.name == payload.name,
                                                               DatasetConfig.version == payload.version)).first()
            if existing and existing.config_sha256 != hashed:
                raise ValueError("Registered crater collection changed; create a new dataset version")
            record = existing or DatasetConfig.model_validate(values | {"config_sha256": hashed})
            session.add(record)
            session.flush()
            saved.append(record)
        session.commit()
        for record in saved:
            session.refresh(record)
        return saved
    except (ValueError, KeyError, OSError) as error:
        session.rollback()
        raise HTTPException(422, f"Cannot register local crater_seg_100_v1: {error}") from error
