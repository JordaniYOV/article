"""Import CLI-trained deployments and score their maps without importing Torch."""
import json
from pathlib import Path
from collections import defaultdict

from fastapi import HTTPException
from sqlmodel import select

from planetary_vlm.heads_contract import TASK, INPUT_PROTOCOL, CLASSES, read_deployment
from benchmark_services.common import settings as paths
from benchmark_services.common.files import file_hash
from benchmark_services.common.store import ModelConfig, ModelConfigCreate, DatasetConfig, DatasetConfigCreate


def import_deployment(payload, session, fingerprint):
    try:
        path, deployment = read_deployment(payload.deployment_path, payload.sha256, paths.WORKSPACE_ROOT)
        manifest = paths.orbital_path(deployment["benchmark_manifest"])
        if file_hash(manifest) != deployment["benchmark_sha256"]:
            raise ValueError("CLI-exported benchmark manifest changed")
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        from collections import Counter
        counts = dict(Counter(r["source_split"] for r in records))
        if not records or not set(counts) <= {"val", "test"}:
            raise ValueError("Head benchmark manifest must contain held-out val/test only")
        if any(r.get("mask_semantics") != "catalog_crater_polygon" or r.get("classes") != CLASSES for r in records):
            raise ValueError("Head benchmark manifest must preserve publisher catalog mask semantics")
        model = ModelConfigCreate(name=payload.name, version=payload.version, model_id=f"frozen/{deployment['encoder']}/{deployment['family']}",
            backend="frozen_crater", enabled=True, revision=deployment["files_sha256"]["head.pt"], seed=deployment["seed"], max_new_tokens=1,
            options={"deployment_path": str(path), "deployment_sha256": payload.sha256, "local_files_only": True, "input_protocol": INPUT_PROTOCOL})
        dataset = DatasetConfigCreate(name="WAC_catalog_craters_heads", version="v1", planet="moon", task_id=TASK,
            manifest_path=str(manifest), splits=["val", "test"], conditions=["clean"], preprocessing={"input_protocol": INPUT_PROTOCOL},
            provenance={"source_dataset": "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection", "split_counts": counts,
                        "sample_count": len(records), "classes": CLASSES, "reference_scope": "publisher catalog polygons, not exhaustive visible-crater GT",
                        "input_warning": "IBM VIS5 vs CLIP RGB: not equal-information", "manifest_sha256": deployment["benchmark_sha256"]})
        ids = {}
        for table, value, key in ((ModelConfig, model, "model_id"), (DatasetConfig, dataset, "dataset_id")):
            values = value.model_dump()
            hashed = fingerprint(values)
            existing = session.exec(select(table).where(table.name == value.name, table.version == value.version)).first()
            if existing and existing.config_sha256 != hashed:
                raise ValueError("An existing configuration has different content; choose a new deployment name/version")
            record = existing or table.model_validate(values | {"config_sha256": hashed})
            session.add(record)
            session.flush()
            ids[key] = record.id
        session.commit()
        return ids | {"backend": "frozen_crater", "clean_only": True, "training_started": False}
    except (ValueError, KeyError, OSError) as error:
        session.rollback()
        raise HTTPException(422, str(error)) from error


def score_maps(results, sources, run, chain):
    import numpy as np
    from PIL import Image
    from benchmark_services.common.execution import map_artifacts
    from benchmark_services.common.files import safe_path
    if chain:
        raise ValueError("No corruption protocol admitted for native VIS5 vs RGB heads")
    per_image, withheld = [], 0
    for row in results:
        source = sources[row.source_sample_id]
        if source.get("reference_status") != "ready":
            withheld += 1
            continue
        if source.get("mask_semantics") != "catalog_crater_polygon" or source.get("classes") != CLASSES:
            raise ValueError("Catalog heads require catalog crater masks, not IMP/terrain labels")
        mask = paths.orbital_path(source["mask"])
        if file_hash(mask) != source["mask_sha256"]:
            raise ValueError("Crater target hash changed")
        with Image.open(mask) as image:
            target = np.asarray(image)
        if target.shape != (512, 512) or not np.isin(target, [0, 1]).all():
            raise ValueError("Expected binary native WAC catalog target")
        if row.status != "ok":
            continue
        artifacts = map_artifacts(row.raw_response, Path(run.output_dir))
        if artifacts["output_type"] != TASK:
            raise ValueError("Another task's map cannot be scored as a crater map")
        with Image.open(safe_path(artifacts["label_png"], Path(run.output_dir) / "maps")) as image:
            predicted = np.asarray(image)
        if predicted.shape != target.shape or not np.isin(predicted, [0, 1]).all():
            raise ValueError("Invalid crater prediction")
        predicted, target = predicted.astype(bool), target.astype(bool)
        tp, fp, fn = int((predicted & target).sum()), int((predicted & ~target).sum()), int((~predicted & target).sum())
        per_image.append({"sample_id": source["sample_id"], "scene_group_id": source["scene_group_id"],
                          "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.,
                          "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.})
    supported = len(results) - withheld
    if not supported:
        return None
    groups = defaultdict(list)
    for row in per_image:
        groups[row["scene_group_id"]].append(row)
    def macro(metric):
        return float(np.mean([np.mean([r[metric] for r in group]) for group in groups.values()])) if groups else None
    return {"crater_iou": macro("iou"), "crater_dice": macro("dice"), "n_labelled_images": supported,
            "n_scored_images": len(per_image), "n_unscored_reference_images": withheld, "prediction_coverage": len(per_image) / supported,
            "reference_coverage": supported / len(results), "per_image": per_image, "averaging": "image within scene, then scene macro",
            "reference_scope": "publisher catalog polygons; not a complete visible-crater census"}


def paired_scene_interval(left, right, seed):
    import numpy as np
    a = {(r["sample_id"], r["scene_group_id"]): r["iou"] for r in left}
    b = {(r["sample_id"], r["scene_group_id"]): r["iou"] for r in right}
    if not a or set(a) != set(b) or len(a) != len(left) or len(b) != len(right):
        raise ValueError("Paired crater CI requires identical unique image/group cohorts")
    groups = defaultdict(list)
    for key in a:
        groups[key[1]].append(b[key] - a[key])
    differences = np.array([np.mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(0, len(differences), (2000, len(differences)))].mean(1)
    return {"interval": np.quantile(samples, [.025, .975]).tolist(), "n_groups": len(differences),
            "warning": "Scene bootstrap for this pair of trained seeds; native VIS5 vs RGB is not an equal-information specialization effect"}
