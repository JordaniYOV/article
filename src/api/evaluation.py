"""Post-inference evaluation, condition reports and matched-run comparisons."""

from collections import Counter
import json
from pathlib import Path
from statistics import mean, median

from fastapi import HTTPException
from sqlmodel import select

from planetary_vlm.evaluation.metrics import classification_metrics, paired_accuracy_bootstrap
from planetary_vlm.evaluation.parser import parse_answer
from . import settings as paths
from .datasets import digest, file_hash, safe_path
from .models import DatasetConfig, MetricAnalysis, ModelConfig, ModelResult
from .runs import find_run
from .transforms import apply_chain, variants

METHOD = "orbital-api-metrics-v1"


def run_results(session, run_id):
    return session.exec(select(ModelResult).where(ModelResult.run_id == run_id).order_by(ModelResult.id)).all()


def classification(rows, sources, run):
    labels = run.protocol["allowed_answers"]
    scored = []
    for row in rows:
        source = sources[row.source_sample_id]
        answer = source.get("answer")
        if answer is None:
            continue
        canonical = parse_answer(answer, labels)
        if canonical is None:
            raise ValueError("A ground-truth answer is outside the declared answer labels")
        parsed = parse_answer(row.raw_response, labels) if row.status == "ok" else None
        scored.append({"request_id": row.request_id, "source_sample_id": row.source_sample_id,
            "group_id": row.scene_group_id, "answer": canonical, "parsed_answer": parsed,
            "correct": parsed == canonical, "scoring_status": "failure" if row.status != "ok" else (
                "valid" if parsed is not None else "invalid"), "allowed_answers": labels,
            "task_id": run.protocol["task_id"], "condition_id": row.condition_id,
            "derivation_version": source.get("derivation_version", "uploaded-answer-v1")})
    return classification_metrics(scored, labels) if scored else None, scored


def segmentation(rows, sources, run, chain):
    import numpy as np
    from PIL import Image
    from .worker import map_artifacts
    confusion = np.zeros((2, 2), dtype=np.int64)
    supported, successful, evaluated_pixels = 0, 0, 0
    for row in rows:
        source = sources[row.source_sample_id]
        if not source.get("mask"):
            continue
        supported += 1
        mask_path = paths.orbital_path(source["mask"])
        if not source.get("mask_sha256") or file_hash(mask_path) != source["mask_sha256"]:
            raise ValueError("Target mask hash missing or changed")
        if source.get("classes") != ["Background", "IMP"]:
            raise ValueError("IBM native metrics require source-local Background/IMP labels")
        with Image.open(mask_path) as image:
            target = np.array(image)
        if target.ndim != 2 or target.dtype.kind not in "uib" or not np.isin(target, [0, 1, 255]).all():
            raise ValueError("Target must be a 2D IMP class mask (0/1; 255 ignored)")
        target = target.astype(np.uint8)
        with Image.open(paths.orbital_path(source["image"])) as image:
            pixels = np.array(image)
        _, aligned = apply_chain(pixels, chain, run.protocol["parameters"], run.protocol["seed"],
            source["sample_id"], mask=target)
        if row.status != "ok":
            continue
        artifacts = map_artifacts(row.raw_response, Path(run.output_dir))
        predicted_path = safe_path(artifacts["label_png"], Path(run.output_dir) / "maps")
        with Image.open(predicted_path) as image:
            predicted = np.array(image)
        if predicted.shape != aligned.shape or not np.isin(predicted, [0, 1]).all():
            raise ValueError("Prediction shape or class IDs do not match ground truth")
        valid = aligned != 255
        count = int(valid.sum())
        if count == 0:
            continue
        successful += 1
        evaluated_pixels += count
        confusion += np.bincount((aligned[valid].astype(np.int64) * 2 + predicted[valid]).ravel(), minlength=4).reshape(2, 2)
    if not supported:
        return None
    per_class = {}
    for i, name in enumerate(["Background", "IMP"]):
        tp = int(confusion[i, i])
        fp, fn = int(confusion[:, i].sum()) - tp, int(confusion[i].sum()) - tp
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    supported_iou = [item["iou"] for item in per_class.values() if item["iou"] is not None]
    return {"n_labelled_images": supported, "n_scored_images": successful,
        "prediction_coverage": successful / supported, "n_evaluated_pixels": evaluated_pixels,
        "pixel_accuracy": int(confusion.trace()) / evaluated_pixels if evaluated_pixels else None,
        "mean_iou": mean(supported_iou) if supported_iou else None,
        "per_class": per_class, "confusion_matrix": confusion.tolist(),
        "denominator_policy": "successful maps with valid aligned pixels only; failures reported in coverage",
        "geometry_policy": "same shear field; nearest-neighbor labels; ignore exterior 255"}


def calculate(session, run_id):
    run = find_run(session, run_id)
    if run.status != "completed":
        raise HTTPException(409, "Complete the run before calculating metrics")
    results = run_results(session, run_id)
    if len(results) != run.total or len({row.mock for row in results}) != 1:
        raise HTTPException(409, "Run responses are incomplete or mix fixture and real outputs")
    model = session.get(ModelConfig, run.model_config_id)
    if any((row.model_config_id, row.dataset_config_id, row.output_type, row.mock) !=
        (run.model_config_id, run.dataset_config_id, "text" if run.kind == "vlm" else "imp_segmentation", model.backend == "mock") for row in results):
        raise HTTPException(409, "Response scope or provenance differs from the run")
    sources = {row["sample_id"]: row for row in run.samples}
    signature = digest([row.model_dump() for row in results])
    previous = session.exec(select(MetricAnalysis).where(MetricAnalysis.run_id == run_id,
        MetricAnalysis.method_version == METHOD, MetricAnalysis.results_sha256 == signature)).first()
    # For segmentation, targets/maps are reverified even for an idempotent call.
    try:
        dataset = session.get(DatasetConfig, run.dataset_config_id)
        if file_hash(dataset.manifest_path) != run.protocol["manifest_sha256"]:
            raise ValueError("Source manifest changed after inference")
        for source in sources.values():
            if file_hash(paths.orbital_path(source["image"])) != source["image_sha256"]:
                raise ValueError("Source image changed after inference")
        for row in results:
            if file_hash(paths.orbital_path(row.image_paths[0])) != row.run_metadata["input_sha256"]:
                raise ValueError("Prepared inference input changed after inference")
        conditions, scored_rows = {}, []
        for condition, chain in variants(run.protocol):
            selected = [row for row in results if row.condition_id == condition]
            if len(selected) != len(sources) or {row.source_sample_id for row in selected} != set(sources):
                raise ValueError("Each condition must contain exactly the selected source images")
            status = Counter(row.status for row in selected)
            timing = [row.elapsed_seconds for row in selected if row.status == "ok"]
            runtime = {"n_expected": len(sources), "n_responses": len(selected),
                "n_ok": status["ok"], "n_error": status["error"], "n_unsupported": status["unsupported"],
                "success_rate": status["ok"] / len(sources),
                "mean_seconds": mean(timing) if timing else None,
                "median_seconds": median(timing) if timing else None,
                "timing_population": "successful requests, initialization excluded"}
            if run.kind == "vit":
                quality = segmentation(selected, sources, run, chain)
                family = "imp_segmentation" if quality is not None else None
            elif run.protocol["allowed_answers"]:
                quality, scored = classification(selected, sources, run)
                scored_rows.extend(scored)
                family = "closed_form_classification" if quality is not None else None
            else:
                quality, family = None, None
            conditions[condition] = {"runtime": runtime, "quality": quality,
                "metric_family": family, "quality_status": "available" if quality is not None else "unavailable",
                "quality_reason": None if quality is not None else "Declared labels and matching ground truth are required; prose is not automatically scored"}
        metric = "accuracy" if run.kind == "vlm" else "mean_iou"
        baseline = (conditions.get("clean", {}).get("quality") or {}).get(metric)
        chart = []
        for condition, item in conditions.items():
            score = (item["quality"] or {}).get(metric)
            item["change_from_clean"] = score - baseline if score is not None and baseline is not None else None
            chart.append({"condition": condition, "metric": metric, "value": score,
                "change_from_clean": item["change_from_clean"], "mean_seconds": item["runtime"]["mean_seconds"]})
        metrics = {"conditions": conditions, "chart": chart, "scored_rows": scored_rows,
            "n_source_images": len(sources), "n_scene_groups": len({row["scene_group_id"] for row in sources.values()}),
            "protocol_sha256": run.protocol_sha256, "task_id": run.protocol["task_id"],
            "planet": run.protocol["planet"], "split": run.protocol["split"],
            "experiments": {
                "E1": {"status": "available" if "clean" in conditions else "not_selected", "condition": "clean"},
                "E2": {"status": "available", "conditions": [c for c in conditions if c != "clean"]},
                "E3": {"status": "derived" if scored_rows else "unavailable", "source": "E1/E2 closed-form confusion matrices"},
                "E4": {"status": "unsupported", "reason": "Requires aligned measured terrain and a fixed route-cost protocol"},
                "E5": {"status": "requires_matched_runs", "reason": "This run covers one declared planet and sensor dataset"}}}
        record = MetricAnalysis(model_config_id=run.model_config_id, dataset_config_id=run.dataset_config_id,
            run_id=run.id, task_id=run.protocol["task_id"], name="evaluation", method_version=METHOD,
            result_ids=[row.id for row in results], metrics=metrics, results_sha256=signature,
            mock=results[0].mock, notes="Fixture outputs are not scientific scores" if results[0].mock else "")
        if previous:
            record = previous
        else:
            session.add(record)
            session.commit()
            session.refresh(record)
        from .worker import atomic_json
        atomic_json(Path(run.output_dir) / "metrics.json", record.model_dump())
        return record
    except (OSError, ValueError, KeyError, TypeError) as error:
        session.rollback()
        raise HTTPException(422, f"Cannot evaluate: {error}") from error


def report(session, run_id):
    run = find_run(session, run_id)
    model = session.get(ModelConfig, run.model_config_id)
    dataset = session.get(DatasetConfig, run.dataset_config_id)
    analyses = session.exec(select(MetricAnalysis).where(MetricAnalysis.run_id == run_id,
        MetricAnalysis.method_version == METHOD).order_by(MetricAnalysis.id.desc())).all()
    analysis = analyses[0] if analyses else None
    return {"run_id": run.id, "status": run.status, "phase": run.phase,
        "progress": {"completed": run.completed, "total": run.total}, "error": run.error_message,
        "model": {"name": model.name, "version": model.version, "backend": model.backend},
        "dataset": {"id": dataset.id, "name": dataset.name, "planet": dataset.planet, "task_id": dataset.task_id},
        "mock": model.backend == "mock", "metrics_status": "calculated" if analysis else "not_calculated",
        "analysis_id": analysis.id if analysis else None, "metrics": analysis.metrics if analysis else None,
        "protocol_sha256": run.protocol_sha256,
        "results_url": f"/runs/{run.id}/results", "output_directory": run.output_dir}


def compare(session, run_ids):
    left, right = (find_run(session, key) for key in run_ids)
    if left.protocol_sha256 != right.protocol_sha256:
        raise HTTPException(422, "Comparison requires identical source selection, seed, split, prompt and distortion protocol")
    if left.status != "completed" or right.status != "completed":
        raise HTTPException(409, "Both runs must be completed")
    reports = [report(session, run.id) for run in (left, right)]
    if any(item["metrics"] is None for item in reports):
        raise HTTPException(409, "Calculate metrics for both runs first")
    if reports[0]["mock"] != reports[1]["mock"]:
        raise HTTPException(422, "Fixture and real model outputs cannot be compared")
    contrasts = []
    comparable = left.kind == right.kind
    for condition, _ in variants(left.protocol):
        a, b = (item["metrics"]["conditions"][condition] for item in reports)
        family = a["metric_family"]
        valid = comparable and family is not None and family == b["metric_family"]
        metric = "accuracy" if family == "closed_form_classification" else "mean_iou"
        x, y = (item["quality"].get(metric) if item["quality"] else None for item in (a, b))
        contrast = {"condition": condition, "comparable": valid and x is not None and y is not None,
            "metric": metric if valid else None,
            "right_minus_left": y - x if valid and x is not None and y is not None else None}
        if valid and family == "closed_form_classification":
            paired = [[row for row in item["metrics"]["scored_rows"] if row["condition_id"] == condition] for item in reports]
            contrast["paired_interval"] = paired_accuracy_bootstrap(*paired, seed=left.protocol["seed"])
        contrasts.append(contrast)
    return {"runs": reports, "contrasts": contrasts, "mock": reports[0]["mock"],
        "quality_comparable": all(item["comparable"] for item in contrasts),
        "note": "Native segmentation and text QA scores have separate definitions; no shared quality ranking" if not comparable else None}
