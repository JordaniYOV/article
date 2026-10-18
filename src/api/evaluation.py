"""Post-inference evaluation by output/task contract, never by architecture.

Text labels from a VLM or an encoder's declared readout share the same scorer.
Native IMP maps remain a separate metric family; missing GT is never NO.
"""

from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, median

from fastapi import HTTPException
from sqlmodel import select

from planetary_vlm.evaluation import evaluate
from planetary_vlm.evaluation.metrics import paired_accuracy_bootstrap
from planetary_vlm.evaluation.references import READY, UNKNOWN, reference_status
from planetary_vlm.datasets.crater_reference import TASKS as CRATER_TASKS
from benchmark_services.common import settings as paths
from benchmark_services.common.files import digest, file_hash, safe_path
from benchmark_services.common.store import DatasetConfig, MetricAnalysis, ModelConfig, ModelResult
from .runs import find_run
from benchmark_services.common.transforms import apply_chain, variants
from benchmark_services.common.preparation import renderer_provenance, load_bundle

METHOD = "orbital-api-metrics-v2"
METRICS = {"closed_form_classification": "accuracy", "imp_segmentation": "mean_iou", "catalog_crater_segmentation": "crater_iou"}


def run_is_mock(run, model):
    # Service fixtures retain the configured real backend name, but the worker
    # persists their actual execution mode in the authoritative SQLite journal.
    return model.backend == "mock" or run.loading.get("mock", False) is True


def run_results(session, run_id):
    return session.exec(select(ModelResult).where(ModelResult.run_id == run_id).order_by(ModelResult.id)).all()


def classification(rows, sources, run):
    """Translate API rows into the same model-independent evaluator as CLI."""
    labels = run.protocol["allowed_answers"]
    requests, targets, predictions = [], [], []
    for row in rows:
        source = sources[row.source_sample_id]
        target = dict(source)
        # Spatial/size eligibility can change under shear. Original-image
        # crater answers cannot be silently reused for transformed geometry.
        condition_answers = source.get("condition_answers", {})
        if not isinstance(condition_answers, dict):
            raise ValueError("condition_answers must be an object keyed by condition ID")
        if row.condition_id in condition_answers:
            target["answer"] = condition_answers[row.condition_id]
            if target["answer"] is None:
                target["reference_status"] = UNKNOWN
        elif run.protocol["task_id"] in CRATER_TASKS and "shear" in row.condition_id.split("+"):
            target.update(answer=None, reference_status=UNKNOWN)
            target.pop("review_status", None)
        admission, _ = reference_status(target, has_reference=target.get("answer") is not None,
            require_explicit=run.protocol["task_id"] in CRATER_TASKS)
        requests.append({"request_id": row.request_id, "image_paths": list(row.image_paths),
            "prompt": row.prompt, "allowed_answers": labels, "max_new_tokens": run.protocol["max_new_tokens"],
            "do_sample": False})
        targets.append({"request_id": row.request_id, "answer": target.get("answer"),
            "source_sample_id": source["sample_id"], "group_id": source["scene_group_id"],
            "task_id": run.protocol["task_id"], "condition_id": row.condition_id,
            "derivation_version": source.get("derivation_version", "uploaded-answer-v1"),
            "reference_status": admission})
        predictions.append({"request_id": row.request_id, "raw_response": row.raw_response,
            "status": row.status, "elapsed_seconds": row.elapsed_seconds, "error_message": row.error_message,
            "mock": row.mock, "run_id": row.run_id})
    result = evaluate(requests, targets, predictions)
    quality = result["overall"] if result["overall"]["n_scored"] else None
    return quality, result["per_request"]


def segmentation(rows, sources, run, chain):
    import numpy as np
    from PIL import Image
    from benchmark_services.common.execution import map_artifacts
    confusion = np.zeros((2, 2), dtype=np.int64)
    supported, successful, evaluated_pixels, withheld = 0, 0, 0, 0
    for row in rows:
        source = sources[row.source_sample_id]
        if not source.get("mask"):
            continue
        admission, _ = reference_status(source, has_reference=True)
        if admission != READY:
            withheld += 1
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
        "n_unscored_reference_images": withheld,
        "reference_coverage": supported / len(rows),
        "n_unlabelled_images": len(rows) - supported - withheld,
        "prediction_coverage": successful / supported, "n_evaluated_pixels": evaluated_pixels,
        "pixel_accuracy": int(confusion.trace()) / evaluated_pixels if evaluated_pixels else None,
        "mean_iou": mean(supported_iou) if supported_iou else None,
        "per_class": per_class, "confusion_matrix": confusion.tolist(),
        "denominator_policy": "successful maps with valid aligned pixels only; failures reported in coverage",
        "geometry_policy": "same shear field; nearest-neighbor labels; ignore exterior 255"}


def _signature(run, results):
    return digest({"method": METHOD, "protocol_sha256": run.protocol_sha256,
        "protocol": run.protocol, "samples": run.samples, "results": [row.model_dump() for row in results]})


def _verify_inputs(run, dataset, results):
    if run.protocol.get("renderer") is not None and run.protocol["renderer"] != renderer_provenance():
        raise ValueError("API renderer environment changed; restore it before evaluating targets")
    if digest({k: v for k, v in run.protocol.items() if k != "model_config_sha256"}) != run.protocol_sha256:
        raise ValueError("Frozen run protocol changed")
    if digest(run.samples) != run.protocol["selection_sha256"]:
        raise ValueError("Frozen source/target selection changed")
    if file_hash(dataset.manifest_path) != run.protocol["manifest_sha256"]:
        raise ValueError("Source manifest changed after inference")
    sources = {row["sample_id"]: row for row in run.samples}
    if not sources or len(sources) != len(run.samples):
        raise ValueError("Source sample IDs must be nonempty and unique")
    if any(not isinstance(s.get("scene_group_id"), str) or not s["scene_group_id"] for s in sources.values()):
        raise ValueError("A nonempty source scene group is required")
    requests, records, _ = load_bundle(run)
    expected = {row.request_id: row for row in requests}
    if len({row.request_id for row in results}) != len(results) or {row.request_id for row in results} != set(expected):
        raise ValueError("Responses must exactly match prepared request IDs")
    output_types = {row.output_type for row in results}
    if len(output_types) != 1 or not output_types <= {"text", "imp_segmentation", "catalog_crater_segmentation"}:
        raise ValueError("A run must contain one declared supported output contract")
    for row in results:
        request = expected[row.request_id]
        record = records[row.request_id]
        if (row.run_metadata != record or list(row.image_paths) != list(request.image_paths)
            or row.prompt != request.prompt or row.source_sample_id != record["source_sample_id"]
            or row.scene_group_id != record["scene_group_id"] or row.source_split != record["source_split"]
            or row.condition_id != record["condition_id"]):
            raise ValueError("Response identity/input/source provenance differs from the prepared request")
        if (row.status not in {"ok", "error", "unsupported"} or type(row.elapsed_seconds) not in (int, float)
            or not math.isfinite(row.elapsed_seconds) or row.elapsed_seconds < 0):
            raise ValueError("Invalid response status or elapsed time")
    return sources, next(iter(output_types))


def _experiments(conditions, task_id):
    def status(names):
        if not names:
            return "not_selected"
        states = [conditions[name]["quality_status"] for name in names]
        return "available" if all(s == "available" for s in states) else (
            "partial" if any(s in {"available", "partial"} for s in states) else "unavailable")
    corrupted = [c for c in conditions if c != "clean"]
    # A closed YES/NO task is not automatically a hallucination task.
    presence = task_id in {"crater_presence_v1", "presence"}
    proxies = {name: item["quality"]["binary"] for name, item in conditions.items()
               if presence and item["metric_family"] == "closed_form_classification"
               and item["quality"] and item["quality"].get("binary") is not None
               and item["quality"]["binary"]["fpr_denominator"] > 0}
    return {
        "E1": {"status": status(["clean"] if "clean" in conditions else []), "condition": "clean"},
        "E2": {"status": status(corrupted), "conditions": corrupted},
        "E3": {"status": "derived_proxy" if proxies else "unavailable", "conditions": proxies,
            "source": "E1/E2 presence-question false-positive rates, valid-answer denominators and coverage",
            "unsupported_claims_status": "not_evaluated",
            "reason": "A presence false positive is a restricted proxy, not a free-form hallucination judgement"},
        "E4": {"status": "unsupported", "reason": "Requires aligned measured terrain and a declared modality protocol"},
        "E5": {"status": "requires_matched_runs", "reason": "One run cannot establish specialization or planetary transfer"}}


def calculate(session, run_id):
    run = find_run(session, run_id)
    if run.status != "completed":
        raise HTTPException(409, "Complete the run before calculating metrics")
    results = run_results(session, run_id)
    if len(results) != run.total or len({row.mock for row in results}) != 1:
        raise HTTPException(409, "Run responses are incomplete or mix fixture and real outputs")
    model = session.get(ModelConfig, run.model_config_id)
    if model is None:
        raise HTTPException(409, "Run model configuration is missing")
    if any((row.model_config_id, row.dataset_config_id, row.mock) !=
        (run.model_config_id, run.dataset_config_id, run_is_mock(run, model)) for row in results):
        raise HTTPException(409, "Response scope or provenance differs from the run")
    signature = _signature(run, results)
    previous = session.exec(select(MetricAnalysis).where(MetricAnalysis.run_id == run_id,
        MetricAnalysis.method_version == METHOD, MetricAnalysis.results_sha256 == signature)).first()
    # For segmentation, targets/maps are reverified even for an idempotent call.
    try:
        dataset = session.get(DatasetConfig, run.dataset_config_id)
        if dataset is None:
            raise ValueError("Run dataset configuration is missing")
        sources, output_type = _verify_inputs(run, dataset, results)
        if output_type == "imp_segmentation" and run.protocol["task_id"] != "imp_segmentation":
            raise ValueError("Native IMP masks cannot answer crater or other closed-form questions")
        if output_type == "catalog_crater_segmentation" and run.protocol["task_id"] != output_type:
            raise ValueError("Catalog crater maps cannot answer closed-form questions or IMP tasks")
        conditions, scored_rows, per_request = {}, [], []
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
            reference = None
            if output_type == "catalog_crater_segmentation":
                from .heads import score_maps
                quality = score_maps(selected, sources, run, chain)
                family = output_type
                ready = sum(s.get("reference_status") == READY for s in sources.values())
                reference = {"n_expected": len(selected), "n_reference_ready": ready, "n_unscored_reference": len(selected) - ready}
            elif output_type == "imp_segmentation":
                quality = segmentation(selected, sources, run, chain)
                family = "imp_segmentation"
                ready = sum(bool(s.get("mask")) and reference_status(s, has_reference=True)[0] == READY
                            for s in sources.values())
                reference = {"n_expected": len(selected), "n_reference_ready": ready,
                             "n_unscored_reference": len(selected) - ready}
            elif run.protocol["allowed_answers"]:
                quality, evaluated = classification(selected, sources, run)
                per_request.extend(evaluated)
                scored_rows.extend(row for row in evaluated if row["scoring_status"] in {"valid", "invalid", "failure"})
                reference = {"n_expected": len(evaluated),
                    "n_reference_ready": sum(row["reference_status"] == READY for row in evaluated),
                    "n_unscored_reference": sum(row["reference_status"] != READY for row in evaluated)}
                family = "closed_form_classification"
            else:
                quality, family = None, None
            available = quality is not None and quality.get(METRICS.get(family)) is not None
            partial = available and ((reference is not None and reference["n_unscored_reference"] > 0)
                or (family in {"imp_segmentation", "catalog_crater_segmentation"} and
                    (quality["prediction_coverage"] < 1 or quality["reference_coverage"] < 1)))
            conditions[condition] = {"runtime": runtime, "quality": quality,
                "reference": reference, "metric_family": family,
                "quality_status": "partial" if partial else "available" if available else "unavailable",
                "quality_reason": None if available else "Admitted matching ground truth and supported outputs are required; unknown references and free-form claims are not scored"}
        chart = []
        for condition, item in conditions.items():
            metric = METRICS.get(item["metric_family"])
            clean = conditions.get("clean", {})
            same_family = clean.get("metric_family") == item["metric_family"]
            baseline = (clean.get("quality") or {}).get(metric) if same_family else None
            score = (item["quality"] or {}).get(metric)
            same_cohort = True
            if item["metric_family"] == "closed_form_classification":
                cohorts = [{r["source_sample_id"] for r in scored_rows if r["condition_id"] == name}
                           for name in ("clean", condition)]
                same_cohort = bool(cohorts[0]) and cohorts[0] == cohorts[1]
            item["change_from_clean"] = score - baseline if same_cohort and score is not None and baseline is not None else None
            item["clean_comparison_reason"] = None if same_cohort else "Different admitted reference cohorts; no clean delta"
            chart.append({"condition": condition, "metric": metric, "value": score,
                "change_from_clean": item["change_from_clean"], "mean_seconds": item["runtime"]["mean_seconds"]})
        metrics = {"conditions": conditions, "chart": chart, "scored_rows": scored_rows, "per_request": per_request,
            "n_source_images": len(sources), "n_scene_groups": len({row["scene_group_id"] for row in sources.values()}),
            "protocol_sha256": run.protocol_sha256, "task_id": run.protocol["task_id"],
            "planet": run.protocol["planet"], "split": run.protocol["split"],
            "mock": results[0].mock, "scientific_result": False,
            "admission_note": "Computed task metrics do not establish training holdout, matched inputs or scientific comparison admission",
            "experiments": _experiments(conditions, run.protocol["task_id"])}
        record = MetricAnalysis(model_config_id=run.model_config_id, dataset_config_id=run.dataset_config_id,
            run_id=run.id, task_id=run.protocol["task_id"], name="evaluation", method_version=METHOD,
            result_ids=[row.id for row in results], metrics=metrics, results_sha256=signature,
            mock=results[0].mock, notes="Fixture outputs are not scientific scores" if results[0].mock else "")
        if previous:
            if digest(previous.metrics) != digest(metrics):
                raise ValueError("Cached evaluation differs from recomputed metrics; create a new method version")
            record = previous
        else:
            session.add(record)
            session.commit()
            session.refresh(record)
        from benchmark_services.common.files import atomic_json
        atomic_json(Path(run.output_dir) / "metrics.json", record.model_dump())
        return record
    except (OSError, ValueError, KeyError, TypeError) as error:
        session.rollback()
        raise HTTPException(422, f"Cannot evaluate: {error}") from error


def report(session, run_id):
    run = find_run(session, run_id)
    model = session.get(ModelConfig, run.model_config_id)
    dataset = session.get(DatasetConfig, run.dataset_config_id)
    if model is None or dataset is None:
        raise HTTPException(409, "Run model or dataset configuration is missing")
    analyses = session.exec(select(MetricAnalysis).where(MetricAnalysis.run_id == run_id,
        MetricAnalysis.method_version == METHOD).order_by(MetricAnalysis.id.desc())).all()
    analysis = analyses[0] if analyses else None
    if analysis and analysis.results_sha256 != _signature(run, run_results(session, run_id)):
        analysis = None
    return {"run_id": run.id, "status": run.status, "phase": run.phase,
        "progress": {"completed": run.completed, "total": run.total}, "error": run.error_message,
        "model": {"name": model.name, "version": model.version, "backend": model.backend},
        "dataset": {"id": dataset.id, "name": dataset.name, "planet": dataset.planet, "task_id": dataset.task_id},
        "mock": run_is_mock(run, model), "metrics_status": "calculated" if analysis else "not_calculated",
        "analysis_id": analysis.id if analysis else None, "metrics": analysis.metrics if analysis else None,
        "protocol_sha256": run.protocol_sha256,
        "results_url": f"/runs/{run.id}/results", "output_directory": run.output_dir}


def compare(session, run_ids):
    if len(run_ids) != 2 or len(set(run_ids)) != 2:
        raise HTTPException(422, "Select two distinct run IDs")
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
    # Reverify external inputs/maps too; a stale cached report is not evidence.
    for run in (left, right):
        calculate(session, run.id)
    contrasts = []
    for condition, _ in variants(left.protocol):
        a, b = (item["metrics"]["conditions"][condition] for item in reports)
        family = a["metric_family"]
        valid = family in METRICS and family == b["metric_family"]
        metric = METRICS.get(family)
        x, y = (item["quality"].get(metric) if item["quality"] else None for item in (a, b))
        reason = None
        if valid and family in {"imp_segmentation", "catalog_crater_segmentation"}:
            valid = (a["quality"] is not None and b["quality"] is not None
                and a["quality"]["prediction_coverage"] == b["quality"]["prediction_coverage"] == 1
                and a["quality"]["reference_coverage"] == b["quality"]["reference_coverage"] == 1
                and a["quality"]["n_labelled_images"] == b["quality"]["n_labelled_images"])
            if not valid:
                reason = "Native segmentation comparison requires complete matched map coverage"
        contrast = {"condition": condition, "comparable": valid and x is not None and y is not None,
            "metric": metric if valid else None,
            "right_minus_left": y - x if valid and x is not None and y is not None else None,
            "reason": reason if reason else None if valid and x is not None and y is not None
                else "Different output metric families or unavailable admitted ground truth"}
        if contrast["comparable"] and family == "closed_form_classification":
            paired = [[row for row in item["metrics"]["scored_rows"] if row["condition_id"] == condition] for item in reports]
            try:
                contrast["paired_interval"] = paired_accuracy_bootstrap(*paired, seed=left.protocol["seed"])
            except (ValueError, KeyError) as error:
                raise HTTPException(422, f"Incompatible scoring cohorts: {error}") from error
        elif contrast["comparable"] and family == "catalog_crater_segmentation":
            from .heads import paired_scene_interval
            try:
                contrast["paired_interval"] = paired_scene_interval(a["quality"]["per_image"], b["quality"]["per_image"], left.protocol["seed"])
            except (ValueError, KeyError) as error:
                raise HTTPException(422, f"Incompatible crater cohorts: {error}") from error
        contrasts.append(contrast)
    return {"runs": reports, "contrasts": contrasts, "mock": reports[0]["mock"],
        "quality_comparable": all(item["comparable"] for item in contrasts),
        "scientific_result": False,
        "note": ("Fixture metrics are not scientific model results" if reports[0]["mock"] else
            "Native masks and closed-form answers require the same task/metric contract, not the same architecture")}
