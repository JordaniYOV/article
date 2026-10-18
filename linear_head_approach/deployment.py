"""Publish a fitted head for UI/FastAPI; never trains, tunes or scores test."""
from pathlib import Path
import shutil

from .common import ROOT, atomic_json, code_hash, environment, read_json, rows, sha, write_rows
from .data import DATA, audit


def publish(run, assets_dir, output, name, root=ROOT):
    import torch
    from planetary_vlm.heads_contract import TASK, INPUT_PROTOCOL, CLASSES, pixel_hash
    root, run, output, assets_dir = Path(root).resolve(), Path(run).resolve(), Path(output).resolve(), Path(assets_dir).resolve()
    if not output.is_relative_to(root / "outputs") or output.exists():
        raise ValueError("Publish into a NEW directory beneath outputs/")
    summary = read_json(run / "fit_summary.json")
    if not summary["epochs"] or summary["encoder_trained"] or summary["test_used_for_selection"]:
        raise ValueError("Only a completed, frozen-encoder, validation-selected run can be published")
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    identity = read_json(run / "identity.json")
    if saved["identity"] != identity or identity["code"] != code_hash() or identity["environment"] != environment():
        raise ValueError("Training identity changed; restore the exact environment/code")
    if identity["targets_sha256"] != sha(root / DATA / "targets.jsonl"):
        raise ValueError("Training targets changed")
    audit(root)
    encoder = identity["feature_identity"]["encoder"]
    if encoder not in {"ibm", "spacellava"}:
        raise ValueError("Only admitted IBM/SpaceLLaVA frozen encoders can be published")
    from .encoders import assets
    if assets(assets_dir, encoder) != identity["feature_identity"]["assets"]:
        raise ValueError("Encoder assets differ from those used for training features")
    inputs = [r for r in rows(root / DATA / "inputs.jsonl") if r["split"] in {"val", "test"}]
    target_map = {r["sample_id"]: r for r in rows(root / DATA / "targets.jsonl")}
    evaluation = []
    for row in inputs:
        target = target_map[row["sample_id"]]
        evaluation.append({"sample_id": row["sample_id"], "scene_group_id": row["scene_group_id"], "source_split": row["split"],
            "image": row["rgb_image"], "image_sha256": row["rgb_image_sha256"], "native_vis": row["native_vis"],
            "native_vis_sha256": row["native_vis_sha256"], "mask": target["mask"], "mask_sha256": target["mask_sha256"],
            "mask_semantics": "catalog_crater_polygon", "classes": CLASSES, "reference_status": "ready",
            "source_dataset": row["source_dataset"], "source_revision": row["source_revision"],
            "evaluation_eligible": True, "reference_scope": "publisher catalog polygons, not all visible craters"})
    # One canonical evaluation manifest, common to ALL models/families/seeds.
    evaluation_path = root / DATA / "benchmark_manifest.jsonl"
    if evaluation_path.exists() and rows(evaluation_path) != evaluation:
        raise ValueError("Existing benchmark manifest changed")
    write_rows(evaluation_path, evaluation)
    output.mkdir(parents=True)
    shutil.copyfile(run / "best.pt", output / "head.pt")
    model_inputs = [r | {"rgb_pixels_sha256": pixel_hash(root / r["rgb_image"])} for r in inputs]
    write_rows(output / "encoder_inputs.jsonl", model_inputs)
    deployment = {"schema": "frozen-crater-deployment-v1", "task": TASK, "input_protocol": INPUT_PROTOCOL,
        "encoder": encoder, "family": identity["family"], "seed": identity["seed"], "channels": saved["channels"],
        "hidden_channels": identity["config"]["hidden_channels"], "threshold": saved["threshold"],
        "threshold_selected_on": "val", "best_epoch": saved["epoch"] + 1, "training_identity": identity,
        "assets_dir": str(assets_dir), "checkpoint_parameters": saved["parameters"],
        "source_run": str(run), "benchmark_manifest": str(evaluation_path), "benchmark_sha256": sha(evaluation_path),
        "classes": CLASSES, "shape": [512, 512], "clean_only": True,
        "files_sha256": {file: sha(output / file) for file in ("head.pt", "encoder_inputs.jsonl")},
        "input_warning": "Native IBM VIS5 vs CLIP RGB: not equal information; source holdout is not pretraining independence"}
    path = output / "deployment.json"
    atomic_json(path, deployment)
    pointer = {"deployment_path": str(path), "sha256": sha(path), "name": name,
               "version": "v1", "kind": "frozen_crater_deployment"}
    atomic_json(output / "import_in_ui.json", pointer)
    return {"deployment": str(path), "ui_import": str(output / "import_in_ui.json"),
            "training_started": False, "test_evaluated": False}
