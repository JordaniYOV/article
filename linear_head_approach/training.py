"""Train-only statistics, validation-only selection, atomic epoch resume, sealed test."""
from __future__ import annotations

import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .common import ROOT, atomic_json, code_hash, config, environment, fingerprint, identity_guard, logger, read_json, rows, safe_asset, sha, closes_stage_logs
from .data import DATA, metrics, scene_macro, bootstrap, validate_inputs
from .heads import CraterHead, loss_per_image


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def atomic_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def checked_features(cache, root=ROOT):
    cache = Path(cache)
    input_rows = rows(Path(root) / DATA / "inputs.jsonl")
    validate_inputs(input_rows)
    source = {r["sample_id"]: r for r in input_rows}
    feature_rows = rows(cache / "features.jsonl")
    if len(feature_rows) != len(source) or {r["sample_id"] for r in feature_rows} != set(source):
        raise ValueError("Incomplete/duplicate feature cache; extract the complete cohort first")
    identity = read_json(cache / "identity.json")
    if (identity["inputs_sha256"] != sha(Path(root) / DATA / "inputs.jsonl")
            or identity["code"] != code_hash() or identity["environment"] != environment()):
        raise ValueError("Features belong to different inputs")
    for row in feature_rows:
        expected = source[row["sample_id"]]
        if any(row[k] != expected[k] for k in ("split", "scene_group_id")) or row["file"] != f"{row['sample_id']}.npy":
            raise ValueError("Feature ID/split/group/path mismatch")
        path = cache / row["file"]
        feature = np.load(path, mmap_mode="r", allow_pickle=False)
        if sha(path) != row["sha256"] or list(feature.shape) != row["shape"] or feature.shape[1:] != (24, 24) or feature.dtype != np.float32 or not np.isfinite(feature).all():
            raise ValueError("Feature integrity/shape error")
    if len({r["shape"][0] for r in feature_rows}) != 1:
        raise ValueError("Mixed feature channel counts")
    return feature_rows


class CachedDataset(Dataset):
    def __init__(self, cache, records, targets, mean, std, root=ROOT):
        self.cache, self.records, self.targets, self.root = Path(cache), records, targets, Path(root)
        self.mean, self.std = mean, std
        # <= ~0.4 GiB for this fixed cohort: avoid reopening Drive NPYs every epoch.
        # Only admitted split rows are loaded, never test rows during fit.
        self.loaded = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if index in self.loaded:
            return self.loaded[index]
        row = self.records[index]
        target = self.targets[row["sample_id"]]
        if any(target[k] != row[k] for k in ("split", "scene_group_id")) or target["target"] != "catalog_crater_polygon":
            raise ValueError("Target alignment/task mismatch")
        feature = torch.from_numpy(np.load(self.cache / row["file"], allow_pickle=False).copy())
        feature = (feature - self.mean[:, None, None]) / self.std[:, None, None]
        with Image.open(safe_asset(self.root, target["mask"], target["mask_sha256"])) as mask:
            y = np.asarray(mask, dtype=np.float32).copy()
        if y.shape != (512, 512) or not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("Invalid binary segmentation target")
        self.loaded[index] = feature, torch.from_numpy(y)[None], index
        return self.loaded[index]


def train_statistics(cache, records, targets, root, max_positive_weight):
    sums = squares = None
    pixels = 0
    groups = Counter(r["scene_group_id"] for r in records)
    foreground, background = 0., 0.
    for row in records:
        if row["split"] != "train":
            raise ValueError("Statistics can only use train")
        x = np.load(Path(cache) / row["file"], allow_pickle=False).astype(np.float64)
        if sums is None:
            sums, squares = np.zeros(x.shape[0]), np.zeros(x.shape[0])
        sums += x.sum((1, 2))
        squares += (x * x).sum((1, 2))
        pixels += x.shape[1] * x.shape[2]
        target = targets[row["sample_id"]]
        with Image.open(safe_asset(root, target["mask"], target["mask_sha256"])) as source:
            y = np.asarray(source)
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("Train target not binary")
        weight = 1 / groups[row["scene_group_id"]]
        foreground += float(y.sum()) * weight
        background += float(y.size - y.sum()) * weight
    if foreground == 0 or background == 0:
        raise ValueError("Train must contain both foreground and background pixels")
    mean = sums / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean**2, 1e-12))
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32), min(max_positive_weight, background / foreground)


@torch.inference_mode()
def predict(head, dataset, device, thresholds):
    head.eval()
    results = {float(t): [] for t in thresholds}
    for x, y, index in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        probability = head(x.to(device), 512).sigmoid()[0, 0].cpu().numpy()
        target = y[0, 0].numpy().astype(bool)
        row = dataset.records[int(index.item())]
        for threshold in results:
            results[threshold].append({"sample_id": row["sample_id"], "scene_group_id": row["scene_group_id"],
                                       **metrics(probability >= threshold, target)})
    return results


@closes_stage_logs
def fit(cache, output, family, seed, resume=False, root=ROOT, cfg_path=None, device="cuda"):
    cfg = config() if cfg_path is None else config(cfg_path)
    cache, output, root = Path(cache), Path(output), Path(root)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if family not in cfg["heads"] or seed not in cfg["seeds"]:
        raise ValueError("Family/seed not in predeclared protocol")
    records = checked_features(cache, root)
    identity = {"config": cfg, "family": family, "seed": seed, "code": code_hash(),
                "environment": environment(), "feature_identity": read_json(cache / "identity.json"),
                "feature_manifest_sha256": sha(cache / "features.jsonl"),
                "targets_sha256": sha(root / DATA / "targets.jsonl"),
                "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    identity_guard(output / "identity.json", identity)
    log = logger(output)
    checkpoint_file = output / "last.pt"
    if checkpoint_file.exists() and not resume:
        raise ValueError("Run exists; pass --resume or choose a new output directory")
    seed_all(seed)
    # Test rows/labels are never loaded by a training/validation Dataset.
    train_rows = [r for r in records if r["split"] == "train"]
    val_rows = [r for r in records if r["split"] == "val"]
    targets = {r["sample_id"]: r for r in rows(root / DATA / "targets.jsonl") if r["split"] in {"train", "val"}}
    if not train_rows or not val_rows:
        raise ValueError("Train and validation required")
    groups = Counter(r["scene_group_id"] for r in train_rows)
    weights = torch.tensor([len(train_rows) / (len(groups) * groups[r["scene_group_id"]]) for r in train_rows])
    mean, std, positive_weight = train_statistics(cache, train_rows, targets, root, cfg["max_positive_weight"])
    train_data = CachedDataset(cache, train_rows, targets, mean, std, root)
    val_data = CachedDataset(cache, val_rows, targets, mean, std, root)
    head = CraterHead(len(mean), family, cfg["hidden_channels"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])
    history, start, best, wait = [], 0, -1., 0
    generator = torch.Generator().manual_seed(seed)
    if resume and checkpoint_file.exists():
        # These files are generated by this program; NEVER load an untrusted .pt.
        saved = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Checkpoint identity differs")
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best, wait, history = saved["epoch"] + 1, saved["best"], saved["wait"], saved["history"]
        generator.set_state(saved["generator"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True, generator=generator, num_workers=0)
    positive_weight_tensor = torch.tensor(positive_weight, device=device)
    for epoch in range(start, cfg["max_epochs"]):
        if wait >= cfg["patience"]:
            break
        begin = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        head.train()
        total, batches = 0., 0
        for x, y, indices in loader:
            x = x.to(device)
            y = F.interpolate(y.to(device), (cfg["loss_grid"], cfg["loss_grid"]), mode="area")
            optimizer.zero_grad(set_to_none=True)
            logits = head(x, cfg["loss_grid"])
            losses = loss_per_image(logits, y, positive_weight_tensor, cfg["dice_weight"])
            loss = (losses * weights[indices].to(device)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(x)
            batches += len(x)
        validation = predict(head, val_data, device, cfg["thresholds"])
        # Stable ties: first/lower threshold in declared grid, earlier epoch.
        threshold = max(validation, key=lambda t: scene_macro(validation[t]))
        score = scene_macro(validation[threshold])
        improved = score > best + 1e-8
        if improved:
            best, wait = score, 0
            atomic_checkpoint(output / "best.pt", {"head": head.state_dict(), "mean": mean, "std": std,
                "threshold": threshold, "epoch": epoch, "validation_scene_iou": score, "identity": identity,
                "channels": len(mean), "parameters": sum(p.numel() for p in head.parameters())})
        else:
            wait += 1
        scheduler.step()
        duration = time.perf_counter() - begin
        record = {"epoch": epoch + 1, "train_loss": total / batches, "validation_scene_iou": score,
                  "threshold": threshold, "best_validation_iou": best, "seconds": duration,
                  "learning_rate_next_epoch": scheduler.get_last_lr()[0],
                  "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.}
        history.append(record)
        atomic_checkpoint(checkpoint_file, {"head": head.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best, "wait": wait, "history": history,
            "identity": identity, "generator": generator.get_state(), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []})
        atomic_json(output / "history.json", history)
        log.info("epoch=%s/%s loss=%.5f val_scene_iou=%.4f threshold=%.2f best=%.4f wait=%s seconds=%.1f max_remaining_minutes=%.1f",
                 epoch + 1, cfg["max_epochs"], record["train_loss"], score, threshold, best, wait,
                 duration, np.mean([r["seconds"] for r in history]) * (cfg["max_epochs"] - epoch - 1) / 60)
    result = {"best_validation_scene_iou": best, "epochs": len(history), "family": family, "seed": seed,
              "test_used_for_selection": False, "encoder_trained": False, "training_role": "supporting_frozen_feature_probe"}
    atomic_json(output / "fit_summary.json", result)
    return result


def evaluate(cache, output, split="val", allow_test=False, root=ROOT, device="cuda"):
    root, cache, output = Path(root), Path(cache), Path(output)
    if split == "test" and not allow_test:
        raise ValueError("Test sealed. Freeze protocol, then pass --allow-test explicitly")
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    records = checked_features(cache, root)
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    identity = read_json(output / "identity.json")
    if (saved["identity"] != identity or identity["code"] != code_hash()
            or identity["environment"] != environment()
            or identity["targets_sha256"] != sha(root / DATA / "targets.jsonl")
            or identity["feature_manifest_sha256"] != sha(cache / "features.jsonl")
            or identity["feature_identity"] != read_json(cache / "identity.json")):
        raise ValueError("Evaluation identity changed; never evaluate a different cache/target/code")
    cfg = identity["config"]
    targets = {r["sample_id"]: r for r in rows(root / DATA / "targets.jsonl") if r["split"] == split}
    selected = [r for r in records if r["split"] == split]
    dataset = CachedDataset(cache, selected, targets, saved["mean"], saved["std"], root)
    head = CraterHead(saved["channels"], identity["family"], cfg["hidden_channels"]).to(device)
    head.load_state_dict(saved["head"])
    predictions = predict(head, dataset, device, [saved["threshold"]])[saved["threshold"]]
    baselines = {"all_background": [], "all_foreground": []}
    for _, y, index in dataset:
        row = selected[index]
        for name, value in (("all_background", 0), ("all_foreground", 1)):
            baselines[name].append({"scene_group_id": row["scene_group_id"], **metrics(np.full((512, 512), value), y[0].numpy())})
    result = {"split": split, "images": len(predictions), "scene_iou": bootstrap(predictions),
              "scene_dice": bootstrap(predictions, "dice"), "image_macro_iou": float(np.mean([r["iou"] for r in predictions])),
              "baselines": {name: bootstrap(items) for name, items in baselines.items()},
              "threshold_selected_on": "val", "threshold": saved["threshold"], "best_epoch": saved["epoch"] + 1,
              "head_parameters": saved["parameters"], "rows": predictions, "seed": identity["seed"], "family": identity["family"],
              "encoder": identity["feature_identity"]["encoder"], "protocol_sha256": fingerprint(cfg),
              "targets_sha256": identity["targets_sha256"], "checkpoint_sha256": sha(output / "best.pt"),
              "limitation": "Catalog-polygon segmentation, not full VLM QA, exhaustive crater detection or route safety. Native input information differs."}
    atomic_json(output / f"evaluation_{split}.json", result)
    return {k: v for k, v in result.items() if k != "rows"}


def smoke(output, device="cuda"):
    """Synthetic software check ONLY, no dataset or pretrained model training."""
    seed_all(13)
    result = {}
    for channels in (768, 1024):
        for family in ("linear", "spatial"):
            head = CraterHead(channels, family).to(device)
            x = torch.randn(2, channels, 24, 24, device=device)
            y = torch.zeros(2, 1, 128, 128, device=device)
            y[:, :, 30:80, 30:80] = 1
            optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
            before = next(head.parameters()).detach().clone()
            value = loss_per_image(head(x), y, torch.tensor(1., device=device)).mean()
            value.backward()
            optimizer.step()
            if not torch.isfinite(value) or torch.equal(before, next(head.parameters())):
                raise ValueError("Head forward/backprop smoke failed")
            result[f"{family}_{channels}"] = float(value.detach())
    result.update(scientific_result=False, kind="synthetic_software_check", device=device)
    atomic_json(Path(output) / "head_smoke.json", result)
    return result
