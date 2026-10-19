"""Executable frozen-feature and online-LoRA stages, bounded RAM and sealed test."""
from contextlib import nullcontext
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import Index, cache_directory, cache_identity, prepare
from .encoder import load, inspect_assets, restore_trainable, trainable_state
from .io import atomic, code_identity, environment, fingerprint, guard, logger, read, rows, sha, within
from .metrics import binary_counts, select_threshold, summarize
from .model import System, loss, make_head


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Record this policy honestly: CUDA resize/SDPA may not be bitwise deterministic.
    torch.use_deterministic_algorithms(False)


def device_info(device):
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no automatic CPU/size/precision fallback")
        return {"device": device, "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "vram_bytes": torch.cuda.get_device_properties(0).total_memory}
    return {"device": "cpu", "gpu": None, "cuda": None}


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()


def context(root, cfg, output, device):
    hardware = device_info(device)
    index = Index(root, cfg)
    stats = prepare(index, output)
    assets = inspect_assets(root, cfg)
    runtime = {"environment": environment(), "hardware": hardware, "code": code_identity(),
               "math_policy": "seeded_no_tf32_cudnn_benchmark_off_not_bitwise_guaranteed"}
    identity = {"config": cfg, "dataset": index.identity, "stats": stats, "assets": assets, **runtime}
    cache_id = cache_identity(index, stats, assets, runtime["code"], {**runtime["environment"], "hardware": hardware})
    return index, stats, identity, cache_directory(output, cache_id), cache_id


def image_tensor(index, row, stats):
    return torch.from_numpy(((index.image(row) - stats["mean"]) / stats["std"])[None].copy())


def create_system(root, cfg, device, seed, with_encoder):
    encoder = load(root, cfg, device)[0] if with_encoder else None
    # B and C get identical decoder initialization despite encoder constructor RNG use.
    seed_all(seed)
    head = make_head(cfg).to(device)
    adapted = cfg["method"] == "lora_decoder"
    if adapted:
        if encoder is None:
            raise ValueError("LoRA requires an online encoder")
        encoder.add_lora()
    system = System(head, encoder, adapted)
    for name, parameter in system.named_parameters():
        if parameter.requires_grad and not (name.startswith("head.") or name.endswith((".a", ".b"))):
            raise ValueError("Unexpected trainable base weight")
    return system


def smoke(root, cfg, output, device="cuda"):
    index, stats, identity, _, _ = context(root, cfg, output, device)
    directory = within(output, cfg["method"])
    system = create_system(root, cfg, device, cfg["seeds"][0], True)
    controls = index.controls()
    signatures = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with logger(directory) as log:
        for row in controls:
            x = image_tensor(index, row, stats)[None].to(device)
            y = torch.from_numpy(index.mask(row["sample_id"]).astype(np.float32))[None, None].to(device)
            system.train()
            with autocast(device):
                logits = system(x, cfg["encoder"]["input_size"])
                target = F.interpolate(y, logits.shape[-2:], mode="area")
                value = loss(logits, target, stats["positive_weight"], cfg["training"]["dice_weight"])
            if not torch.isfinite(value):
                raise FloatingPointError("Smoke loss is nonfinite")
            value.backward()
            gradients = [p.grad for p in system.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError("Smoke gradients absent/nonfinite")
            if system.encoder is not None and any(p.grad is not None for p in system.encoder.parameters() if not p.requires_grad):
                raise ValueError("A frozen encoder weight received gradients")
            system.zero_grad(set_to_none=True)
            system.eval()
            with torch.no_grad(), autocast(device):
                features = system.encoder(x)
                repeated = system.encoder(x)
            if not all(torch.isfinite(f).all() and torch.allclose(f, r, rtol=.01, atol=.002)
                       for f, r in zip(features, repeated)):
                raise ValueError("Nonfinite/nonrepeatable smoke features")
            signatures.append(features[-1].float().cpu())
            log.info("model_smoke sample=%s loss=%.5f gradient_tensors=%s", row["sample_id"], value.item(), len(gradients))
        if torch.allclose(signatures[0], signatures[1], rtol=.01, atol=.002):
            raise ValueError("Distinct images produced indistinguishable features")
        result = {"identity": identity, "controls": [r["sample_id"] for r in controls],
                  "software_and_real_model_smoke": True, "quality_claim": False,
                  "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                  "trainable_parameters": sum(p.numel() for p in system.parameters() if p.requires_grad)}
        atomic(directory / "smoke.json", result)
    return {k: v for k, v in result.items() if k != "identity"}


def require_smoke(output, cfg, identity):
    path = within(output, cfg["method"] + "/smoke.json")
    if not path.is_file() or read(path).get("identity") != identity:
        raise ValueError("Run model smoke in this exact environment/config before this stage")


def extract(root, cfg, output, device="cuda"):
    if cfg["method"] == "lora_decoder":
        raise ValueError("LoRA uses online images, never a frozen-feature cache")
    index, stats, identity, cache, cache_id = context(root, cfg, output, device)
    require_smoke(output, cfg, identity)
    guard(cache / "identity.json", cache_id)
    selected = index.split("train") + index.split("val")
    manifest = cache / "features.jsonl"
    records = rows(manifest) if manifest.exists() else []
    known = {r["sample_id"]: r for r in records}
    allowed = {r["sample_id"]: r for r in selected}
    if len(known) != len(records) or not set(known) <= set(allowed):
        raise ValueError("Unknown/duplicate cached sample")
    side = cfg["encoder"]["input_size"] // cfg["encoder"]["patch_size"]
    shape = [4, cfg["encoder"]["channels"], side, side]
    for record in records:
        row = allowed[record["sample_id"]]
        path = within(cache, row["sample_id"] + ".npy")
        if record["sha256"] != sha(path) or record["shape"] != shape:
            raise ValueError("Changed cached feature file")
    required = (len(selected) - len(known)) * math.prod(shape) * 2 + 1024 ** 3
    if shutil.disk_usage(cache).free < required:
        raise OSError(f"Insufficient feature-cache disk space; need approximately {required / 1024**3:.2f} GiB")
    encoder, _ = load(root, cfg, device)
    with logger(cache) as log:
        for number, row in enumerate(selected):
            sid = row["sample_id"]
            if sid in known:
                continue
            x = image_tensor(index, row, stats)[None].to(device)
            with torch.no_grad(), autocast(device):
                features = torch.stack(encoder(x), dim=1)[0].float().cpu().numpy().astype(np.float16)
            if list(features.shape) != shape or not np.isfinite(features).all() or float(features.astype(np.float32).std()) < 1e-8:
                raise ValueError("Invalid extracted feature shape/values")
            path = within(cache, sid + ".npy")
            if path.exists():
                old = np.load(path, allow_pickle=False)
                if old.shape != features.shape or not np.allclose(old, features, rtol=.01, atol=.002):
                    raise ValueError("Orphan feature differs; choose a fresh experiment")
            else:
                temporary = path.with_suffix(".partial")
                with temporary.open("wb") as stream:
                    np.save(stream, features, allow_pickle=False)
                temporary.replace(path)
            record = {"sample_id": sid, "split": row["split"], "scene_group_id": row["scene_group_id"],
                      "file": path.name, "shape": shape, "sha256": sha(path)}
            records.append(record)
            known[sid] = record
            if len(records) % 32 == 0:
                atomic(manifest, records, lines=True)
                log.info("features_completed=%s total=%s", len(records), len(selected))
        atomic(manifest, records, lines=True)
        atomic(cache / "complete.json", {"identity": fingerprint(cache_id), "count": len(records), "manifest_sha256": sha(manifest)})
    return {"cache": str(cache), "images": len(records), "test_extracted": False}


class Samples(Dataset):
    """Reads one item at a time; never accumulates an all-images RAM dictionary."""
    def __init__(self, index, selected, stats, cache=None):
        self.index, self.records, self.stats, self.cache = index, selected, stats, cache

    def __len__(self):
        return len(self.records)

    def __getitem__(self, position):
        row = self.records[position]
        if self.cache is None:
            x = image_tensor(self.index, row, self.stats)
        else:
            array = np.load(within(self.cache, row["sample_id"] + ".npy"), allow_pickle=False)
            if array.dtype != np.float16 or not np.isfinite(array).all():
                raise ValueError("Invalid cached feature values/precision")
            x = [torch.from_numpy(value.astype(np.float32)) for value in array]
        y = torch.from_numpy(self.index.mask(row["sample_id"]).astype(np.float32))[None]
        return x, y, position


def to_device(x, device):
    return [t.to(device) for t in x] if isinstance(x, (list, tuple)) else x.to(device)


@torch.no_grad()
def predictions(system, dataset, device, thresholds, save=None):
    system.eval()
    results = {float(t): [] for t in thresholds}
    for x, y, position in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        row = dataset.records[int(position.item())]
        target = y[0, 0].numpy().astype(bool)
        record_path = within(save, row["sample_id"] + ".json") if save is not None else None
        if record_path is not None and record_path.is_file():
            if len(results) != 1:
                raise ValueError("Persistent scoring requires one frozen threshold")
            stored = read(record_path)
            path = within(save, row["sample_id"] + ".png")
            if stored["prediction_sha256"] != sha(path):
                raise ValueError("Saved evaluation prediction changed")
            with Image.open(path) as image:
                labels = np.asarray(image)
            if labels.shape != target.shape or not set(np.unique(labels)) <= {0, 1}:
                raise ValueError("Saved prediction has an invalid grid/classes")
            expected = {"sample_id": row["sample_id"], "scene_group_id": row["scene_group_id"],
                        **binary_counts(labels, target)}
            if any(stored.get(k) != v for k, v in expected.items()):
                raise ValueError("Saved evaluation record differs from mask/target")
            results[next(iter(results))].append(stored)
            continue
        with autocast(device):
            probability = system(to_device(x, device), 512).float().sigmoid()[0, 0].cpu().numpy()
        if not np.isfinite(probability).all():
            raise ValueError("Nonfinite prediction")
        for threshold in results:
            prediction = probability >= threshold
            result = {"sample_id": row["sample_id"], "scene_group_id": row["scene_group_id"],
                      **binary_counts(prediction, target)}
            if save is not None:
                path = within(save, row["sample_id"] + ".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(prediction.astype(np.uint8)).save(path)
                result["prediction_mask"] = str(path)
                result["prediction_sha256"] = sha(path)
                atomic(record_path, result)
            results[threshold].append(result)
    return results


def check_cache(cache, cache_id, index):
    if read(cache / "identity.json") != cache_id:
        raise ValueError("Cache belongs to another encoder/preprocessing")
    complete = read(cache / "complete.json")
    expected = {r["sample_id"] for r in index.inputs if r["split"] != "test"}
    manifest = rows(cache / "features.jsonl")
    if (complete["identity"] != fingerprint(cache_id) or complete["manifest_sha256"] != sha(cache / "features.jsonl")
            or len(manifest) != len(expected) or {r["sample_id"] for r in manifest} != expected):
        raise ValueError("Feature cache is incomplete")
    for number, row in enumerate(manifest):
        if row["file"] != row["sample_id"] + ".npy" or sha(within(cache, row["file"])) != row["sha256"]:
            raise ValueError("Feature cache asset changed")
        if (number + 1) % 100 == 0:
            print(f"cache_verified={number + 1}/{len(manifest)}", flush=True)


def save_checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(saved, generator):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])
    if saved["cuda"]:
        torch.cuda.set_rng_state_all(saved["cuda"])
    generator.set_state(saved["loader"])


def fit(root, cfg, output, seed, device="cuda", resume=False):
    if seed not in cfg["seeds"]:
        raise ValueError("Seed not in predeclared protocol")
    index, stats, identity, cache, cache_id = context(root, cfg, output, device)
    require_smoke(output, cfg, identity)
    adapted = cfg["method"] == "lora_decoder"
    if not adapted:
        check_cache(cache, cache_id, index)
    run = within(output, f"{cfg['method']}/seed_{seed}")
    identity = {**identity, "seed": seed, "cache": None if adapted else fingerprint(cache_id)}
    guard(run / "identity.json", identity)
    if (run / "last.pt").exists() and not resume:
        raise ValueError("Run exists; use --resume or a new experiment")
    system = create_system(root, cfg, device, seed, adapted)
    train_data = Samples(index, index.split("train"), stats, None if adapted else cache)
    val_data = Samples(index, index.split("val"), stats, None if adapted else cache)
    t = cfg["training"]
    groups = [{"params": system.head.parameters(), "lr": t["head_lr"]}]
    if adapted:
        groups.append({"params": [p for p in system.encoder.parameters() if p.requires_grad], "lr": t["lora_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    generator = torch.Generator().manual_seed(seed)
    start, best, wait, history = 0, -1., 0, []
    if resume and (run / "last.pt").is_file():
        # Own local run checkpoint only: RNG state requires weights_only=False.
        saved = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Checkpoint identity differs")
        restore_trainable(system, saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start, best, wait, history = saved["epoch"] + 1, saved["best"], saved["wait"], saved["history"]
        restore_rng(saved["rng"], generator)
    loader = DataLoader(train_data, batch_size=t["batch_size"], shuffle=True, generator=generator, num_workers=0)
    with logger(run) as log:
        for epoch in range(start, t["max_epochs"]):
            if wait >= t["patience"]:
                break
            begin, total = time.monotonic(), 0.
            optimizer_steps, skipped_steps = 0, 0
            system.train()
            optimizer.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for number, (x, y, _) in enumerate(loader):
                size = cfg["encoder"]["input_size"]
                target = F.interpolate(y.to(device), (size, size), mode="area")
                window_start = (number // t["accumulate"]) * t["accumulate"]
                window = min(t["accumulate"], len(loader) - window_start)
                with autocast(device):
                    value = loss(system(to_device(x, device), size), target, stats["positive_weight"], t["dice_weight"])
                if not torch.isfinite(value):
                    raise FloatingPointError("Nonfinite train loss")
                scaler.scale(value / window).backward()
                total += value.detach().item()
                if (number + 1) % t["accumulate"] == 0 or number + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    # AMP must be allowed to skip overflowing scaled gradients and lower its scale.
                    # Raising on the first overflow would disable GradScaler's normal recovery.
                    norm = torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad],
                        t["gradient_clip"], error_if_nonfinite=not scaler.is_enabled())
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < previous_scale:
                        skipped_steps += 1
                    else:
                        optimizer_steps += 1
                        if not torch.isfinite(norm):
                            raise FloatingPointError("Nonfinite gradient norm without an AMP-recovered overflow")
                    optimizer.zero_grad(set_to_none=True)
                if (number + 1) % 20 == 0:
                    log.info("epoch=%s batch=%s/%s loss=%.5f", epoch + 1, number + 1, len(loader), value.item())
            if optimizer_steps == 0:
                raise FloatingPointError("Every optimizer step overflowed; no successful training update")
            validation = predictions(system, val_data, device, t["thresholds"])
            threshold = select_threshold(validation)
            score = summarize(validation[threshold])["scene_macro_iou"]
            improved = score > best + 1e-8
            best, wait = (score, 0) if improved else (best, wait + 1)
            weights = trainable_state(system)
            if improved:
                save_checkpoint(run / "best.pt", {"identity": identity, "weights": weights,
                    "threshold": threshold, "epoch": epoch, "validation_scene_macro_iou": score})
            scheduler.step()
            history.append({"epoch": epoch + 1, "loss": total / len(loader), "validation_scene_macro_iou": score,
                            "threshold": threshold, "seconds": time.monotonic() - begin,
                            "optimizer_steps": optimizer_steps, "amp_skipped_steps": skipped_steps,
                            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0})
            save_checkpoint(run / "last.pt", {"identity": identity, "weights": weights, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(generator),
                "epoch": epoch, "best": best, "wait": wait, "history": history})
            atomic(run / "history.json", history)
            log.info("epoch=%s val_scene_iou=%.5f best=%.5f patience=%s/%s seconds=%.1f", epoch + 1, score, best, wait, t["patience"], history[-1]["seconds"])
        result = {"method": cfg["method"], "seed": seed, "epochs": len(history), "best_validation_scene_iou": best,
                  "trainable_parameters": sum(p.numel() for p in system.parameters() if p.requires_grad),
                  "test_used_for_selection": False, "best_checkpoint_sha256": sha(run / "best.pt"),
                  "identity_sha256": sha(run / "identity.json")}
        atomic(run / "complete.json", result)
    return result


def evaluation_system(root, cfg, output, seed, device):
    index, stats, identity, _, cache_id = context(root, cfg, output, device)
    adapted = cfg["method"] == "lora_decoder"
    identity = {**identity, "seed": seed, "cache": None if adapted else fingerprint(cache_id)}
    run = within(output, f"{cfg['method']}/seed_{seed}")
    if read(run / "identity.json") != identity:
        raise ValueError("Evaluation must use the frozen training identity/environment")
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if saved["identity"] != identity:
        raise ValueError("Best checkpoint identity changed")
    system = create_system(root, cfg, device, seed, True)
    restore_trainable(system, saved["weights"])
    return index, stats, identity, run, saved, system


def evaluate(root, cfg, output, seed, split="val", allow_test=False, device="cuda"):
    if split == "test" and not allow_test:
        raise ValueError("Test scoring requires explicit --allow-test after protocol freeze")
    index, stats, identity, run, saved, system = evaluation_system(root, cfg, output, seed, device)
    if split == "test":
        complete = read(run / "complete.json")
        if complete["best_checkpoint_sha256"] != sha(run / "best.pt"):
            raise ValueError("Complete-run checkpoint differs")
    directory = within(run, "evaluation/" + split)
    guard(directory / "identity.json", {"run": identity, "checkpoint_sha256": sha(run / "best.pt"),
                                       "split": split, "threshold": saved["threshold"]})
    records = predictions(system, Samples(index, index.split(split), stats), device, [saved["threshold"]], directory / "masks")[saved["threshold"]]
    report = {"schema": "mars-segmentation-report-v1", "method": cfg["method"], "seed": seed, "split": split,
              "dataset": index.identity, "protocol": cfg, "threshold": saved["threshold"],
              "checkpoint_sha256": sha(run / "best.pt"), "records": records, "metrics": summarize(records),
              "test_used_for_selection": False,
              "interpretation": "Publisher catalog coverage, not exhaustive physical crater truth"}
    atomic(directory / "report.json", report)
    return {"report": str(directory / "report.json"), "metrics": report["metrics"]}


def predict_image(root, cfg, output, seed, image_path, destination, device="cuda"):
    from .io import orbital
    index, stats, _, _, saved, system = evaluation_system(root, cfg, output, seed, device)
    path = orbital(root, image_path)
    row = {"image": str(path.relative_to(Path(root).resolve())), "image_sha256": sha(path)}
    x = image_tensor(index, row, stats)[None].to(device)
    system.eval()
    with torch.no_grad(), autocast(device):
        probability = system(x, 512).float().sigmoid()[0, 0].cpu().numpy()
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite prediction")
    destination = within(output, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Prediction output exists; choose a new filename")
    Image.fromarray((probability >= saved["threshold"]).astype(np.uint8)).save(destination)
    return {"mask": str(destination), "mask_sha256": sha(destination), "threshold": saved["threshold"], "classes": {"0": "Background", "1": "Crater"}}
