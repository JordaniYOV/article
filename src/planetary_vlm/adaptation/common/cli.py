"""Light CLI orchestration. Tensor/model dependencies are imported only by GPU stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import load
from .data import Index, prepare
from .io import atomic, experiment, gpu_lock, guard, within

STAGES = ("plan", "audit", "prepare", "smoke", "extract", "fit", "train", "evaluate", "predict", "summarize")


def plan(cfg, index, output):
    e = cfg["encoder"]
    cache_bytes = sum(cfg["counts"][s] for s in ("train", "val")) * 4 * e["channels"] * (e["input_size"] // e["patch_size"]) ** 2 * 2
    return {"method": cfg["method"], "splits": cfg["counts"], "dataset": str(index.folder),
            "experiment": str(output), "seeds": [] if cfg["method"] == "sam_baseline" else cfg["seeds"],
            "frozen_cache_gib": round(cache_bytes / 1024**3, 3) if cfg["method"] in ("linear_probe", "frozen_decoder") else 0,
            "input_size": e["input_size"] if cfg["method"] != "sam_baseline" else "native512 -> official SAM processor",
            "model_smoke": "NOT_RUN_BY_PLAN", "training_started": False,
            "test_policy": "Never during train; evaluate --split test --allow-test explicitly",
            "note": "Three trained approaches share NAC input, normalization, source splits and native target grid. SAM is a separate frozen-system baseline."}


def pipeline(args, cfg, output):
    if cfg["method"] == "sam_baseline":
        raise ValueError("SAM baseline has no train stage; use smoke then evaluate")
    guard(within(output, cfg["method"] + "/coordinator_config.json"), cfg)
    steps = [["prepare"], ["smoke"]]
    if cfg["method"] != "lora_decoder":
        steps.append(["extract"])
    for seed in cfg["seeds"]:
        steps.append(["fit", "--seed", str(seed)] + (["--resume"] if args.resume else []))
        steps.append(["evaluate", "--seed", str(seed), "--split", "val"])
    steps.append(["summarize", "--split", "val"])
    for number, step in enumerate(steps):
        command = [sys.executable, "-X", "utf8", "-u", "-B", "-m", "planetary_vlm.adaptation", cfg["method"], *step,
                   "--workspace", str(Path(args.workspace).resolve()), "--output", str(output), "--device", args.device]
        if args.config:
            command.extend(["--config", str(Path(args.config).resolve())])
        log = within(output, f"{cfg['method']}/logs/{number:02d}_{step[0]}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        with log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=args.workspace, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    return {"method": cfg["method"], "validation_complete": True, "test_evaluated": False}


def main(method, argv=None):
    parser = argparse.ArgumentParser(prog=f"planetary-adaptation {method}")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--output", default="outputs/mars_adaptation_v1")
    parser.add_argument("--config", help="Complete/partial TOML overrides with matching method; immutable run identity")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--deep", action="store_true", help="Audit every image/mask hash and format; CPU only")
    parser.add_argument("--image", help="Explicit data_orbital image; prediction never reads its target")
    parser.add_argument("--prediction", default="predictions/prediction.png", help="New image path relative to experiment output")
    args = parser.parse_args(argv)
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        cfg = load(method, args.config)
        args.workspace = str(Path(args.workspace).resolve())
        # Plan is read-only, including the output directory.
        if args.stage == "plan":
            result = plan(cfg, Index(args.workspace, cfg), Path(args.workspace) / args.output)
        else:
            output = experiment(args.workspace, args.output)
            if args.stage in ("prepare", "audit"):
                index = Index(args.workspace, cfg)
                if args.stage == "prepare" and cfg["method"] != "sam_baseline":
                    result = prepare(index, output)
                else:
                    if args.deep:
                        for row in index.inputs:
                            index.image(row, resized=False)
                            index.mask(row["sample_id"])
                    result = {"valid_manifest": True, "counts": cfg["counts"], "data": index.identity,
                              "all_image_mask_bytes_checked": args.deep, "training": False}
                    atomic(within(output, "preparation/audit.json"), result)
            elif args.stage == "train":
                result = pipeline(args, cfg, output)
            elif args.stage == "summarize":
                if cfg["method"] == "sam_baseline":
                    raise ValueError("SAM has one deterministic configuration, no training seeds to average")
                from .reporting import summarize_seeds
                result = summarize_seeds(cfg, output, args.split)
            else:
                if args.stage == "evaluate" and args.split == "test" and not args.allow_test:
                    raise ValueError("Explicit --allow-test is required; no model loaded")
                if cfg["method"] == "sam_baseline" and args.stage in ("fit", "extract"):
                    raise ValueError("SAM baseline is frozen and has no fit/extract stage")
                if cfg["method"] == "lora_decoder" and args.stage == "extract":
                    raise ValueError("LoRA requires online encoding, not a frozen-feature cache")
                with gpu_lock(args.workspace):
                    if cfg["method"] == "sam_baseline":
                        from ..sam_baseline import pipeline as backend
                    else:
                        from . import engine as backend
                    if args.stage == "smoke":
                        result = backend.smoke(args.workspace, cfg, output, args.device)
                    elif args.stage == "extract":
                        result = backend.extract(args.workspace, cfg, output, args.device)
                    elif args.stage == "fit":
                        result = backend.fit(args.workspace, cfg, output, args.seed, args.device, args.resume)
                    elif args.stage == "evaluate":
                        if cfg["method"] == "sam_baseline":
                            result = backend.evaluate(args.workspace, cfg, output, args.split, args.allow_test, args.device)
                        else:
                            result = backend.evaluate(args.workspace, cfg, output, args.seed, args.split, args.allow_test, args.device)
                    elif args.stage == "predict":
                        if not args.image:
                            raise ValueError("--image is required")
                        if cfg["method"] == "sam_baseline":
                            result = backend.predict_image(args.workspace, cfg, output, args.image, args.prediction, args.device)
                        else:
                            result = backend.predict_image(args.workspace, cfg, output, args.seed, args.image, args.prediction, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, ImportError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
