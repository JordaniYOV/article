"""python -m linear_head_approach <independent stage>."""
import argparse
import json
import os
from pathlib import Path
import uuid


def main(argv=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    commands.add_parser("doctor", help="Read-only Windows/CUDA/disk diagnostics, no model load")
    runtime = commands.add_parser("runtime", help="Verify/fetch pinned IBM source code only")
    runtime.add_argument("--allow-download", action="store_true")
    pipeline = commands.add_parser("train", help="Smoke, extract and fit all heads serially; NEVER evaluates test")
    pipeline.add_argument("--encoder", choices=["ibm", "spacellava"], required=True)
    pipeline.add_argument("--experiment", type=Path, default=Path("outputs/heads_rtx3060ti_v1"))
    pipeline.add_argument("--assets", type=Path, default=Path("weights/encoder_heads"))
    pipeline.add_argument("--session-id", default=None)
    pipeline.add_argument("--resume", action="store_true")
    publish = commands.add_parser("publish", help="Export an already-trained head for UI/FastAPI (no training/test)")
    publish.add_argument("--run", type=Path, required=True)
    publish.add_argument("--assets", type=Path, default=Path("weights/encoder_heads"))
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("--name", required=True)
    asset = commands.add_parser("assets")
    asset.add_argument("--encoder", choices=["ibm", "spacellava"], required=True)
    asset.add_argument("--assets", type=Path, required=True)
    asset.add_argument("--allow-download", action="store_true")
    extract = commands.add_parser("extract")
    extract.add_argument("--encoder", choices=["ibm", "spacellava"], required=True)
    extract.add_argument("--assets", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--session-id", required=True)
    extract.add_argument("--smoke", action="store_true")
    smoke = commands.add_parser("head-smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    for name in ("fit", "evaluate", "preview"):
        sub = commands.add_parser(name)
        sub.add_argument("--cache", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "fit":
            sub.add_argument("--family", choices=["linear", "spatial"], required=True)
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--resume", action="store_true")
        elif name == "evaluate":
            sub.add_argument("--split", choices=["val", "test"], default="val")
            sub.add_argument("--allow-test", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args(argv)
    if args.command in {"extract", "head-smoke", "fit", "evaluate", "preview"}:
        from .common import gpu_reservation
        with gpu_reservation():
            result = dispatch(args)
    else:
        result = dispatch(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def dispatch(args):
    if args.command == "audit":
        from .data import audit
        result = audit()
    elif args.command == "doctor":
        from .local import doctor
        result = doctor()
    elif args.command == "runtime":
        from .local import runtime
        result = runtime(args.allow_download)
    elif args.command == "train":
        from .local import pipeline
        result = pipeline(args.encoder, args.experiment, args.assets, args.session_id or uuid.uuid4().hex, args.resume)
    elif args.command == "publish":
        from .deployment import publish
        result = publish(args.run, args.assets, args.output, args.name)
    elif args.command == "assets":
        from .encoders import assets
        result = assets(args.assets, args.encoder, args.allow_download)
    elif args.command == "extract":
        from .encoders import extract as run
        result = run(args.encoder, args.assets, args.output, args.session_id, args.smoke)
    elif args.command == "head-smoke":
        from .training import smoke
        result = smoke(args.output, args.device)
    elif args.command == "fit":
        from .training import fit
        result = fit(args.cache, args.output, args.family, args.seed, args.resume)
    elif args.command == "evaluate":
        from .training import evaluate
        result = evaluate(args.cache, args.output, args.split, args.allow_test)
    elif args.command == "preview":
        from .reporting import preview
        result = preview(args.cache, args.output)
    else:
        from .reporting import compare
        result = compare(args.output, args.split)
    return result


if __name__ == "__main__":
    main()
