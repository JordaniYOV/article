"""Universal command entrypoints; heavy backends import only during inference."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from planetary_vlm.datasets.manifest import load_requests, check_assets
from planetary_vlm.io import read_jsonl, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="planetary-vlm")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check strict model-visible request manifest")
    validate.add_argument("--requests", required=True)
    validate.add_argument("--targets")
    validate.add_argument("--check-assets", action="store_true")
    run_parser = commands.add_parser("run", help="Run one configured backend serially")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--resume", action="store_true")
    evaluate_parser = commands.add_parser("evaluate", help="Score saved responses without loading models")
    for name in ("requests", "targets", "predictions", "output"):
        evaluate_parser.add_argument(f"--{name}", required=True)
    prepare_parser = commands.add_parser("prepare", help="Create image variants without reading answers")
    for name in ("requests", "config", "output"):
        prepare_parser.add_argument(f"--{name}", required=True)
    expand = commands.add_parser("expand-targets", help="Join evaluation targets to variant IDs")
    for name in ("targets", "mapping", "output"):
        expand.add_argument(f"--{name}", required=True)
    depth = commands.add_parser("render-depth", help="Render an existing NPY depth asset")
    depth.add_argument("--input", required=True)
    depth.add_argument("--output", required=True)
    depth.add_argument("--minimum", type=float, required=True)
    depth.add_argument("--maximum", type=float, required=True)
    depth.add_argument("--provenance", choices=("measured", "synthetic", "estimated"), required=True)
    compose = commands.add_parser("compose", help="Compose pre-aligned images left to right")
    compose.add_argument("--images", nargs="+", required=True)
    compose.add_argument("--output", required=True)
    convert = commands.add_parser("convert-semantic", help="Derive ROI questions from native semantic masks")
    for name in ("samples", "specification", "output"):
        convert.add_argument(f"--{name}", required=True)
    compare = commands.add_parser("compare", help="Paired cluster-bootstrap accuracy on saved reports")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            requests = load_requests(args.requests)
            if args.check_assets:
                check_assets(requests)
            if args.targets:
                from planetary_vlm.evaluation.report import evaluate
                evaluate([asdict(request) for request in requests], read_jsonl(args.targets), [])
            print(json.dumps({"valid": True, "requests": len(requests)}))
        elif args.command == "run":
            from planetary_vlm.inference.config import load_config
            from planetary_vlm.inference.runner import run
            print(run(load_config(args.config), resume=args.resume))
        elif args.command == "evaluate":
            from planetary_vlm.evaluation.report import evaluate
            requests = [asdict(request) for request in load_requests(args.requests)]
            predictions = read_jsonl(args.predictions)
            report = evaluate(requests, read_jsonl(args.targets), predictions)
            write_json(args.output, report)
            print(args.output)
        elif args.command == "prepare":
            from planetary_vlm.transforms.prepare import prepare
            print(prepare(args.requests, args.config, args.output))
        elif args.command == "expand-targets":
            from planetary_vlm.transforms.prepare import expand_targets
            expand_targets(args.targets, args.mapping, args.output)
            print(args.output)
        elif args.command == "render-depth":
            from planetary_vlm.transforms.images import render_depth
            render_depth(args.input, args.output, minimum=args.minimum,
                         maximum=args.maximum, provenance=args.provenance)
            write_json(str(args.output) + ".provenance.json", {
                "source": str(Path(args.input).resolve()), "provenance": args.provenance,
                "minimum": args.minimum, "maximum": args.maximum,
                "representation": "fixed_grayscale", "independent_ground_truth": False})
            print(args.output)
        elif args.command == "compose":
            from planetary_vlm.transforms.images import compose_images
            compose_images(tuple(args.images), args.output)
            print(args.output)
        elif args.command == "convert-semantic":
            from planetary_vlm.datasets.semantic import convert_semantic
            print(convert_semantic(args.samples, args.specification, args.output))
        elif args.command == "compare":
            from planetary_vlm.evaluation.metrics import paired_accuracy_bootstrap
            left = json.loads(Path(args.left).read_text(encoding="utf-8"))
            right = json.loads(Path(args.right).read_text(encoding="utf-8"))
            report = paired_accuracy_bootstrap(left["per_request"], right["per_request"],
                                               seed=args.seed, repetitions=args.repetitions)
            markers = (left.get("mock"), right.get("mock"))
            report["mock"] = True if True in markers else (False if markers == (False, False) else None)
            write_json(args.output, report)
            print(args.output)
    except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0
