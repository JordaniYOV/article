"""Universal command entrypoints; heavy backends import only during inference."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import subprocess

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
    dataset = commands.add_parser("dataset", help="Download, prepare and audit datasets independently of inference")
    datasets = dataset.add_subparsers(dest="dataset_command", required=True)
    download = datasets.add_parser("download", help="Explicit source acquisition; never triggered by prepare")
    download.add_argument("--planet", choices=("mars", "moon", "all"), default="all")
    download.add_argument("--root", default="data")
    download.add_argument("--mars-archive", action="store_true")
    download.add_argument("--lunar-pilot-ids", nargs="*", default=["001", "084", "168"])
    download.add_argument("--stereo-stage", choices=("catalog", "labels", "aliases", "images", "pilot"))
    download.add_argument("--images", action="store_true")
    build = datasets.add_parser("prepare", help="Offline versioned RGB/semantic dataset assembly")
    build.add_argument("--planet", choices=("mars", "moon"), required=True)
    build.add_argument("--root", default="data")
    build.add_argument("--samples")
    build.add_argument("--specification")
    build.add_argument('--depth-index', help='Require a QA stereo depth map for every admitted photograph')
    build.add_argument("--output", required=True)
    batch = datasets.add_parser('depth-batch', help='Plan, acquire or verify a resumable depth-complete Mars subset')
    batch.add_argument('--stage', choices=('plan','acquire','run'), required=True)
    batch.add_argument('--root', default='data')
    batch.add_argument('--plan')
    batch.add_argument('--cache')
    batch.add_argument('--output')
    batch.add_argument('--resume', action='store_true')
    batch.add_argument('--min-coverage', type=float, default=.5)
    batch.add_argument('--max-mae', type=float, default=8.)
    batch.add_argument('--limit', type=int)
    batch.add_argument('--build-output', help='After complete verification, assemble the final depth-complete semantic subset')
    for operation in ("extract", "inspect", "search-pairs", "audit-stereo", "check-pair", "pilot", "alignment", "compare-depth"):
        action = datasets.add_parser(operation)
        action.add_argument("--root", default="data")
        if operation == "inspect":
            action.add_argument("--planet", choices=("mars", "moon"), required=True)
        elif operation == "search-pairs":
            action.add_argument("--stage", choices=("requests", "audit"), default="audit")
        elif operation == "pilot":
            action.add_argument("--stage", choices=("select", "verify"), required=True)
            action.add_argument("--output")
        elif operation == "audit-stereo":
            action.add_argument("--source")
            action.add_argument("--output", required=True)
        elif operation in {"check-pair", "alignment"}:
            action.add_argument("--audit", required=True)
            action.add_argument("--output", required=True)
            if operation == "check-pair":
                action.add_argument("--sol", required=True)
                action.add_argument("--left-product")
                action.add_argument("--right-product")
                action.add_argument("--refine-pointing", action="store_true")
        elif operation == "compare-depth":
            action.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "dataset":
            from planetary_vlm.datasets import MarsData, LuneData
            operation = args.dataset_command
            if operation == "download":
                from planetary_vlm.download_dataset import download_dataset
                result = download_dataset(planet=args.planet, root=args.root,
                    mars_archive=args.mars_archive, lunar_pilot_ids=args.lunar_pilot_ids,
                    stereo_stage=args.stereo_stage, images=args.images)
            elif operation == "prepare":
                from planetary_vlm.prepare_dataset import prepare_dataset
                result = prepare_dataset(planet=args.planet, root=args.root,
                    samples=args.samples, specification=args.specification, output=args.output, depth_index=args.depth_index)
            elif operation == "inspect":
                result = (MarsData if args.planet == "mars" else LuneData)(args.root).inspect()
            else:
                mars = MarsData(args.root)
                if operation == 'depth-batch':
                    result = mars.depth_batch(stage=args.stage, output=args.output, plan=args.plan, cache=args.cache,
                        resume=args.resume, min_coverage=args.min_coverage, max_mae=args.max_mae, limit=args.limit,
                        build_output=args.build_output)
                elif operation == "extract":
                    result = mars.extract()
                elif operation == "search-pairs":
                    result = mars.search_pairs(args.stage)
                elif operation == "audit-stereo":
                    result = mars.audit_stereo_sources(source=args.source, output=args.output)
                elif operation == "check-pair":
                    result = mars.check_pair(audit=args.audit, output=args.output, sol=args.sol,
                        left_product=args.left_product, right_product=args.right_product,
                        refine_pointing=args.refine_pointing)
                elif operation == "pilot":
                    if args.stage == "select" and args.output is not None:
                        raise ValueError("--output is only applicable to pilot verify")
                    result = mars.select_pilot_pairs() if args.stage == "select" else mars.verify_pilot_pairs(output=args.output)
                elif operation == "alignment":
                    result = mars.check_alignment(audit=args.audit, output=args.output)
                elif operation == "compare-depth":
                    result = mars.compare_depth_images(output=args.output)
            print(json.dumps(result, default=str, ensure_ascii=False))
        elif args.command == "validate":
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
    except (ValueError, OSError, ImportError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0
