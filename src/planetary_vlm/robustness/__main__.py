"""One command: shared preparation -> ten serial inference jobs -> summary."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from planetary_vlm.adaptation.common.config import METHODS
from planetary_vlm.adaptation.common.io import fingerprint, gpu_lock, guard, logger, read, sha
from . import bundle
from .protocol import dedicated, discover, job_id, load_protocol, selection


def serial(command, root, log_path):
    """Wait for each child; cancellation terminates it before releasing the queue."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    # Also works from an editable/source checkout without reinstalling metadata.
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
            if code:
                raise RuntimeError(f"Stage exited ({code}); see {log_path}. Repeat the same command to resume.")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def output_guard(path, identity):
    if path.exists() and any(path.iterdir()) and not (path / "identity.json").is_file():
        raise ValueError("Existing non-robustness output directory is protected")
    guard(path / "identity.json", identity)


def coordinate(args, root, clean, output, folder, index, protocol, selected, jobs, definition):
    bundle.prepare(root, folder, index, protocol, selected, args.split)
    bundle.verify(root, folder, definition)
    bundle_hash = sha(folder / "complete.json")
    for number, job in enumerate(jobs):
        command = [sys.executable, "-X", "utf8", "-u", "-B", "-m", "planetary_vlm.robustness", "worker",
                   "--workspace", str(root), "--clean-output", str(clean), "--output", str(output),
                   "--bundle", str(folder), "--split", args.split, "--limit", str(args.limit),
                   "--device", args.device, "--job", str(number), "--methods", *args.methods]
        if args.config:
            command.extend(["--config", str(Path(args.config).resolve())])
        if args.allow_test:
            command.append("--allow-test")
        print(f"approach_started={number + 1}/{len(jobs)} {job_id(job)}", flush=True)
        serial(command, root, output / "logs" / (job_id(job).replace("/", "_") + ".log"))
    from .reporting import summarize
    return summarize(output, jobs, definition, bundle_hash)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m planetary_vlm.robustness")
    parser.add_argument("stage", choices=("plan", "prepare", "verify", "run", "worker", "summarize"))
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--clean-output", default="outputs/mars_adaptation_v1", help="Trusted original completed training/SAM runs")
    parser.add_argument("--output", default="outputs/mars_robustness_v1")
    parser.add_argument("--bundle", default="data_orbital/mars_corruptions_v1")
    parser.add_argument("--config", help="Robustness TOML only; training configuration is read from original identities")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0=all; fixed blind subset for a separate pilot protocol")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--job", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.split == "test" and not args.allow_test:
            raise ValueError("Test protocol requires explicit --allow-test; no models loaded")
        root = Path(args.workspace).resolve()
        protocol = load_protocol(args.config)
        clean, index, jobs = discover(root, args.clean_output, args.methods)
        selected = selection(index, args.split, args.limit, protocol["seed"])
        folder = dedicated(root, args.bundle, "data_orbital")
        output = dedicated(root, args.output, "outputs")
        if output == clean or output.is_relative_to(clean) or clean.is_relative_to(output):
            raise ValueError("Robustness output must be separate from original training runs")
        definition = bundle.definition(index, protocol, selected, args.split)
        identity = {"schema": "mars-robustness-v1", "jobs": jobs, "clean_output": clean.relative_to(root).as_posix(),
                    "bundle": folder.relative_to(root).as_posix(), "bundle_identity_sha256": fingerprint(definition),
                    "device": args.device}
        if args.stage == "plan":
            count = len(selected) * len(protocol["conditions"])
            result = {"images": len(selected), "split": args.split, "conditions": protocol["conditions"],
                      "prepared_images_including_TRAIN_controls": (len(selected) + 2) * len(protocol["conditions"]),
                      "approaches_in_order": [job_id(j) for j in jobs], "total_evaluation_predictions": count * len(jobs),
                      "bundle": str(folder), "output": str(output), "training": False, "GPU_started": False,
                      "resume": "Repeat exactly the same run command; existing files are verified, not regenerated",
                      "disk_note": "PNG size depends on images; all per-model binary predictions are also saved."}
        elif args.stage == "prepare":
            result = bundle.prepare(root, folder, index, protocol, selected, args.split)
        elif args.stage == "verify":
            inputs, _ = bundle.verify(root, folder, definition)
            result = {"valid": True, "images": len(inputs), "GPU_started": False}
        elif args.stage == "worker":
            output_guard(output, identity)
            if args.job < 0 or args.job >= len(jobs):
                raise ValueError("Worker requires a valid planned job index")
            inputs, targets = bundle.verify(root, folder, definition)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            with gpu_lock(root):
                from .evaluation import evaluate_job
                result = evaluate_job(root, clean, output, jobs[args.job], definition,
                                      sha(folder / "complete.json"), inputs, targets, args.device)
        else:
            output_guard(output, identity)
            if args.stage == "summarize":
                bundle.verify(root, folder, definition)
                from .reporting import summarize
                result = summarize(output, jobs, definition, sha(folder / "complete.json"))
            else:
                # One coordinator per output. Children separately acquire the
                # workspace GPU lock used by the UI and training processes.
                with gpu_lock(output), logger(output) as log:
                    log.info("robustness_started training=false approaches=%s", len(jobs))
                    result = coordinate(args, root, clean, output, folder, index, protocol, selected, jobs, definition)
                    log.info("robustness_completed summary=%s", result["summary"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, ImportError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
