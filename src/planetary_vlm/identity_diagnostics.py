"""Read-only diagnostics for an existing adaptation identity; never load a model.

Kept outside adaptation/ and robustness/ so installing this diagnostic does not
change either historical training hashes or completed corruption bundle hashes.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

from planetary_vlm.adaptation.common.io import code_identity, environment, fingerprint, orbital, read, sha


def difference(expected, actual, prefix=""):
    """Show leaf differences without dropping unknown or merely added fields."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in expected:
                result.append({"field": path, "status": "added", "current": actual[key]})
            elif key not in actual:
                result.append({"field": path, "status": "missing", "training": expected[key]})
            else:
                result.extend(difference(expected[key], actual[key], path))
        return result
    if expected == actual:
        return []
    return [{"field": prefix, "status": "changed", "training": expected, "current": actual}]


def package_delta(expected, actual):
    def groups(values):
        result = defaultdict(list)
        for entry in values:
            name, separator, version = entry.partition("==")
            if not separator:
                raise ValueError("Invalid recorded distribution entry")
            result[re.sub(r"[-_.]+", "-", name).lower()].append(version)
        return {key: sorted(versions) for key, versions in result.items()}
    before, after = groups(expected), groups(actual)
    result = {
        "added": {name: after[name] for name in sorted(after.keys() - before.keys())},
        "missing": {name: before[name] for name in sorted(before.keys() - after.keys())},
        "changed_versions": {name: {"training": before[name], "current": after[name]}
                             for name in sorted(before.keys() & after.keys()) if before[name] != after[name]},
        # The training guard compares raw Name==version entries too. Do not
        # pretend normalized naming or duplicate metadata is interchangeable.
        "raw_added_entries": sorted(set(actual) - set(expected)),
        "raw_missing_entries": sorted(set(expected) - set(actual)),
    }
    return result


def report(root, run, *, full=False, device="cuda"):
    root = Path(root).resolve()
    run = (root / run).resolve()
    outputs = (root / "outputs").resolve()
    if not outputs.is_relative_to(root) or run == outputs or not run.is_relative_to(outputs):
        raise ValueError("Use an explicit original training run inside workspace outputs/")
    identity = read(run / "identity.json")
    cfg = identity["config"]
    if run.name != f"seed_{identity['seed']}" or run.parent.name != cfg["method"]:
        raise ValueError("Run directory does not match its method/seed identity")
    clean = run.parents[1]
    complete = read(run / "complete.json")
    identity_hash = sha(run / "identity.json")
    checkpoint_hash = sha(run / "best.pt")
    checkpoint_seals = {
        "identity_sha256_matches": complete.get("identity_sha256") == identity_hash,
        "checkpoint_sha256_matches": complete.get("best_checkpoint_sha256") == checkpoint_hash,
    }
    current_environment = environment()
    expected_environment = identity["environment"]
    env_difference = difference({k: v for k, v in expected_environment.items() if k != "packages"},
                                {k: v for k, v in current_environment.items() if k != "packages"}, "environment")
    packages_equal = expected_environment["packages"] == current_environment["packages"]
    dataset = {}
    for name, key in (("inputs.jsonl", "inputs_sha256"), ("targets.jsonl", "targets_sha256"), ("metadata.json", "metadata_sha256")):
        dataset[key] = sha(orbital(root, (Path(cfg["dataset"]) / name).as_posix()))
    current = {"code": code_identity(), "dataset": dataset, "stats": read(clean / "preparation/stats.json")}
    relevant = {key: identity[key] for key in current}
    changes = difference(relevant, current)
    changes.extend(env_difference)
    if not packages_equal:
        changes.append({"field": "environment.packages", "status": "changed",
                        "details": package_delta(expected_environment["packages"], current_environment["packages"])})
    result = {
        "schema": "adaptation-identity-diagnostics-v1", "run": str(run), "python_executable": sys.executable,
        "checkpoint_seals": checkpoint_seals, "differences": changes,
        "checked": ["checkpoint_file_sha256", "identity_file_sha256", "environment", "code", "dataset_manifests", "stats"],
        "not_checked": ["hardware", "assets", "cache", "math_policy", "checkpoint_embedded_identity"],
        "fast_fields_match": all(checkpoint_seals.values()) and not changes,
        "full_identity_match": None, "model_loaded": False, "files_modified": False,
        "note": "The cache fingerprint also depends on the environment; a cache difference can be derived from a package change. No bypass or automatic package mutation.",
    }
    if full:
        # context() reads existing normalization and inspects local assets, but
        # would create missing preparation files. Explicitly require both first.
        for name in ("stats.json", "identity.json"):
            if not (clean / "preparation" / name).is_file():
                raise ValueError("Original preparation files are missing; diagnostics will not create them")
        from planetary_vlm.adaptation.common.engine import context
        _, _, current_identity, _, cache = context(root, cfg, clean, device)
        current_identity = {**current_identity, "seed": identity["seed"],
                            "cache": None if cfg["method"] == "lora_decoder" else fingerprint(cache)}
        result["full_differences"] = difference(identity, current_identity)
        result["full_identity_match"] = identity == current_identity and all(checkpoint_seals.values())
        result["checked"].extend(["hardware", "assets", "cache", "math_policy"])
        result["not_checked"] = ["checkpoint_embedded_identity"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--run", default="outputs/mars_adaptation_v1/linear_probe/seed_13")
    parser.add_argument("--full", action="store_true", help="Also import Torch, query device and inspect backbone hashes; no model or inference")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    try:
        result = report(args.workspace, args.run, full=args.full, device=args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["fast_fields_match"] and result["full_identity_match"] is not False else 1
    except (KeyError, ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
