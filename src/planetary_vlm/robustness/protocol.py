"""Read-only planning; no Torch, model construction or checkpoint unpickling."""
from pathlib import Path
import re
import tomllib

from planetary_vlm.adaptation.common.config import METHODS, validate
from planetary_vlm.adaptation.common.data import Index
from planetary_vlm.adaptation.common.io import code_identity, fingerprint, read, sha, within


def load_protocol(path=None):
    with (Path(path) if path else Path(__file__).with_name("protocol.toml")).open("rb") as stream:
        cfg = tomllib.load(stream)
    if set(cfg) != {"schema", "seed", "conditions"} or cfg["schema"] != "mars-robustness-v1":
        raise ValueError("Expected robustness schema, seed and conditions only")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("Declare a nonnegative corruption seed")
    if not isinstance(cfg["conditions"], list):
        raise ValueError("Conditions must be an array of TOML tables")
    from benchmark_services.common.transforms import PARAMETERS
    seen = set()
    for condition in cfg["conditions"]:
        if not isinstance(condition, dict):
            raise ValueError("Conditions must be tables")
        if not {"id", "chain"} <= set(condition) or set(condition) - {"id", "chain", "parameters"}:
            raise ValueError("A condition requires id, chain and optional parameters")
        cid, chain = condition["id"], condition["chain"]
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", cid) or cid in seen:
            raise ValueError("Unsafe or duplicate condition ID")
        if (not isinstance(chain, list) or not all(isinstance(e, str) for e in chain)
                or len(set(chain)) != len(chain) or set(chain) - PARAMETERS.keys()):
            raise ValueError("Unknown/repeated effect in condition chain")
        condition.setdefault("parameters", {})
        if not isinstance(condition["parameters"], dict) or set(condition["parameters"]) != set(chain):
            raise ValueError("Explicit parameter tables required for every effect")
        for effect in chain:
            if not isinstance(condition["parameters"][effect], dict) or set(condition["parameters"][effect]) - PARAMETERS[effect]:
                raise ValueError(f"Unknown parameters for {effect}")
        if (cid == "clean") != (not chain):
            raise ValueError("Only the clean condition may have an empty chain")
        seen.add(cid)
    if "clean" not in seen or len(seen) < 2:
        raise ValueError("Declare clean and at least one corruption condition")
    return cfg


def dedicated(root, relative, base):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    parent = (root / base).resolve()
    if not parent.is_relative_to(root) or path == parent or not path.is_relative_to(parent):
        raise ValueError(f"Use a dedicated directory inside {base}/")
    return path


def discover(root, clean_output, methods=METHODS):
    """Only explicit, trusted local runs; verify seals before any torch.load."""
    clean = dedicated(root, clean_output, "outputs")
    if len(set(methods)) != len(methods) or not methods or set(methods) - set(METHODS):
        raise ValueError("Unknown/duplicate approach")
    jobs, common, reference = [], None, None
    current_code = code_identity()
    # Canonical order, even when options are provided in a different order.
    for method in METHODS:
        if method not in methods:
            continue
        folder = within(clean, method)
        if method == "sam_baseline":
            marker = folder / "smoke.json"
            identity = read(marker)["identity"]
            cfg = identity["config"]
            candidates = [(None, identity, {"smoke_sha256": sha(marker)})]
        else:
            first = read(folder / "seed_13/identity.json")
            cfg = first["config"]
            if cfg["seeds"] != [13, 37, 73]:
                raise ValueError("This comparison requires all declared seeds 13/37/73")
            candidates = []
            for seed in cfg["seeds"]:
                run = within(folder, f"seed_{seed}")
                identity = read(run / "identity.json")
                complete = read(run / "complete.json")
                seal = {"checkpoint_sha256": sha(run / "best.pt"), "identity_sha256": sha(run / "identity.json")}
                if any(complete.get(key) != value for key, value in {
                    "best_checkpoint_sha256": seal["checkpoint_sha256"],
                    "identity_sha256": seal["identity_sha256"], "method": method, "seed": seed,
                }.items()):
                    raise ValueError(f"Incomplete/changed trusted training run: {run}")
                if identity["config"] != cfg or identity["seed"] != seed:
                    raise ValueError("Training seed/config differs")
                candidates.append((seed, identity, seal))
        validate(cfg)
        if cfg["method"] != method:
            raise ValueError("Training identity belongs to another approach")
        signature = {k: cfg[k] for k in ("dataset", "counts", "encoder")}
        if common is not None and common != signature:
            raise ValueError("Approaches must use identical source data and input preprocessing")
        common = signature
        for seed, identity, seal in candidates:
            if identity["code"] != current_code:
                raise ValueError("Original adaptation code changed; restore it, do not republish/retrain")
            index = Index(root, cfg)
            if identity["dataset"] != index.identity:
                raise ValueError("Original training dataset identity changed")
            reference = index if reference is None else reference
            if seed is not None:
                stats = read(clean / "preparation/stats.json")
                if stats != identity["stats"] or not (clean / "preparation/identity.json").is_file():
                    raise ValueError("Original TRAIN normalization missing/changed")
            jobs.append({"method": method, "seed": seed, "config": cfg,
                         "source_identity_sha256": fingerprint(identity), **seal})
    return clean, reference, jobs


def selection(index, split, limit, seed):
    if split not in ("val", "test") or type(limit) is not int or limit < 0:
        raise ValueError("Use val/test and a nonnegative image limit (0 = all)")
    selected = index.split(split)
    if limit > len(selected):
        raise ValueError("Requested image limit exceeds the split")
    if limit:
        # Blind, deterministic subset: never inspect labels or model outcomes.
        selected = sorted(selected, key=lambda r: fingerprint([seed, r["sample_id"]]))[:limit]
    return sorted(selected, key=lambda r: r["sample_id"])


def job_id(job):
    return job["method"] + (f"/seed_{job['seed']}" if job["seed"] is not None else "")
