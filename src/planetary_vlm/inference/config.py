"""Resolve one backend run without selecting a specific architecture in core."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import tomllib
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    requests: Path
    output_dir: Path
    model_id: str
    backend: str
    options: dict[str, Any]
    seed: int
    config_sha256: str
    config_path: Path


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    content = path.read_bytes()
    config = tomllib.loads(content.decode("utf-8-sig"))
    if set(config) - {"run", "model"}:
        raise ValueError("Run config accepts only [run] and [model]")
    run, model = config.get("run", {}), config.get("model", {})
    if set(run) - {"run_id", "requests", "output_dir", "seed"}:
        raise ValueError("Unknown run configuration fields")
    if set(model) - {"id", "backend", "options"}:
        raise ValueError("Unknown model configuration fields")
    for table, names in ((run, ("run_id", "requests", "output_dir")), (model, ("id", "backend"))):
        for name in names:
            if not isinstance(table.get(name), str) or not table[name].strip():
                raise ValueError(f"Missing nonempty config field: {name}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run["run_id"]):
        raise ValueError("run_id must be a simple name without path separators")
    seed = run.get("seed", 42)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    options = model.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("model.options must be a table")
    def resolve(value: str) -> Path:
        candidate = Path(value)
        return (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
    # Shared convention for all adapters: declared local path options are
    # relative to their TOML, while checkpoint IDs/model names stay literal.
    options = dict(options)
    for name, value in options.items():
        if name.endswith(("_path", "_dir", "_repo")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty local path")
            options[name] = str(resolve(value))
    return RunConfig(run["run_id"], resolve(run["requests"]), resolve(run["output_dir"]),
                     model["id"], model["backend"], options, seed,
                     hashlib.sha256(content).hexdigest(), path)
