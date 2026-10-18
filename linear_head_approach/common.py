"""Small I/O and integrity contract; importing this module never loads torch."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tomllib
from contextlib import contextmanager
from functools import wraps
from importlib.metadata import distributions
from pathlib import Path

from . import VERSION

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name("config.toml")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records), encoding="utf-8")
    os.replace(temporary, path)


def safe_asset(root, relative, expected=None):
    """Only explicit data_orbital assets; no traversal, symlinks or archive access."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or relative.parts[0] != "data_orbital" or ".." in relative.parts:
        raise ValueError(f"Unsafe orbital path: {relative}")
    path = root / relative
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError(f"Symbolic links not admitted for data: {relative}")
    if not path.is_file() or not path.resolve().is_relative_to(root / "data_orbital"):
        raise ValueError(f"Missing orbital asset: {relative}")
    if expected is not None and sha(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    return path


def config(path=CONFIG):
    with Path(path).open("rb") as stream:
        cfg = tomllib.load(stream)
    if cfg["schema"] != VERSION or cfg["common_grid"] != 24 or cfg["loss_grid"] < 24:
        raise ValueError("Unsupported protocol / grid")
    if cfg["feature_augmentation"] or cfg["feature_precision"] != "float32":
        raise ValueError("This protocol uses FP32, unaugmented cached features")
    if cfg["selection_metric"] != "validation_scene_macro_iou":
        raise ValueError("Unknown selection rule")
    for key in ("max_epochs", "patience", "batch_size", "hidden_channels"):
        if cfg[key] < 1:
            raise ValueError(f"Invalid {key}")
    return cfg


def environment():
    packages = sorted({f"{d.metadata['Name'].lower()}=={d.version}" for d in distributions() if d.metadata['Name']})
    return {"python": platform.python_version(), "packages": packages}


def code_hash():
    return fingerprint({p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def logger(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("frozen_heads")
    close_logger(log)
    log.setLevel(logging.INFO)
    log.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(directory / "events.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def close_logger(log):
    for handler in list(log.handlers):
        handler.flush()
        handler.close()
        log.removeHandler(handler)


def closes_stage_logs(function):
    """Release Windows log handles on completion AND interruption/failed stages."""
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        finally:
            close_logger(logging.getLogger("frozen_heads"))
    return call


def identity_guard(path, identity):
    """Refuse cache/run reuse if dependencies, code, inputs, model or config changed."""
    path = Path(path)
    if path.exists() and read_json(path) != identity:
        raise ValueError(f"Identity changed; choose a NEW output directory: {path.parent}")
    if not path.exists():
        atomic_json(path, identity)


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


@contextmanager
def gpu_reservation(root=ROOT):
    """Same OS-released lock as UI workers: CLI training cannot overlap inference."""
    path = Path(root) / ".state/api-worker.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        locked = False
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise RuntimeError("GPU is reserved by CLI training or a UI worker. Finish that job first.") from error
        try:
            yield
        finally:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

