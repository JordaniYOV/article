"""Bounded paths, atomic records, exact resume identities and shared GPU lock."""
from contextlib import contextmanager
import hashlib
from importlib.metadata import distributions
import json
import logging
import os
from pathlib import Path
import platform
import sys


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def atomic(path, value, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in value) if lines else
               json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def guard(path, identity):
    path = Path(path)
    if path.exists() and read(path) != identity:
        raise ValueError(f"Identity changed; use a new experiment directory: {path.parent}")
    if not path.exists():
        atomic(path, identity)


def orbital(root, relative, digest=None):
    root, part = Path(root).resolve(), Path(relative)
    if part.is_absolute() or not part.parts or part.parts[0] != "data_orbital" or ".." in part.parts:
        raise ValueError("Only explicit data_orbital files are allowed")
    path = (root / part).resolve()
    if not path.is_relative_to(root / "data_orbital") or not path.is_file():
        raise ValueError(f"Missing or unsafe orbital asset: {part}")
    if digest is not None and sha(path) != digest:
        raise ValueError(f"SHA-256 mismatch: {part}")
    return path


def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Asset must stay within its declared directory")
    return path


def experiment(root, output):
    root = Path(root).resolve()
    path = (root / output).resolve()
    base = (root / "outputs").resolve()
    if not base.is_relative_to(root) or path == base or not path.is_relative_to(base):
        raise ValueError("Use a dedicated experiment inside workspace outputs/")
    marker = path / "experiment.json"
    if path.exists() and any(path.iterdir()) and not marker.is_file():
        raise ValueError("Existing non-experiment output directory is protected")
    guard(marker, {"schema": "mars-adaptation-v1"})
    return path


def code_identity():
    base = Path(__file__).parents[1]
    return {p.relative_to(base).as_posix(): sha(p) for pattern in ("*.py", "*.toml") for p in sorted(base.rglob(pattern))}


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata['Name'])}


@contextmanager
def logger(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("adaptation." + str(directory))
    log.setLevel(logging.INFO)
    log.propagate = False
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(directory / "events.log", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        log.addHandler(handler)
    try:
        yield log
    finally:
        for handler in handlers:
            handler.close()
            log.removeHandler(handler)


@contextmanager
def gpu_lock(root):
    path = within(root, ".state/api-worker.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if not stream.seek(0, 2):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("GPU is reserved by training or a UI worker") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
