"""Local runtime checks shared by the two complete-system adapters."""

from contextlib import contextmanager
import gc
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys

from planetary_vlm.inference.runner import file_hash


def local_path(options, key, *, directory=False):
    value = options.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Required local option: {key}")
    path = Path(value).resolve()
    if "data_rover" in {part.lower() for part in path.parts}:
        raise ValueError("Protected archive cannot be a runtime/input path")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Missing local {key}: {path}")
    return path


def checked_repo(options, package):
    repo = local_path(options, "runtime_repo", directory=True)
    revision = options.get("code_revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("code_revision must be an exact 40-character Git commit")
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    if actual != revision or dirty.strip():
        raise ValueError(f"Runtime must be clean and at commit {revision}")
    if not (repo / package / "__init__.py").is_file():
        raise ValueError(f"Runtime does not contain {package}")
    return repo


@contextmanager
def offline_imports(repo):
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    previous_path = list(sys.path)
    try:
        os.environ.update({key: "1" for key in keys})
        sys.path.insert(0, str(repo))
        yield
    finally:
        sys.path[:] = previous_path
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def import_local(package, repo):
    module = importlib.import_module(package)
    if not Path(module.__file__).resolve().is_relative_to(repo):
        raise RuntimeError(f"Conflicting installed {package}; use a fresh process")
    return module


def hashes(paths):
    return {str(path): file_hash(path) for path in sorted(set(paths))}


def release(adapter):
    adapter.model = None
    gc.collect()
    if adapter.torch is not None and adapter.torch.cuda.is_available():
        adapter.torch.cuda.empty_cache()
