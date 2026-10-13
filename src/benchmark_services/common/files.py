"""Shared bounded paths, hashes and atomic JSON; no models or GT access."""

import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, default=str).encode()).hexdigest()


def file_hash(path):
    from planetary_vlm.inference.runner import file_hash as implementation
    return implementation(Path(path))


def safe_path(value, root):
    root = Path(root).resolve()
    result = (root / value).resolve()
    if not result.is_relative_to(root) or "data_rover" in {part.casefold() for part in result.parts}:
        raise ValueError("Path leaves its permitted directory")
    return result


def atomic_json(path, value, *, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        if lines:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, default=str, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
