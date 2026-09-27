"""Portable request manifests with a strict model-visible allowlist."""

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

from planetary_vlm.contracts import ModelRequest
from planetary_vlm.io import read_jsonl

REQUEST_FIELDS = frozenset(field.name for field in fields(ModelRequest))


def request_from_dict(row: Mapping[str, Any], root: Path | None = None) -> ModelRequest:
    unknown = set(row) - REQUEST_FIELDS
    if unknown:
        raise ValueError(f"Forbidden/unknown request fields: {sorted(unknown)}")
    payload = dict(row)
    for name in ("image_paths", "allowed_answers", "modality_paths"):
        default = [] if name != "allowed_answers" else None
        value = payload.get(name, default)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be an array")
        if name == "modality_paths":
            if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in value):
                raise ValueError("modality_paths requires name/path pairs")
            payload[name] = tuple(tuple(pair) for pair in value)
        else:
            payload[name] = tuple(value)
    try:
        request = ModelRequest(**payload)
    except TypeError as error:
        raise ValueError(f"Malformed model request: {error}") from error
    if root is not None:
        def resolve(value: str) -> str:
            path = Path(value)
            return str((path if path.is_absolute() else root / path).resolve())
        payload = asdict(request)
        payload["image_paths"] = tuple(resolve(path) for path in request.image_paths)
        payload["modality_paths"] = tuple((name, resolve(path)) for name, path in request.modality_paths)
        request = ModelRequest(**payload)
    return request


def load_requests(path: str | Path, *, resolve_paths: bool = True) -> list[ModelRequest]:
    path = Path(path).resolve()
    root = path.parent if resolve_paths else None
    requests = [request_from_dict(row, root) for row in read_jsonl(path)]
    ids = [request.request_id for request in requests]
    if not ids:
        raise ValueError("Request manifest is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate request_id in request manifest")
    return requests


def check_assets(requests: list[ModelRequest]) -> None:
    for request in requests:
        for path in (*request.image_paths, *(path for _, path in request.modality_paths)):
            if not Path(path).exists():
                raise ValueError(f"Missing input asset for {request.request_id}: {path}")
