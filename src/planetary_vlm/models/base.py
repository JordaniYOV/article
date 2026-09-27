"""Shared adapter utilities; importing this module never loads ML libraries."""
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def local_model_options(options: Mapping[str, Any]) -> dict[str, Any]:
    source = str(options.get("model_path") or options.get("model_name", ""))
    if not source:
        raise ValueError("model_path or pinned cached model_name is required")
    if options.get("local_files_only", True) is not True:
        raise ValueError("Implicit model downloads disabled; local_files_only must be true")
    revision = str(options.get("revision", ""))
    if not Path(source).is_dir() and (len(revision) != 40 or any(c not in "0123456789abcdefABCDEF" for c in revision)):
        raise ValueError("Use an existing local model directory or a cached hub ID with a 40-character revision")
    result: dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    if revision:
        result["revision"] = revision
    return result


def modalities(request: Any) -> dict[str, str]:
    return dict(getattr(request, "modality_paths", ()))
