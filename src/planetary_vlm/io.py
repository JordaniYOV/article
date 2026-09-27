"""Strict JSONL input and durable per-response output."""

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Any


def records_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    """Canonical manifest identity, independent of JSON whitespace/row order."""
    canonical = sorted((dict(row) for row in rows), key=lambda row: row["request_id"])
    content = json.dumps(canonical, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
