"""Bounded ZIP ingestion and orbital manifests; no model or target inference."""

from io import BytesIO
import hashlib
import http.client
import ipaddress
import json
from pathlib import Path, PurePosixPath
import shutil
import socket
import stat
import time
from urllib.parse import urlsplit, urljoin
from uuid import uuid4
import zipfile

from PIL import Image
from fastapi import HTTPException
from sqlmodel import select

from . import settings as paths
from .models import DatasetConfig, DatasetConfigCreate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, default=str).encode()).hexdigest()


def file_hash(path):
    from planetary_vlm.inference.runner import file_hash as implementation
    return implementation(Path(path))


def safe_path(value, root):
    root = Path(root).resolve()
    result = (root / value).resolve()
    if not result.is_relative_to(root) or "data_rover" in {part.lower() for part in result.parts}:
        raise ValueError("Path leaves its permitted directory")
    return result


def read_rows(manifest):
    manifest = paths.orbital_path(str(manifest))
    result = []
    with manifest.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Manifest rows must be JSON objects")
                result.append(row)
    return result


def sample_rows(dataset):
    """Read only manifest/image bytes; keep targets in metadata for later scoring."""
    rows = read_rows(dataset.manifest_path)
    selected, seen = [], set()
    for row in rows:
        if row.get("evaluation_eligible") is False:
            continue
        source_filter = dataset.provenance.get("source_dataset")
        if source_filter and row.get("source_dataset") != source_filter:
            continue
        if dataset.task_id == "imp_segmentation" and row.get("mask_semantics", "imp") != "imp":
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("Source sample IDs must be nonempty and unique")
        seen.add(sample_id)
        split = row.get("source_split", row.get("split", "test"))
        if split not in dataset.splits:
            continue
        image = paths.orbital_path(row["image"])
        checksum = file_hash(image)
        if row.get("image_sha256") and row["image_sha256"] != checksum:
            raise ValueError(f"Changed source image: {sample_id}")
        # Path validation without opening ground truth.
        for field in ("mask", "dem"):
            if row.get(field):
                paths.orbital_path(row[field])
        selected.append({**row, "image": str(image), "image_sha256": checksum,
            "source_split": split, "scene_group_id": row.get("scene_group_id", sample_id)})
    return sorted(selected, key=lambda item: item["sample_id"])


def ingest_zip(stream, metadata, settings, session):
    """Extract only data files into a fresh server-owned orbital directory."""
    folder = paths.WORKSPACE_ROOT / "data_orbital" / "uploads" / uuid4().hex
    folder.mkdir(parents=True)
    try:
        archive_hash = hashlib.sha256()
        stream.seek(0)
        compressed_size = 0
        while chunk := stream.read(1024 * 1024):
            compressed_size += len(chunk)
            if compressed_size > settings.max_upload_bytes:
                raise ValueError("Archive exceeds compressed size limit")
            archive_hash.update(chunk)
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            if len(entries) > settings.max_archive_files or sum(e.file_size for e in entries) > settings.max_unpacked_bytes:
                raise ValueError("Archive exceeds unpacked size or file count limits")
            seen = set()
            for entry in entries:
                name = entry.filename
                relative = PurePosixPath(name)
                if (not name or "\\" in name or relative.is_absolute() or ".." in relative.parts
                    or any(":" in part or part.endswith((".", " ")) for part in relative.parts)
                    or any(ord(char) < 32 for char in name)
                    or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1):
                    raise ValueError("Unsafe archive member")
                if entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}:
                    raise ValueError("Unsupported ZIP compression")
                if any(part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL",
                        *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))} for part in relative.parts):
                    raise ValueError("Reserved archive filename")
                target = safe_path(name, folder)
                key = str(target).casefold()
                if key in seen:
                    raise ValueError("Duplicate archive path")
                seen.add(key)
                if entry.is_dir():
                    continue
                if target.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".json", ".jsonl"}:
                    raise ValueError("Only images and JSON metadata are accepted")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        supplied = folder / "manifest.jsonl"
        if supplied.exists():
            rows = read_rows(supplied)
        else:
            if (folder / "masks").exists() or (folder / "targets").exists():
                raise ValueError("Archives with targets require manifest.jsonl")
            images_root = folder / "images" if (folder / "images").is_dir() else folder
            images = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
            rows = [{"sample_id": f"sample_{i:06d}", "image": str(p.relative_to(folder)),
                "source_split": "test"} for i, p in enumerate(images)]
        if not rows or len(rows) > settings.max_archive_files:
            raise ValueError("Archive needs a nonempty image manifest")
        normalized, seen_ids = [], set()
        for row in rows:
            sample_id = row["sample_id"]
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
                raise ValueError("Duplicate/invalid sample_id")
            seen_ids.add(sample_id)
            item = dict(row)
            if item.get("mask") and item.get("dem"):
                raise ValueError("A sample may have either a segmentation mask or a DEM")
            for field in ("image", "mask", "dem"):
                if not item.get(field):
                    continue
                asset = safe_path(item[field], folder)
                if not asset.is_file():
                    raise ValueError(f"Missing {field} file")
                if field == "image":
                    with Image.open(asset) as image:
                        image.verify()
                checksum = file_hash(asset)
                if item.get(field + "_sha256") and item[field + "_sha256"] != checksum:
                    raise ValueError("Uploaded asset checksum mismatch")
                item[field], item[field + "_sha256"] = str(asset), checksum
            if "image" not in item:
                raise ValueError("Every sample requires an image")
            item["source_split"] = item.get("source_split", item.get("split", "test"))
            if item["source_split"] not in {"train", "val", "test"}:
                raise ValueError("Unknown split")
            item.setdefault("scene_group_id", sample_id)
            normalized.append(item)
        manifest = folder / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in normalized), encoding="utf-8")
        payload = DatasetConfigCreate(name=metadata.name, version=metadata.version,
            planet=metadata.planet, task_id=metadata.task_id, manifest_path=str(manifest),
            splits=sorted({row["source_split"] for row in normalized}),
            preprocessing=metadata.preprocessing,
            provenance={**metadata.provenance, "import_version": "orbital-zip-v1",
                "archive_sha256": archive_hash.hexdigest(),
                "manifest_sha256": file_hash(manifest), "sample_count": len(rows)})
        if session.exec(select(DatasetConfig.id).where(DatasetConfig.name == payload.name,
                DatasetConfig.version == payload.version)).first() is not None:
            raise HTTPException(409, "Dataset version already exists")
        values = payload.model_dump()
        record = DatasetConfig.model_validate({**values, "config_sha256": digest(values)})
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    except Exception:
        session.rollback()
        # Only the freshly created UUID directory owned by this ingestion.
        checked = safe_path(folder.name, paths.WORKSPACE_ROOT / "data_orbital" / "uploads")
        shutil.rmtree(checked)
        raise


def download_zip(url, settings):
    """HTTP(S), public addresses, pinned DNS, bounded body and checked redirects."""
    deadline = time.monotonic() + settings.download_timeout
    for _ in range(4):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Use a direct HTTP(S) URL without credentials")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise ValueError("Private and local download addresses are not allowed")
        address = addresses[0][4][0]
        timeout = max(.1, deadline - time.monotonic())
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection = cls(parsed.hostname, port, timeout=timeout)
        # TLS still verifies the original hostname; TCP connects to the checked IP.
        connection._create_connection = lambda _, timeout, source_address=None: socket.create_connection(
            (address, port), timeout, source_address)
        try:
            resource = parsed.path or "/"
            if parsed.query:
                resource += "?" + parsed.query
            connection.request("GET", resource, headers={"Accept": "application/zip"})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise ValueError("Redirect has no destination")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f"Download returned HTTP {response.status}")
            if int(response.getheader("Content-Length", "0")) > settings.max_upload_bytes:
                raise ValueError("Download exceeds size limit")
            content = BytesIO()
            while True:
                if time.monotonic() > deadline:
                    raise ValueError("Download exceeded time limit")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                content.write(chunk)
                if content.tell() > settings.max_upload_bytes:
                    raise ValueError("Download exceeds size limit")
            content.seek(0)
            return content
        finally:
            connection.close()
    raise ValueError("Too many redirects")
