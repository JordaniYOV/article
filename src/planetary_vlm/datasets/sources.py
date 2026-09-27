"""Collect source evidence and a small pilot; never fabricate a complete benchmark.

The lunar inventory is a candidate list, not an admitted evaluation manifest.
Only the requested pilot triplets are downloaded from LuMonDepth. Mars-Bench's
small archive is optional. No models, synthetic replacements or inferred labels
are used.
"""

from datetime import datetime, timezone
import hashlib
import json
import urllib.request


USER_AGENT = "planetary-vlm-benchmark/source-acquisition-v1"


def request(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=45
    )


def read_json(url):
    with request(url) as response:
        if 'rel="next"' in response.headers.get("Link", ""):
            raise ValueError("Paginated inventory: refusing an incomplete source count")
        return json.load(response)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download(url, path, expected_size=None, checksum=None):
    """Reuse only validated downloads and bound response size to the source metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def valid(file):
        if expected_size is not None and file.stat().st_size != expected_size:
            return False
        if checksum:
            algorithm, expected = checksum.split(":", 1)
            with file.open("rb") as stream:
                actual = hashlib.file_digest(stream, algorithm).hexdigest()
            if actual != expected:
                return False
        return True

    if path.exists():
        if not valid(path):
            raise ValueError(f"Existing source file failed validation: {path}")
    else:
        partial = path.with_name(path.name + ".partial")
        with request(url) as response, partial.open("wb") as stream:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if expected_size is not None and total > expected_size:
                    raise ValueError("Response exceeds source metadata size")
                stream.write(chunk)
        if not valid(partial):
            raise ValueError(f"Downloaded source failed validation: {partial}")
        partial.rename(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "size": path.stat().st_size,
            "sha256": digest, "url": url}


def collect(root, pilot_ids, mars_archive, domains=("moon", "mars")):
    if not set(domains) or set(domains) - {"moon", "mars"}:
        raise ValueError("Select moon and/or mars sources")
    evidence = root / "source_audit_v1"
    root.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "1.0", "checked_at": datetime.now(timezone.utc).isoformat(),
              "requested_lunar_images": 300, "depth_policy": "provided_or_stereo_only",
              "status": "incomplete", "complete_admitted_lunar_triplets": 0,
              "sources": {}, "errors": []}
    report["selected_domains"] = list(domains)
    report["requested_lunar_images"] = 300 if "moon" in domains else 0

    if "moon" in domains:
        from .lune import LuneData
        LuneData(root)._collect_sources(pilot_ids, report)
    if "mars" in domains:
        from .mars import MarsData
        MarsData(root)._collect_sources(mars_archive, report)

    report["blockers"] = [
        "No verified source currently supplies 300 real lunar RGB + semantic/instance segmentation + stereo depth triplets",
        "Masks from unrelated missions cannot be attached to ChangE-3 depth frames",
        "Observed traversal difficulty requires image-aligned movement outcomes, absent from these semantic sources",
    ]
    report_name = "collection_report.json" if len(domains) == 2 else f"{domains[0]}_collection_report.json"
    save_json(evidence / report_name, report)
    print(json.dumps({"report": str((evidence / report_name).resolve()),
                      "sources": {key: {k: v for k, v in value.items() if k != "assets" and k != "pilot_assets"}
                                  for key, value in report["sources"].items()},
                      "errors": report["errors"], "status": report["status"]}, ensure_ascii=False, indent=2))

    return report
