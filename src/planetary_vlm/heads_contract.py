"""CPU-only contract for trained catalog-crater heads; no Torch or target inference."""
import hashlib
import json
from pathlib import Path

TASK = "catalog_crater_segmentation"
INPUT_PROTOCOL = "frozen-wac-native-clean-v1"
CLASSES = ["Background", "CatalogCrater"]


def pixel_hash(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        if image.size != (512, 512):
            raise ValueError("Frozen WAC inputs must be square 512x512")
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def read_deployment(path, expected_sha, root):
    """Check self-contained deployment metadata, before deserializing weights."""
    from benchmark_services.common.files import file_hash, safe_path
    root = Path(root).resolve()
    path = safe_path(str(path), root / "outputs")
    if file_hash(path) != expected_sha:
        raise ValueError("Head deployment SHA-256 differs")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["schema"] != "frozen-crater-deployment-v1" or data["task"] != TASK or data["input_protocol"] != INPUT_PROTOCOL:
        raise ValueError("Unsupported trained-head deployment contract")
    if data["encoder"] not in {"ibm", "spacellava"} or data["family"] not in {"linear", "spatial"} or not 0 < data["threshold"] < 1:
        raise ValueError("Invalid head/encoder/validation threshold")
    for name, digest in data["files_sha256"].items():
        asset = safe_path(name, path.parent)
        if file_hash(asset) != digest:
            raise ValueError(f"Deployment file changed: {name}")
    if not {"head.pt", "encoder_inputs.jsonl"} <= set(data["files_sha256"]):
        raise ValueError("Deployment lacks weights or input-only provenance")
    inputs = [json.loads(line) for line in (path.parent / "encoder_inputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not inputs or len({r["sample_id"] for r in inputs}) != len(inputs):
        raise ValueError("Missing/duplicate deployed input IDs")
    forbidden = {"mask", "mask_sha256", "answer", "targets", "height_map", "outcome", "reference_status"}
    if any(forbidden.intersection(row) or row.get("split") not in {"val", "test"} for row in inputs):
        raise ValueError("Deployment inputs cannot contain targets or training examples")
    return path, data
