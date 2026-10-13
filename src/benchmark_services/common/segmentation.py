"""Portable segmentation artifacts and IMP display input checks; no model imports."""

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

from .files import file_hash


def restore_imp_png(path, maximum=0.12):
    """Approximate reflectance from the documented 8-bit display transform.

    Clipping, quantization and lost nodata are irreversible. This is an
    exploratory PNG input track, not a reproduction using the original TIFFs.
    """
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.shape != (256, 256):
        raise ValueError("IMP PNG track requires a grayscale uint8 256x256 image")
    return pixels.astype(np.float32) * (maximum / 255.0)

def save_maps(directory, request_id, logits):
    """Persist raw two-class logits, probabilities and label IDs atomically."""
    import numpy as np
    from PIL import Image
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (2, 256, 256) or not np.isfinite(logits).all():
        raise ValueError("Expected finite IBM logits shaped (2,256,256)")
    scores = np.exp(logits - logits.max(axis=0, keepdims=True))
    probabilities = scores / scores.sum(axis=0, keepdims=True)
    labels = logits.argmax(axis=0).astype(np.uint8)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(request_id.encode()).hexdigest()
    arrays = BytesIO()
    np.savez_compressed(arrays, logits=logits, probabilities=probabilities)
    png = BytesIO()
    Image.fromarray(labels).save(png, format="PNG")
    files = {}
    for suffix, content in ((".npz", arrays.getvalue()), (".png", png.getvalue())):
        destination = directory / (name + suffix)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        files[destination.name] = hashlib.sha256(content).hexdigest()
    return {"output_type": "imp_segmentation", "classes": ["Background", "IMP"],
            "shape": [256, 256], "decision": "argmax_of_two_logits",
            "map_directory": str(directory.resolve()), "files_sha256": files,
            "label_png": name + ".png", "arrays_npz": name + ".npz"}

def verify_saved_maps(raw_response):
    payload = json.loads(raw_response)
    root = Path(payload["map_directory"]).resolve()
    for name, digest in payload["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != digest:
            raise ValueError("Resume rejected: a saved IBM map is missing or changed")
