"""CPU-only contract for trained catalog-crater heads; no Torch or target inference."""
import hashlib
import json
from pathlib import Path

TASK = "catalog_crater_segmentation"
INPUT_PROTOCOL = "frozen-wac-native-clean-v1"
TRANSFER_PROTOCOL = "frozen-crater-image-clean-v2"
CLASSES = ["Background", "CatalogCrater"]
CRATER_SEMANTICS = {"catalog_crater_polygon", "publisher_binary_crater_segmentation"}


def transfer_input(source, root, encoder=None):
    """Validate external held-out inputs and return an input-only encoder row.

    The deployed weights/threshold remain unchanged. Never synthesize VIS bands
    or copy masks, answers or other reference fields into model options.
    """
    from benchmark_services.common.files import file_hash, safe_path
    root = Path(root).resolve()
    split = source.get("source_split", source.get("split"))
    if split not in {"val", "test"} or source.get("evaluation_eligible") is False:
        raise ValueError("Frozen-head evaluation requires held-out val/test images, not train")
    image = safe_path(root / source.get("image", source.get("rgb_image", "")), root / "data_orbital")
    checksum = file_hash(image)
    if checksum != source.get("image_sha256", source.get("rgb_image_sha256")):
        raise ValueError("External head source image SHA-256 changed")
    pixels = pixel_hash(image)
    # The collection hashes size+RGB bytes; legacy deployments hash RGB bytes.
    # Verify either documented schema, then pass the canonical legacy hash on.
    if source.get("rgb_pixels_sha256") and source["rgb_pixels_sha256"] not in {pixels, pixel_hash(image, size_bound=True)}:
        raise ValueError("External head source pixels changed")
    row = {"sample_id": source["sample_id"], "scene_group_id": source["scene_group_id"],
           "split": split, "rgb_image": image.relative_to(root).as_posix(),
           "rgb_image_sha256": checksum, "rgb_pixels_sha256": pixels}
    native = source.get("native_vis")
    if encoder == "ibm" and not native:
        raise ValueError("This IBM head requires real native VIS5 (5x512x512). "
                         "The Mars PNG has no VIS5; use CLIP or a separately validated IBM input adapter")
    if native:
        path = safe_path(root / native, root / "data_orbital")
        if file_hash(path) != source.get("native_vis_sha256"):
            raise ValueError("Paired native VIS input is missing or changed")
        row.update(native_vis=path.relative_to(root).as_posix(), native_vis_sha256=source["native_vis_sha256"])
    return row


def reject_training_overlap(sources, root, expected_sha=None):
    """Keep the local training snapshot a denylist, not an evaluation allowlist."""
    from benchmark_services.common.files import file_hash
    path = Path(root) / "data_orbital/head_training_v2/inputs.jsonl"
    if not path.is_file():
        if expected_sha:
            raise ValueError("Training input snapshot is required to audit external evaluation overlap")
        return
    if expected_sha and file_hash(path) != expected_sha:
        raise ValueError("Training input snapshot changed; cannot audit external evaluation overlap")
    training = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    training = [row for row in training if row.get("split") == "train"]
    ids = {row["sample_id"] for row in training}
    images = {row["rgb_image_sha256"] for row in training}
    natives = {row.get("native_vis_sha256") for row in training} - {None}
    for row in sources:
        if (row["sample_id"] in ids or row.get("image_sha256") in images
                or row.get("native_vis_sha256") in natives):
            raise ValueError("External benchmark overlaps the head training images")


def pixel_hash(path, *, size_bound=False):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        if image.size != (512, 512):
            raise ValueError("Frozen crater-head inputs must be square 512x512")
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    prefix = str((512, 512)).encode() if size_bound else b""
    return hashlib.sha256(prefix + pixels.tobytes()).hexdigest()


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
