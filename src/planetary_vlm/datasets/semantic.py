"""Generic semantic-mask converter; dataset-native mappings are supplied explicitly."""

from collections import Counter
from dataclasses import asdict
import hashlib
from pathlib import Path
import tomllib

from planetary_vlm.contracts import ModelRequest
from planetary_vlm.io import read_jsonl, write_json, write_jsonl


def convert_semantic(samples_path: str | Path, specification_path: str | Path,
                     destination: str | Path) -> Path:
    """Create terrain/rock questions on cropped ROIs, with GT in a separate file."""
    from PIL import Image

    samples_path = Path(samples_path).resolve()
    specification_path = Path(specification_path).resolve()
    config_bytes = specification_path.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8-sig"))
    permitted = {"label_to_class", "ignore_labels", "roi", "min_valid_fraction",
                 "dominant_min_fraction", "rock_min_pixels", "rock_classes", "tasks"}
    if set(config) - permitted:
        raise ValueError("Unknown semantic conversion parameters")
    mapping = config.get("label_to_class", {})
    if not mapping or any(not isinstance(value, str) or not value.strip() for value in mapping.values()):
        raise ValueError("Provide dataset-native label_to_class mapping")
    try:
        mapping = {int(key): value for key, value in mapping.items()}
    except ValueError as error:
        raise ValueError("Mask labels must be integers") from error
    ignore = config.get("ignore_labels", [])
    if not isinstance(ignore, list) or any(type(value) is not int for value in ignore):
        raise ValueError("ignore_labels must be integer values")
    if set(ignore) & set(mapping):
        raise ValueError("Ignore labels must not also map to terrain classes")
    roi = config.get("roi")
    if (not isinstance(roi, list) or len(roi) != 4 or
            any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 <= value <= 1 for value in roi) or roi[0] >= roi[2] or roi[1] >= roi[3]):
        raise ValueError("Supply normalized ROI [left,top,right,bottom]")
    for key in ("min_valid_fraction", "dominant_min_fraction"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(f"Specify {key} in (0,1]")
    tasks = config.get("tasks")
    if (not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks)
            or set(tasks) - {"terrain", "rock_presence"}):
        raise ValueError("tasks must select terrain and/or rock_presence")
    rock_classes = config.get("rock_classes", [])
    if "rock_presence" in tasks:
        if (not isinstance(rock_classes, list) or not rock_classes
                or any(not isinstance(label, str) for label in rock_classes)
                or len(set(rock_classes)) != len(rock_classes)
                or set(rock_classes) - set(mapping.values())):
            raise ValueError("Supply native rock_classes from label_to_class")
        if type(config.get("rock_min_pixels")) is not int or config["rock_min_pixels"] < 1:
            raise ValueError("Specify positive rock_min_pixels")
    terrain_answers = tuple(sorted(set(mapping.values()))) + ("MIXED",)
    if "MIXED" in mapping.values():
        raise ValueError("MIXED is reserved for the defined aggregation rule")
    records = read_jsonl(samples_path)
    required = {"sample_id", "image_path", "mask_path", "group_id", "split", "license", "source_url"}
    seen = set()
    for record in records:
        if set(record) != required or any(not isinstance(record[key], str) or not record[key].strip()
                                          for key in required):
            raise ValueError("Sample index must contain exactly nonempty " + ", ".join(sorted(required)))
        if record["sample_id"] in seen:
            raise ValueError("Duplicate source sample IDs")
        seen.add(record["sample_id"])
    if not records:
        raise ValueError("Sample index is empty")
    version = "semantic-v1-" + hashlib.sha256(config_bytes).hexdigest()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    requests, targets, provenance, exclusions = [], [], [], []
    for record in records:
        def resolve(value):
            path = Path(value)
            return path if path.is_absolute() else samples_path.parent / path
        source_image, source_mask = resolve(record["image_path"]), resolve(record["mask_path"])
        image_hash = hashlib.sha256(source_image.read_bytes()).hexdigest()
        mask_hash = hashlib.sha256(source_mask.read_bytes()).hexdigest()
        with Image.open(source_image) as opened:
            image = opened.convert("RGB")
        with Image.open(source_mask) as opened:
            if opened.mode not in {"L", "P", "I", "I;16"}:
                raise ValueError("Semantic mask must store scalar label IDs, not RGB colors")
            mask = opened.copy()
        if image.size != mask.size:
            raise ValueError(f"Image/mask dimensions differ: {record['sample_id']}")
        box = (int(roi[0] * image.width), int(roi[1] * image.height),
               int(roi[2] * image.width), int(roi[3] * image.height))
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("ROI rounds to an empty crop")
        cropped_mask = mask.crop(box)
        pixels = list(getattr(cropped_mask, "get_flattened_data", cropped_mask.getdata)())
        unknown = set(pixels) - set(mapping) - set(ignore)
        if unknown:
            raise ValueError(f"Unmapped mask labels {sorted(unknown)}: {record['sample_id']}")
        counts = Counter(mapping[label] for label in pixels if label in mapping)
        valid = sum(counts.values())
        if not valid or valid / len(pixels) < config["min_valid_fraction"]:
            exclusions.append({"sample_id": record["sample_id"], "reason": "insufficient_valid_annotation",
                               "valid_fraction": valid / len(pixels)})
            continue
        output = destination / "images" / (hashlib.sha256(record["sample_id"].encode()).hexdigest() + ".png")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            image.crop(box).save(stream, format="PNG")
        for task in tasks:
            if task == "terrain":
                if valid != len(pixels):
                    exclusions.append({"sample_id": record["sample_id"], "task_id": task,
                                       "reason": "terrain_requires_complete_roi_annotation"})
                    continue
                largest = max(counts.values())
                winners = [name for name, value in counts.items() if value == largest]
                answer = winners[0] if len(winners) == 1 and largest / valid >= config["dominant_min_fraction"] else "MIXED"
                allowed = terrain_answers
                prompt = "Which terrain class dominates this cropped region? Answer exactly one of: " + ", ".join(allowed) + "."
            else:
                rock_pixels = sum(counts[label] for label in rock_classes)
                if rock_pixels < config["rock_min_pixels"] and valid != len(pixels):
                    exclusions.append({"sample_id": record["sample_id"], "task_id": task,
                                       "reason": "negative_requires_complete_roi_annotation"})
                    continue
                answer = "YES" if rock_pixels >= config["rock_min_pixels"] else "NO"
                allowed = ("YES", "NO")
                prompt = (f"Does rock terrain cover at least {config['rock_min_pixels']} image pixels "
                          "in this cropped region? Answer exactly YES or NO.")
            key = hashlib.sha256(f"{record['sample_id']}:{task}:{version}:{image_hash}:{mask_hash}".encode()).hexdigest()
            requests.append(asdict(ModelRequest(key, (str(output),), prompt, allowed)))
            targets.append({"request_id": key, "answer": answer, "source_sample_id": record["sample_id"],
                            "group_id": record["group_id"], "derivation_version": version,
                            "task_id": task, "condition_id": "clean"})
        provenance.append({**record, "image_path": str(source_image.resolve()),
                           "mask_path": str(source_mask.resolve()), "prepared_image": str(output),
                           "roi_pixels": list(box), "valid_fraction": valid / len(pixels),
                           "image_sha256": image_hash, "mask_sha256": mask_hash,
                           "derivation_version": version})
    write_jsonl(destination / "requests.jsonl", requests)
    write_jsonl(destination / "targets.jsonl", targets)
    write_jsonl(destination / "provenance.jsonl", provenance)
    write_jsonl(destination / "exclusions.jsonl", exclusions)
    write_json(destination / "conversion.json", {"schema_version": "1.0", "specification": config,
                                                "derivation_version": version,
                                                "independence_audit": "pending",
                                                "physical_traversability_ground_truth": False})
    return destination
