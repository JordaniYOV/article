"""Pinned annotation-only source recovery. Never downloads images or train data."""
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from planetary_vlm.datasets.orbital_questions import orbital_path, file_hash

SOURCES = (
    ("moon_debris_val.json", "F1nnSBK/lunar-debris-and-voids", "f449ccb792c435a36c2e7f7d51137122f3281bcb", "annotations/instances_val.json", 98880, None),
    ("moon_debris_test.json", "F1nnSBK/lunar-debris-and-voids", "f449ccb792c435a36c2e7f7d51137122f3281bcb", "annotations/instances_test.json", 68233, None),
    ("moon_wac_val.json", "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection", "20f800becb64a0c7ad314652a683f97f0281fe67", "val.json", 14255094, "d70ba255fff335818609bf8201331f4044f3e8d6070b6cbb8b1b9c27b64d0190"),
    ("moon_wac_test.json", "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection", "20f800becb64a0c7ad314652a683f97f0281fe67", "test.json", 14159333, "caf72eac333d62f06041949c3b4e4db67672b273a96694e2ac329b096bc82dff"),
    ("mars_crater_metadata.zip", "Mirali33/mb-crater_multi_seg", "9a26ac9001532bc9f4fe37abfa3fce13e4ea1b74", "metadata.zip", 10857134, "fe92462495dd59fde79ca773031abc78b35640ab9ac9242ac65d9a959e2306e4"),
)


def fetch_annotations(root, cache, *, transport="urllib"):
    """Explicit network operation, strict allowlist, bounded size and no overwrite."""
    from planetary_vlm.datasets.sources import download
    root = Path(root).resolve()
    cache = Path(cache).resolve()
    cache = orbital_path(root, cache.relative_to(root).as_posix())
    if cache == root / "data_orbital" or cache.is_relative_to(root / "data_orbital/orbital_300_v3"):
        raise ValueError("Annotation cache must be separate from original data")
    cache.mkdir(parents=True, exist_ok=True)
    inventory = []
    if transport not in {"urllib", "curl"}:
        raise ValueError("Use urllib or the system curl TLS transport")
    for name, repo, revision, remote, size, checksum in SOURCES:
        if size > 16 * 1024**2 or "train" in remote:
            raise ValueError("Annotation-only recovery limit exceeded")
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}"
        path = orbital_path(root, (cache.relative_to(root) / name).as_posix())
        if transport == "curl" and not path.exists():
            with tempfile.NamedTemporaryFile(prefix=name+".", suffix=".partial", dir=cache, delete=False) as temporary:
                partial = Path(temporary.name)
            try:
                subprocess.run(["curl.exe" if __import__("os").name == "nt" else "curl",
                                "--fail", "--location", "--silent", "--show-error", "--connect-timeout", "15",
                                "--max-time", "60", "--retry", "2", "--retry-delay", "1",
                                "--max-filesize", str(size), "--output", str(partial), url], check=True, timeout=195)
                if partial.stat().st_size != size or (checksum and file_hash(partial) != checksum):
                    raise ValueError("Source download failed size/checksum validation")
                partial.rename(path)
            finally:
                partial.unlink(missing_ok=True)
        result = download(url, path, expected_size=size, checksum=f"sha256:{checksum}" if checksum else None)
        inventory.append({"file": name, "repo": repo, "revision": revision,
                          "remote_path": remote, "size": size, "sha256": result["sha256"], "url": url})
    record = {"schema": "crater-source-annotations-v1", "sources": inventory,
              "images_downloaded": False, "train_images_downloaded": False, "weights_downloaded": False,
              "mars_metadata_archive_contains_train_metadata": True,
              "mars_metadata_members_used": "selected source-val/test CSVs only; never train members"}
    provenance = cache / "provenance.json"
    if provenance.exists():
        if json.loads(provenance.read_text(encoding="utf-8")) != record:
            raise ValueError("Annotation cache provenance changed")
    else:
        with provenance.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
    return record


class SourceAnnotations:
    """Recover publisher objects by IDs. Raster comparison validates alignment.

    Mars native-pixel extents use an explicitly reverse-engineered, frozen recipe
    checked by exact class-mask replay on every frame, never semantic components.
    A mismatch or missing renderer keeps the catalog IDs geometry-pending.
    """
    def __init__(self, root, cache):
        self.root, self.cache = Path(root).resolve(), Path(cache).resolve()
        self.provenance = validate_cache(self.root, self.cache)
        self.coco = {}
        self.index = {}
        for name, *_ in SOURCES:
            if not name.endswith(".json"):
                continue
            source = json.loads((self.cache / name).read_text(encoding="utf-8"))
            grouped = defaultdict(list)
            for ann in source["annotations"]:
                grouped[ann["image_id"]].append(ann)
            self.coco[name] = source
            self.index[name] = grouped

    def recover(self, row, size, source_mask):
        """Return known instances, unresolved catalog candidates, and audit."""
        from PIL import Image, ImageDraw
        sid, repo, split = row["sample_id"], row["source_dataset"], row["source_split"]
        if split not in {"val", "test"}:
            raise ValueError("Only publisher val/test annotations are admitted")
        if repo in {"F1nnSBK/lunar-debris-and-voids", "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection"}:
            kind = "debris" if repo.startswith("F1nn") else "wac"
            name = f"moon_{kind}_{split}.json"
            expected = next(s for s in SOURCES if s[0] == name)
            if row["source_revision"] != expected[2]:
                raise ValueError("Wrong source revision for COCO recovery")
            source = self.coco[name]
            matched = [i for i in source["images"] if (str(i["id"]) == str(row["source_image_id"]) if kind == "wac"
                       else i["file_name"] == row["source_image_id"])]
            if len(matched) != 1 or (matched[0]["width"], matched[0]["height"]) != size:
                raise ValueError("COCO source ID/dimensions do not match selected image")
            info = matched[0]
            categories = {c["id"]: c["name"].casefold() for c in source["categories"]}
            class_ids = {c.casefold(): i for i, c in enumerate(row["classes"])}
            raster = Image.new("L", size, 0)
            draw = ImageDraw.Draw(raster)
            objects, annotation_ids = [], set()
            for ann in self.index[name][info["id"]]:
                if ann["id"] in annotation_ids:
                    raise ValueError("Duplicate publisher annotation ID")
                annotation_ids.add(ann["id"])
                label = categories[ann["category_id"]]
                polygons = ann["segmentation"]
                if label not in class_ids or not isinstance(polygons, list):
                    raise ValueError("Unsupported source categories or non-polygon COCO geometry")
                for polygon in polygons:
                    if len(polygon) < 6 or len(polygon) % 2:
                        raise ValueError("Malformed source polygon")
                    draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=class_ids[label])
                if label != "crater":
                    continue
                x, y, width, height = ann["bbox"]
                if ann.get("iscrowd", 0):
                    raise ValueError("Crowd polygon cannot define an individual crater")
                box = [x, y, x+width, y+height]
                objects.append({"instance_id": f"{sid}__source_coco_{ann['id']}", "origin": "publisher_coco",
                    "source_annotation_id": ann["id"], "source_annotation_file": name,
                    "source_image_id_in_coco": info["id"], "bbox_xyxy": box, "uncertain": False,
                    "truncated": x < 0 or y < 0 or x+width > size[0] or y+height > size[1],
                    "segmentation": polygons, "class": "crater", "geometry_verified_against_image": False})
            with Image.open(source_mask) as mask:
                if mask.size != size or mask.convert("L").tobytes() != raster.tobytes():
                    raise ValueError("Recovered source polygons do not reproduce immutable class mask: " + sid)
            return objects, [], {"source_instances_complete": True,
                "source_geometry_status": "publisher_objects_raster_alignment_verified_visual_rim_review_pending",
                "source_mask_reproduced_exactly": True, "source_annotation_file": name}
        if repo in {"Mirali33/mb-crater_binary_seg", "Mirali33/mb-crater_multi_seg"}:
            revisions = {"Mirali33/mb-crater_binary_seg": "127ebfb30f2ec68781534a3071093a234ec90257",
                         "Mirali33/mb-crater_multi_seg": "9a26ac9001532bc9f4fe37abfa3fce13e4ea1b74"}
            if row["source_revision"] != revisions[repo]:
                raise ValueError("Wrong source revision for Mars catalog recovery")
            member = f"metadata/{split}/{Path(row['source_image_id']).stem}.csv"
            with zipfile.ZipFile(self.cache / "mars_crater_metadata.zip") as bundle:
                info = bundle.getinfo(member)
                if info.file_size > 2 * 1024**2:
                    raise ValueError("Selected CSV exceeds metadata limit")
                raw = bundle.read(info)
            records = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
            ids = [r["CRATER_ID"] for r in records]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate catalog crater ID")
            candidates = [{"instance_id": f"{sid}__source_catalog_{r['CRATER_ID']}",
                "origin": "publisher_catalog_geometry_pending", "source_annotation_id": r["CRATER_ID"],
                "source_annotation_file": "mars_crater_metadata.zip", "source_zip_member": member,
                "class": "crater", "subtype": r["STATUS"], "catalog_record": r,
                "bbox_xyxy": None, "geometry_status": "native_pixel_extent_unverified_not_inferred_from_semantic_components"}
                for r in records]
            audit = {"source_instances_complete": False, "source_mask_reproduced_exactly": False,
                     "source_annotation_file": "mars_crater_metadata.zip", "source_zip_member": member,
                     "source_zip_member_sha256": __import__("hashlib").sha256(raw).hexdigest()}
            from planetary_vlm.datasets.mars_catalog_geometry import replay_catalog
            try:
                geometries, replay = replay_catalog(records, size, source_mask, multiclass=repo.endswith("multi_seg"))
            except ModuleNotFoundError as error:
                if error.name not in {"cv2", "numpy"}:
                    raise
                audit.update(source_geometry_status="catalog_geometry_pending_missing_CPU_raster_dependency",
                             missing_dependency=error.name)
                return [], candidates, audit
            audit["catalog_raster_audit"] = replay
            if not replay["source_mask_reproduced_exactly"]:
                audit["source_geometry_status"] = "catalog_geometry_pending_frozen_recipe_does_not_reproduce_source_mask"
                return [], candidates, audit
            objects = [{**g, "instance_id": f"{sid}__source_catalog_{g['source_annotation_id']}",
                "origin": "publisher_catalog_reconstructed_raster", "class": "crater",
                "source_annotation_file": "mars_crater_metadata.zip", "source_zip_member": member,
                "geometry_status": "raster_alignment_verified_visual_rim_and_identity_second_review_pending"}
                for g in geometries]
            audit.update(source_instances_complete=True, source_mask_reproduced_exactly=True,
                         source_geometry_status="catalog_IDs_reconstructed_exact_class_raster_visual_rim_second_review_pending")
            return objects, [], audit
        if row["mask_semantics"] not in {"imp", "cone", "boulder"}:
            raise ValueError("Unknown source semantics; do not assume absence of crater labels")
        return [], [], {"source_instances_complete": True, "source_mask_reproduced_exactly": None,
            "source_geometry_status": "source_has_no_crater_class_complete_crater_inventory_depends_on_full_CVAT_review"}


def validate_cache(root, cache):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    cache = orbital_path(root, cache.relative_to(root).as_posix())
    record = json.loads((cache / "provenance.json").read_text(encoding="utf-8"))
    expected = {item[0]: item for item in SOURCES}
    rows = record["sources"]
    if len(rows) != len(expected) or {row["file"] for row in rows} != set(expected):
        raise ValueError("Incomplete annotation cache")
    for row in rows:
        name, repo, revision, remote, size, checksum = expected[row["file"]]
        if (row["repo"], row["revision"], row["remote_path"], row["size"]) != (repo, revision, remote, size):
            raise ValueError("Wrong source annotation identity")
        path = orbital_path(root, (cache.relative_to(root) / name).as_posix())
        if path.stat().st_size != size or file_hash(path) != row["sha256"] or (checksum and row["sha256"] != checksum):
            raise ValueError("Source annotation hash mismatch")
    return record
