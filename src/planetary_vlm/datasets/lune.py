"""Lunar sources: provided/stereo depth stays distinct from semantic masks."""
from collections import Counter
import json
from pathlib import PurePosixPath
from .sources import read_json, save_json, download
from .base import BaseData


def lunar_candidates(files, revision):
    by_path = {item["path"]: item for item in files if item["type"] == "file"}
    rows = []
    for name in sorted(by_path):
        if not name.startswith("images/") or not name.endswith(".png"):
            continue
        sample_id = PurePosixPath(name).stem
        depth = f"depth/{sample_id}.npy"
        valid = f"masks/{sample_id}.npy"
        rows.append({
            "sample_id": f"lumon_change3_{sample_id}", "source_revision": revision,
            "source_image": name, "source_depth": depth if depth in by_path else None,
            "source_depth_validity_mask": valid if valid in by_path else None,
            "source_semantic_mask": None, "source_instance_mask": None,
            "depth_origin": "publisher_stereo_reconstruction",
            "license_claim": "mit_dataset_card", "original_split": "train",
            "scene_group_id": None, "original_mission_image_id": None,
            "benchmark_eligible": False,
            "blockers": ["semantic_segmentation_missing", "scene_provenance_pending"],
        })
    return rows


class LuneData(BaseData):
    domain = "moon"

    def download(self, *, pilot_ids=("001", "084", "168")):
        if any(not (sid.isascii() and sid.isdigit() and len(sid) == 3) for sid in pilot_ids):
            raise ValueError("Pilot IDs must be three ASCII digits")
        from .sources import collect
        return collect(self.root, list(pilot_ids), False, domains=("moon",))

    @staticmethod
    def candidate_inventory(files, revision):
        return lunar_candidates(files, revision)

    def inspect(self):
        from .inspection import inspect
        return inspect(self.root, domains=("moon",))

    def prepare(self, *, output, specification=None, samples=None, depth_index=None,
                track_metadata=None):
        if samples is None or specification is None:
            raise ValueError("Moon preparation requires a verified semantic samples index and native label specification; depth-validity masks are not semantic GT")
        from .preparation import build
        return build(samples, specification, self.new_version(output), domain=self.domain,
                     depth_index=depth_index, track_metadata=track_metadata)

    def _collect_sources(self, pilot_ids, report):
        root, evidence = self.root, self.evidence
        try:
            repository = "LuMonDepth/ChangE-3"
            metadata = read_json(f"https://huggingface.co/api/datasets/{repository}")
            revision = metadata["sha"]
            files = read_json(f"https://huggingface.co/api/datasets/{repository}/tree/{revision}?recursive=true&expand=false")
            save_json(evidence / "lumon_metadata.json", metadata)
            save_json(evidence / "lumon_file_inventory.json", files)
            rows = lunar_candidates(files, revision)
            (evidence / "lunar_depth_candidates.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            by_path = {item["path"]: item for item in files if item["type"] == "file"}
            assets = []
            for name in ["README.md"] + [f"{folder}/{sid}.{extension}"
                    for sid in pilot_ids for folder, extension in
                    (("images", "png"), ("depth", "npy"), ("masks", "npy"))]:
                item = by_path[name]
                checksum = "sha256:" + item["lfs"]["oid"] if "lfs" in item else None
                assets.append(download(f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{name}",
                                       root / "raw" / "lumon_change3" / name,
                                       item["size"], checksum))
            report["sources"]["lumon_change3"] = {
                "revision": revision, "image_count": len(rows),
                "matched_image_depth_validity_count": sum(bool(r["source_depth"] and
                    r["source_depth_validity_mask"]) for r in rows),
                "semantic_masks": 0, "pilot_assets": assets,
                "note": "masks are depth validity, not semantic segmentation; train is the publisher split"}
        except Exception as error:
            report["errors"].append({"source": "lumon_change3", "error": str(error)})

        try:
            repository = "lothanspace/change4-tcm-dataset"
            tree = read_json(f"https://api.github.com/repos/{repository}/git/trees/main?recursive=1")
            if tree.get("truncated"):
                raise ValueError("Incomplete GitHub tree")
            revision = tree["sha"]
            save_json(evidence / "change4_file_inventory.json", tree)
            base = f"https://raw.githubusercontent.com/{repository}/{revision}"
            annotation_path = root / "raw" / "change4_tcam_annotations" / "train.jsonl"
            entry = next(item for item in tree["tree"] if item["path"] == "data/masks/train.jsonl")
            assets = [download(base + "/data/masks/train.jsonl", annotation_path, entry["size"])]
            assets.append(download(base + "/README.md", annotation_path.parent / "README.md"))
            annotations = [json.loads(line) for line in annotation_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            labels = Counter(shape["label"] for row in annotations for shape in row["shapes"])
            report["sources"]["change4_tcam"] = {
                "revision": revision, "annotation_count": len(annotations),
                "label_instance_counts": dict(labels), "assets": assets,
                "embedded_images": sum(bool(row.get("imageData")) for row in annotations),
                "provided_depth_maps": 0, "mission_image_access": "CLPDS_account_required",
                "camera_provenance": "TCAM_is_lander_camera_per_NSSDC",
                "camera_source_url": "https://www.nssdc.ac.cn/mobile/nssdc_en/html/task/change4.html",
                "compatible_with_lumon_change3_by_ids": False}
        except Exception as error:
            report["errors"].append({"source": "change4_tcam", "error": str(error)})
