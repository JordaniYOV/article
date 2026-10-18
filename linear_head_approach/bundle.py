"""Build the portable training kit from existing selected files, never download data."""
from __future__ import annotations

import argparse
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from .common import ROOT, atomic_json, rows, sha, write_rows
from .data import DATA, audit


def migrate(archive, root=ROOT):
    """Normalize the previous prepared cohort into immutable orbital-only paths."""
    root, archive = Path(root).resolve(), Path(archive).resolve()
    old = archive / "outputs/encoder_probe_wac_v1"
    destination = root / DATA
    if (destination / "inputs.jsonl").exists() and (destination / "targets.jsonl").exists():
        return audit(root)
    source_inputs, source_targets = rows(old / "encoder_inputs.jsonl"), rows(old / "targets.jsonl")
    target_map = {r["sample_id"]: r for r in source_targets}
    inputs, targets = [], []

    def copy_checked(source, relative, expected):
        if not source.is_file() or sha(source) != expected:
            raise ValueError(f"Old asset does not match recorded SHA-256: {source}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != expected:
            raise ValueError("Copy verification failed")

    for row in source_inputs:
        sid = row["sample_id"]
        folder = f"{DATA}/samples/{sid}"
        native = archive / "encoder_probe_wac_v1/native_vis" / Path(row["native_vis_5band_path"]).name
        mask_row = target_map[sid]
        mask = root / mask_row["mask"]
        if not mask.is_file():
            mask = archive / "encoder_probe_wac_v1/derived_masks" / Path(mask_row["mask"]).name
        for source, name, digest in ((root / row["rgb_image"], "rgb.png", row["rgb_image_sha256"]),
                                     (native, "native.tif", row["native_vis_5band_source_sha256"]),
                                     (mask, "mask.png", mask_row["mask_sha256"])):
            copy_checked(source, f"{folder}/{name}", digest)
        inputs.append({k: row[k] for k in ("sample_id", "split", "scene_group_id", "source_dataset", "source_revision", "source_image_id")}
                     | {"rgb_image": f"{folder}/rgb.png", "rgb_image_sha256": row["rgb_image_sha256"],
                        "native_vis": f"{folder}/native.tif", "native_vis_sha256": row["native_vis_5band_source_sha256"],
                        "display_transform": row["rgb_display_transform"], "license_claim": "source dataset card CC-BY-4.0; retain attribution"})
        targets.append({k: mask_row[k] for k in ("sample_id", "split", "scene_group_id", "target", "target_source_track")}
                      | {"mask": f"{folder}/mask.png", "mask_sha256": mask_row["mask_sha256"]})
    write_rows(destination / "inputs.jsonl", inputs)
    write_rows(destination / "targets.jsonl", targets)
    atomic_json(destination / "provenance.json", {
        "previous_inputs_sha256": sha(old / "encoder_inputs.jsonl"),
        "previous_targets_sha256": sha(old / "targets.jsonl"),
        "previous_audit": json.loads((old / "audit.json").read_text()),
        "selection": "Previously curated 120 WAC tiles, unchanged source splits and publisher masks",
        "source_url": "https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-WAC-Crater-Detection",
        "negative_definition": "outside catalog polygons, NOT guaranteed absence of all visible craters",
        "conversion": "byte-identical RGB/native/masks; only paths normalized; no heights used",
    })
    return audit(root)


def build(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Never overwrite a published bundle: {output}")
    audit(root)
    paths = list((root / DATA).rglob("*"))
    paths += list((root / "linear_head_approach").rglob("*"))
    paths += [root / "notebooks/encoder_heads_colab.ipynb", root / "requirements/encoder-heads-colab.txt",
              root / "requirements/ibm-val14-colab-lock.txt"]
    selected = sorted({p for p in paths if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"})
    # Explicit allowlist of IBM code only. Include license and original commit evidence.
    runtime = root / "external/ibm-lfm"
    for folder in ("ni_lfm", "terratorch_integration"):
        selected += sorted(p for p in (runtime / folder).rglob("*.py") if "__pycache__" not in p.parts)
    selected += [runtime / "LICENSE", runtime / "NOTICE"]
    from .common import git_revision
    revision = git_revision(runtime)
    if revision != "d54c67aad513cb9daca444afa425cfb278e4fbf8":
        raise ValueError("IBM source revision differs from protocol")
    import subprocess
    if subprocess.check_output(["git", "-C", str(runtime), "status", "--porcelain", "--", "ni_lfm", "terratorch_integration"], text=True).strip():
        raise ValueError("IBM bundled source has uncommitted edits")
    manifest = {"schema": "head-bundle-v2", "ibm_runtime_revision": revision,
                "weights_included": False, "training_executed": False,
                "files": {p.relative_to(root).as_posix(): sha(p) for p in selected}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        archive.writestr("bundle.json", json.dumps(manifest, indent=2))
        for path in selected:
            archive.write(path, path.relative_to(root).as_posix())
    result = {"zip": str(output), "bytes": output.stat().st_size, "sha256": sha(output),
              "uncompressed_bytes": sum(p.stat().st_size for p in selected), "files": len(selected),
              "training_executed": False, "gpu_smoke_verified": False}
    atomic_json(output.with_suffix(".json"), result)
    return result


def unpack(bundle, destination, expected_sha):
    """Verified no-traversal extraction; identical workspace reuse only."""
    if sha(bundle) != expected_sha:
        raise ValueError("Bundle SHA-256 mismatch")
    destination = Path(destination).resolve()
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename or ":" in info.filename or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Unsafe ZIP member")
        manifest = json.loads(archive.read("bundle.json"))
        if set(names) != {"bundle.json", *manifest["files"]}:
            raise ValueError("ZIP allowlist differs from manifest")
        import hashlib
        for name, digest in manifest["files"].items():
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"ZIP integrity error: {name}")
            path = destination / name
            if any(p.is_symlink() for p in [path, *path.parents]):
                raise ValueError("Extraction destination contains a symlink")
            if path.exists() and sha(path) != digest:
                raise ValueError(f"Existing workspace differs: {path}")
        for name in names:
            target = destination / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--migrate-archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.migrate_archive:
        print(json.dumps(migrate(args.migrate_archive), indent=2))
    if args.output:
        print(json.dumps(build(args.output), indent=2))
