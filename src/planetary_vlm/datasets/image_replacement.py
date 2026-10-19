"""Versioned image-only replacement; annotations and height assets stay immutable.

Registration uses image features only, never semantic masks or VLM predictions.
No networking, weight downloads, source overwrites or implicit GT admission.
"""
import json
from pathlib import Path
import shutil

from planetary_vlm.datasets.orbital_questions import digest, file_hash, orbital_path
from planetary_vlm.io import read_jsonl, write_json, write_jsonl

VERSION = "registered-source-image-replacement-v1"


def local(root, value):
    path = Path(value)
    if path.is_absolute():
        path = path.resolve().relative_to(root)
    return orbital_path(root, path.as_posix())


def resample_to_reference(source, destination_to_source, size):
    """Use real source support at every output pixel; never invent border data."""
    import cv2
    import numpy as np
    matrix = np.asarray(destination_to_source, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite destination-to-source affine matrix")
    width, height = size
    corners = np.array([[0,0,1], [width-1,0,1], [0,height-1,1], [width-1,height-1,1]]) @ matrix.T
    if (corners[:,0].min() < 1 or corners[:,1].min() < 1
            or corners[:,0].max() > source.shape[1]-2 or corners[:,1].max() > source.shape[0]-2):
        raise ValueError("Insufficient real source padding; no artificial border filling permitted")
    return cv2.warpAffine(source, matrix, (width,height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def register_themis(original, native_crop, padded_crop):
    """Frozen, image-only feature alignment on this source-val replacement."""
    import cv2
    import numpy as np
    from PIL import Image
    with Image.open(original) as image:
        reference = np.asarray(image.convert("L")); size = image.size
    with Image.open(native_crop) as image:
        native_size = image.size
        display = np.asarray(image.convert("L").resize(size, Image.Resampling.BOX))
    with Image.open(padded_crop) as image:
        padded = np.asarray(image.convert("L"))
    if size != (512,512) or native_size != (1185,1186) or padded.shape != (1202,1193):
        raise ValueError("Unexpected source crop or native reference geometry")
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(nfeatures=3000)
    ka, da = sift.detectAndCompute(reference, None); kb, db = sift.detectAndCompute(display, None)
    matches = [m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance < .7*n.distance]
    a = np.float32([ka[m.queryIdx].pt for m in matches]); b = np.float32([kb[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3)
    if matrix is None or inliers is None or int(inliers.sum()) < 100:
        raise ValueError("Insufficient image-only alignment evidence")
    selected = inliers.ravel().astype(bool)
    residual = np.linalg.norm(np.c_[a,np.ones(len(a))] @ matrix.T - b,axis=1)[selected]
    if float(np.median(residual)) > 1 or float(np.quantile(residual,.95)) > 2:
        raise ValueError("Registration residual exceeds native annotation review scale")
    # cv2 inverse map goes directly from old native pixel centers to the padded
    # higher-resolution source. The eight-pixel top/left margin is real imagery.
    native_map = matrix.copy()
    for axis, scale in enumerate((native_size[0]/size[0], native_size[1]/size[1])):
        native_map[axis,:2] *= scale
        native_map[axis,2] = (matrix[axis,2]+.5)*scale-.5+8
    result = resample_to_reference(padded,native_map,size)
    return result, {"method":"SIFT_ratio_0.7_affine_RANSAC_3px_seed_0",
        "image_features_only":True,"good_matches":len(matches),"inliers":int(inliers.sum()),
        "median_residual_px":float(np.median(residual)),"p95_residual_px":float(np.quantile(residual,.95)),
        "reference_to_source_display":matrix.tolist(),"reference_to_padded_native":native_map.tolist(),
        "source_size":list(reversed(padded.shape)),"output_size":list(size),
        "interpolation":"cv2.INTER_LINEAR inverse-map from real padded source",
        "artificial_border_fill_used":False,"opencv":cv2.__version__,"numpy":np.__version__}


def verify_replacement(root, row):
    """Validate the exact image/mask anchor and replay its resampling recipe."""
    import numpy as np
    from PIL import Image
    from planetary_vlm.datasets.retired_assets import retained_orbital_path
    root = Path(root).resolve()
    path = local(root,row["image_replacement_record"])
    proof = json.loads(path.read_text(encoding="utf-8"))
    checksum = proof.pop("record_sha256")
    if proof.get("schema") != VERSION or digest(proof) != checksum or proof["sample_id"] != row["sample_id"]:
        raise ValueError("Image replacement provenance changed")
    for field in ("original_image", "native_source_crop", "padded_source_crop", "replacement_image"):
        item = proof[field]
        try:
            retained = retained_orbital_path(root,item["path"],item["sha256"])
        except (FileNotFoundError, ValueError) as error:
            raise ValueError("Replacement input/output changed: " + field) from error
        if file_hash(retained) != item["sha256"]:
            raise ValueError("Replacement input/output changed: " + field)
    if (proof["replacement_image"]["path"] != row["image"]
            or proof["replacement_image"]["sha256"] != row["image_sha256"]
            or proof["mask_sha256"] != row["mask_sha256"]
            or file_hash(local(root,row["mask"])) != row["mask_sha256"]):
        raise ValueError("Replacement no longer matches image/unchanged mask anchor")
    with Image.open(local(root,proof["padded_source_crop"]["path"])) as image:
        source = np.asarray(image.convert("L"))
    expected = resample_to_reference(source,proof["registration"]["reference_to_padded_native"],
                                     tuple(proof["registration"]["output_size"]))
    with Image.open(local(root,row["image"])) as image:
        if image.size != tuple(proof["registration"]["output_size"]) or not np.array_equal(expected,np.asarray(image.convert("L"))):
            raise ValueError("Replacement pixels do not reproduce registered source")
    return proof


def replace_mars_image(root, source_collection="data_orbital/orbital_300_v3", output_collection="data_orbital/orbital_300_v4"):
    """Create a new active collection; exactly one data asset has new pixels."""
    from PIL import Image
    root = Path(root).resolve(); source = local(root,source_collection); output = local(root,output_collection)
    if output == root/"data_orbital" or output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Use a new, separate versioned collection")
    native = local(root,"data_orbital/source_comparison_v1/themis_v11_geographic_crop.png")
    padded = local(root,"data_orbital/source_comparison_v1/themis_v11_geographic_crop_padded.png")
    manifests = {}
    for planet in ("moon","mars"):
        for track in ("segmentation","height","train"):
            path = source/planet/(track+"_manifest.jsonl")
            manifests[path.relative_to(source).as_posix()] = read_jsonl(path)
    selected = [r for rs in manifests.values() for r in rs if r["sample_id"] == "mars_seg_018"]
    if len(selected) != 1: raise ValueError("Expected exactly one replacement anchor")
    anchor = selected[0]; original = local(root,anchor["image"])
    if file_hash(original) != anchor["image_sha256"] or file_hash(local(root,anchor["mask"])) != anchor["mask_sha256"]:
        raise ValueError("Original image/mask anchor changed")
    rendered, registration = register_themis(original,native,padded)
    paths = sorted(p for p in source.rglob("*") if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(source) for p in paths):
        raise ValueError("Symlink or escaped source asset")
    before = {p.relative_to(source).as_posix():file_hash(p) for p in paths}
    source_prefix = source.relative_to(root).as_posix(); output_prefix = output.relative_to(root).as_posix()
    def rebase(value):
        if isinstance(value,str) and value.startswith(source_prefix+"/"): return output_prefix+value[len(source_prefix):]
        if isinstance(value,list): return [rebase(v) for v in value]
        if isinstance(value,dict): return {k:rebase(v) for k,v in value.items()}
        return value
    image_relative = original.relative_to(source)
    output.mkdir(parents=True,exist_ok=False)
    for path in paths:
        name = path.relative_to(source)
        if name.as_posix() in manifests or name.as_posix() == "audit.json" or name == image_relative: continue
        destination = output/name; destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,destination)
    new_image = output/image_relative; new_image.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rendered).save(new_image)
    def identity(path): return {"path":path.relative_to(root).as_posix(),"sha256":file_hash(path)}
    proof = {"schema":VERSION,"sample_id":anchor["sample_id"],"source_collection":source_prefix,
        "output_collection":output_prefix,"original_image":identity(original),"native_source_crop":identity(native),
        "padded_source_crop":identity(padded),"replacement_image":identity(new_image),"registration":registration,
        "mask_sha256":anchor["mask_sha256"],"source_split":anchor["source_split"],"scene_group_id":anchor["scene_group_id"],
        "image_source":{"product":"ASU THEMIS Day IR 100m global mosaic v11","url":"https://www.mars.asu.edu/data/thm_dir_100m/large/lat-30_lon300.pgm",
            "native_crop_xyxy":[34380,13040,35565,14226],"padded_crop_xyxy":[34372,13032,35565,14234],
            "image_license_status":"Public source archive; separate image-license verification pending. Original mask license claim retained."},
        "authorization":"Human explicitly requested replacing this image only, with mask and height unchanged.",
        "reference_admission":"not_automatically_accepted","converter_sha256":file_hash(Path(__file__))}
    proof["record_sha256"] = digest(proof); write_json(output/"image_replacement.json",proof)
    for name, rows in manifests.items():
        updated = rebase(rows)
        for row in updated:
            if row["sample_id"] == anchor["sample_id"]:
                row.update(image_sha256=file_hash(new_image),image_replacement_record=(output/"image_replacement.json").relative_to(root).as_posix(),
                           image_source=proof["image_source"],annotation_anchor_image=anchor["image"])
        write_jsonl(output/name,updated)
    # Validate all copied data, original files, and the one intentional image change.
    data_names = [n for n in before if n not in manifests and n != "audit.json"]
    changed = [n for n in data_names if file_hash(output/n) != before[n]]
    if changed != [image_relative.as_posix()] or any(file_hash(source/n) != h for n,h in before.items()):
        raise ValueError("Unexpected source or annotation/height modification")
    audit = json.loads((source/"audit.json").read_text(encoding="utf-8"))
    audit.update(selection_version=output.name.replace("_","-"),parent_collection=source_prefix,
                 changed_data_assets=changed,unchanged_data_assets=len(data_names)-1,
                 masks_and_heights_unchanged=True,image_replacement_record_sha256=proof["record_sha256"])
    write_json(output/"audit.json",audit)
    return {"collection":output_prefix,"replacement_image":proof["replacement_image"],"registration":registration,
            "changed_data_assets":changed,"unchanged_data_assets":len(data_names)-1,"masks_and_heights_unchanged":True}


if __name__ == "__main__":
    print(json.dumps(replace_mars_image(Path(__file__).resolve().parents[3]),indent=2))
