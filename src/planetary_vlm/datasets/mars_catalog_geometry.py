"""Catalog-ID-preserving reconstruction of Mars-Bench ellipse rasterization.

This is a reverse-engineered raster recipe, NOT a publisher-documented physical
map projection. The recipe was selected on the 14 selected source-val frames,
then frozen before source-test verification. Every admitted frame must reproduce
its immutable source class mask exactly. A matching raster establishes alignment,
not visual-rim correctness, completeness of the catalog, or accepted reference GT.
No semantic connected components, image fitting, or per-frame parameter search.
"""
import math

VERSION = "mars-catalog-raster-reconstruction-v1"
SEMIAXIS_PIXELS_PER_CATALOG_KM = 2.161
RECIPE = {
    "version": VERSION,
    "status": "reverse_engineered_not_publisher_documented",
    "selected_on": "selected source-val binary crater frames mars_seg_017..030 only",
    "native_size": [512, 512],
    "center_x_field": "LONGITUDE_IMAGE_PIXEL_Y",
    "center_y_field": "LATITUDE_IMAGE_PIXEL_X",
    "semiaxis_pixels_per_catalog_km": SEMIAXIS_PIXELS_PER_CATALOG_KM,
    "axes": "int(major_diameter_km * scale), int(minor_diameter_km * scale)",
    "opencv_angle_degrees": "90 - DIAM_ELLIPSE_ANGLE_IMAGE",
    "renderer": "cv2.ellipse; start=0; end=360; thickness=-1; lineType=LINE_8; shift=0",
    "overdraw": "CSV record order; binary=1; multi=TYPE",
    "bbox": "individual full raster support [min_x,min_y,max_x+1,max_y+1], not a connected component",
    "admission": "exact class raster replay per frame; visual/identity second review remains mandatory",
    "physical_map_scale_verified": False,
}
CLASS_NAMES = {1: "Other", 2: "Layered", 3: "Buried", 4: "Secondary"}


def ellipse_parameters(record):
    """Frozen recipe only; reject malformed records rather than normalize them."""
    cx, cy = (int(record[k]) for k in ("LONGITUDE_IMAGE_PIXEL_Y", "LATITUDE_IMAGE_PIXEL_X"))
    major, minor, angle = (float(record[k]) for k in (
        "DIAM_ELLIPSE_MAJOR_IMAGE", "DIAM_ELLIPSE_MINOR_IMAGE", "DIAM_ELLIPSE_ANGLE_IMAGE"))
    if not all(math.isfinite(v) for v in (major, minor, angle)) or not 0 < minor <= major:
        raise ValueError("Invalid catalog ellipse axes/angle")
    # Publisher CSVs include centers exactly on the exclusive bottom boundary;
    # retain their clipped ellipses rather than silently dropping catalog IDs.
    if not (0 <= cx <= 512 and 0 <= cy <= 512):
        raise ValueError("Catalog center outside native tile boundaries")
    axes = tuple(int(v * SEMIAXIS_PIXELS_PER_CATALOG_KM) for v in (major, minor))
    if not (0 < min(axes) <= max(axes) <= 4096):
        raise ValueError("Unsupported or degenerate catalog ellipse")
    kind = int(record["TYPE"])
    if kind not in CLASS_NAMES or record["STATUS"] != CLASS_NAMES[kind]:
        raise ValueError("Catalog TYPE/STATUS mismatch")
    return (cx, cy), axes, 90 - angle, kind


def replay_catalog(records, size, source_mask, *, multiclass):
    """Return geometry only if the frozen renderer exactly matches the source.

    All catalog IDs remain separate, including overlapped/fully overpainted IDs.
    Hidden instances are flagged for visual review, never removed automatically.
    Dependency imports are lazy: the rest of the benchmark does not need OpenCV.
    """
    import cv2
    import numpy as np
    from PIL import Image

    if tuple(size) != (512, 512):
        raise ValueError("Mars recipe is verified for native 512x512 only")
    ids = [r["CRATER_ID"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate catalog crater ID")
    raster = np.zeros((512, 512), dtype=np.uint8)
    owners = np.zeros((512, 512), dtype=np.int32)
    geometries = []
    for index, record in enumerate(records, 1):
        center, axes, angle, kind = ellipse_parameters(record)
        cv2.ellipse(raster, center, axes, angle, 0, 360, kind if multiclass else 1, -1, cv2.LINE_8)
        cv2.ellipse(owners, center, axes, angle, 0, 360, index, -1, cv2.LINE_8)
        # Render individually on an unclipped local canvas; never use a merged
        # semantic component or the post-overdraw visible fragment as its bbox.
        pad = max(axes) + 3
        individual = np.zeros((2*pad+1, 2*pad+1), dtype=np.uint8)
        cv2.ellipse(individual, (pad, pad), axes, angle, 0, 360, 1, -1, cv2.LINE_8)
        ys, xs = np.nonzero(individual)
        x, y = center
        box = [int(xs.min())-pad+x, int(ys.min())-pad+y,
               int(xs.max())-pad+x+1, int(ys.max())-pad+y+1]
        geometries.append({"source_annotation_id": record["CRATER_ID"], "bbox_xyxy": box,
            "ellipse_center_xy": list(center), "ellipse_semiaxes_px": list(axes),
            "ellipse_angle_opencv_degrees": angle, "subtype": CLASS_NAMES[kind],
            "truncated": box[0] < 0 or box[1] < 0 or box[2] > size[0] or box[3] > size[1],
            "uncertain": False, "catalog_record": dict(record),
            "geometry_verified_against_image": False, "geometry_recipe_version": VERSION})
    with Image.open(source_mask) as mask:
        if mask.size != tuple(size) or mask.mode != "L":
            raise ValueError("Expected native one-channel class-ID source mask")
        actual = np.asarray(mask)
    different = int(np.count_nonzero(raster != actual))
    visible = np.bincount(owners.ravel(), minlength=len(records)+1)
    for index, geometry in enumerate(geometries, 1):
        geometry["visible_source_class_raster_pixels"] = int(visible[index])
        geometry["fully_overpainted_in_source_mask"] = not bool(visible[index])
    audit = {"recipe": dict(RECIPE), "runtime": {"opencv": cv2.__version__, "numpy": np.__version__},
        "catalog_instances": len(records), "differing_class_pixels": different,
        "source_mask_reproduced_exactly": different == 0,
        "catalog_geometry_alignment_verified": different == 0,
        "fully_overpainted_catalog_instances": sum(g["fully_overpainted_in_source_mask"] for g in geometries),
        "visual_rim_review_complete": False}
    # Fail closed: even a single mismatched class pixel leaves all IDs pending.
    return (geometries if different == 0 else []), audit
