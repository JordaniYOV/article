"""Smooth seeded scan-line displacement of an image and one aligned raster."""

import cv2
import numpy as np

from ._common import integer, number, seed_sequence, validate_image, validate_maps


def _smooth_offsets(length, rng, max_shift_px, control_spacing_px):
    """Cubic Hermite interpolation of sparse random offsets, with bounded slope."""
    if length < 2 or max_shift_px == 0:
        return np.zeros(length, dtype=np.float32)
    knots = np.append(np.arange(0, length - 1, control_spacing_px, dtype=float), length - 1)
    if len(knots) < 3 and length > 2:
        knots = np.linspace(0, length - 1, 3)
    controls = rng.normal(size=len(knots))
    slopes = np.zeros_like(controls)
    slopes[1:-1] = (controls[2:] - controls[:-2]) / (knots[2:] - knots[:-2])
    samples = np.arange(length)
    segment = np.clip(np.searchsorted(knots, samples, side="right") - 1, 0, len(knots) - 2)
    span = knots[segment + 1] - knots[segment]
    t = (samples - knots[segment]) / span
    # Endpoint values and derivatives agree between adjacent curve segments.
    offsets = ((2*t**3 - 3*t**2 + 1) * controls[segment]
               + (t**3 - 2*t**2 + t) * span * slopes[segment]
               + (-2*t**3 + 3*t**2) * controls[segment + 1]
               + (t**3 - t**2) * span * slopes[segment + 1])
    offsets -= offsets.mean()
    peak = np.max(np.abs(offsets))
    if peak > 0:
        offsets *= max_shift_px / peak
    # Keep adjacent lines ordered and prevent folds in the two-axis field.
    gradient = np.max(np.abs(np.diff(offsets)))
    if gradient > 0.35:
        offsets *= 0.35 / gradient
    return offsets.astype(np.float32)


def _jitter_maps(h, w, max_shift_px, control_spacing_px, seed, scan_axis):
    x_seed, y_seed = seed_sequence(seed).spawn(2)
    x_length = w if scan_axis == "columns" else h
    y_length = h if scan_axis == "rows" else w
    dx = _smooth_offsets(x_length, np.random.default_rng(x_seed), max_shift_px, control_spacing_px)
    dy = _smooth_offsets(y_length, np.random.default_rng(y_seed), max_shift_px, control_spacing_px)
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # remap maps output coordinates back to the source: subtract the offsets.
    map_x -= dx[None, :] if scan_axis == "columns" else dx[:, None]
    map_y -= dy[:, None] if scan_axis == "rows" else dy[None, :]
    return map_x, map_y


def apply_synchronous_shear(
    image: np.ndarray,
    max_shift_px: float = 3.0,
    dem_mask: np.ndarray | None = None,
    seg_mask: np.ndarray | None = None,
    *,
    control_spacing_px: int = 32,
    seed: int | None = 0,
    scan_axis: str = "both",
    seg_ignore_label: int = 255,
    dem_nodata: float = np.nan,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply smooth seeded line offsets, using identical maps for all rasters.

    max_shift_px bounds each displacement component, not a global shear angle.
    control_spacing_px spaces random anchors; cubic interpolation avoids seams.
    scan_axis='both': horizontal offsets by row and vertical offsets by column.
    'rows' or 'columns': both components depend on that acquisition scan axis.
    A slope bound may reduce actual amplitude to prevent folded raster geometry.

    Supply exactly one map. Image borders are zero; segmentation borders use
    seg_ignore_label. DEM interpolation touching nodata or an exterior pixel
    remains nodata. All outputs are independent arrays, including at zero shift.
    """
    validate_image(image)
    max_shift_px = number(max_shift_px, "max_shift_px", maximum=32766)
    control_spacing_px = integer(control_spacing_px, "control_spacing_px", minimum=2)
    if scan_axis not in ("both", "rows", "columns"):
        raise ValueError("scan_axis must be 'both', 'rows' or 'columns'")
    seed_sequence(seed)
    validate_maps(image, dem_mask, seg_mask, seg_ignore_label, dem_nodata)
    h, w = image.shape[:2]
    if h >= 32767 or w >= 32767:
        raise ValueError("OpenCV remap requires image sides smaller than 32767 pixels")
    if max_shift_px == 0:
        return (image.copy(), dem_mask.copy() if dem_mask is not None else None,
                seg_mask.copy() if seg_mask is not None else None)

    map_x, map_y = _jitter_maps(h, w, max_shift_px, control_spacing_px, seed, scan_axis)
    sh_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if dem_mask is not None:
        valid = np.isfinite(dem_mask)
        if np.isfinite(dem_nodata):
            valid &= dem_mask != dem_nodata
        values = cv2.remap(np.where(valid, dem_mask, 0), map_x, map_y,
                            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        weights = cv2.remap(valid.astype(np.float32), map_x, map_y,
                             interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        sh_dem = np.full(dem_mask.shape, dem_nodata, dtype=dem_mask.dtype)
        # Require all contributors to be valid; do not invent terrain in gaps.
        complete = weights >= 1 - 1e-6
        sh_dem[complete] = values[complete] / weights[complete]
        return sh_image, sh_dem, None

    # Warp pixel indices instead of class values, preserving even int64 IDs.
    indices = np.arange(h * w, dtype=np.float64).reshape(h, w)
    warped = cv2.remap(indices, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    sh_seg = np.full(seg_mask.shape, seg_ignore_label, dtype=seg_mask.dtype)
    inside = warped >= 0
    sh_seg[inside] = seg_mask.ravel()[warped[inside].astype(np.intp)]
    return sh_image, None, sh_seg
