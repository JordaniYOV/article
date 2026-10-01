"""Shared array validation and intensity conversion for synthetic corruptions."""

import numpy as np


def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError("image must be grayscale (H, W) or RGB/BGR (H, W, 3)")
    if 0 in image.shape:
        raise ValueError("image must not be empty")
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError("image dtype must be uint8, uint16, float32 or float64")
    if not np.isfinite(image).all():
        raise ValueError("image must contain finite intensities")
    if image.dtype.kind == "f" and (image.min() < 0 or image.max() > 1):
        raise ValueError("floating image intensities must be in [0, 1]")


def number(value, name: str, *, minimum=0.0, maximum=None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real number")
    if not np.isrealobj(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def integer(value, name: str, *, minimum=0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def seed_sequence(seed: int | None) -> np.random.SeedSequence:
    if seed is not None:
        seed = integer(seed, "seed")
    return np.random.SeedSequence(seed)


def generator(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed_sequence(seed))


def unit_image(image: np.ndarray) -> np.ndarray:
    maximum = np.iinfo(image.dtype).max if image.dtype.kind == "u" else 1.0
    return image.astype(np.float64) / maximum


def restore_image(values: np.ndarray, original: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0, 1)
    if original.dtype.kind == "u":
        values = np.rint(values * np.iinfo(original.dtype).max)
    return values.astype(original.dtype)


def validate_maps(image, dem_mask, seg_mask, seg_ignore_label, dem_nodata) -> None:
    if (dem_mask is None) == (seg_mask is None):
        raise ValueError("Provide exactly one aligned map: dem_mask or seg_mask")
    selected = dem_mask if dem_mask is not None else seg_mask
    if not isinstance(selected, np.ndarray) or selected.shape != image.shape[:2]:
        raise ValueError("The map must be a 2D NumPy array aligned with image")
    if dem_mask is not None:
        if dem_mask.dtype not in (np.float32, np.float64):
            raise ValueError("DEM must use float32/float64 to preserve interpolated elevations")
        if (isinstance(dem_nodata, (bool, np.bool_))
                or not isinstance(dem_nodata, (int, float, np.number))
                or not np.isrealobj(dem_nodata)):
            raise ValueError("dem_nodata must be a real number or NaN")
        try:
            nodata = float(dem_nodata)
        except (TypeError, ValueError) as error:
            raise ValueError("dem_nodata must be a real number or NaN") from error
        if np.isinf(nodata) or (np.isfinite(nodata) and abs(nodata) > np.finfo(dem_mask.dtype).max):
            raise ValueError("dem_nodata must be NaN or representable in DEM dtype")
    else:
        if seg_mask.dtype.kind not in "iu":
            raise ValueError("seg_mask must contain integer class IDs")
        if isinstance(seg_ignore_label, (bool, np.bool_)) or not isinstance(seg_ignore_label, (int, np.integer)):
            raise ValueError("seg_ignore_label must be an integer")
        limits = np.iinfo(seg_mask.dtype)
        if not limits.min <= seg_ignore_label <= limits.max:
            raise ValueError("seg_ignore_label must fit the segmentation dtype")
