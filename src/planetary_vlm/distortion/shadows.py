"""Hard-edged polygon illumination overlays, independent of target annotations."""

from collections.abc import Sequence

import cv2
import numpy as np

from ._common import generator, integer, number, restore_image, unit_image, validate_image


def apply_hard_shadows(
    image: np.ndarray,
    strength: float = 0.6,
    seed: int | None = 0,
    *,
    vertices: Sequence[tuple[float, float]] | None = None,
    num_shadows: int = 1,
) -> np.ndarray:
    """Multiply intensity inside hard-edged polygons by (1 - strength).

    vertices, when supplied, is one polygon of normalized (x, y) points in
    [0, 1]. Otherwise num_shadows seeded edge-spanning quadrilaterals are used.
    Polygons overlap by union; overlapping shadows are not darkened twice.
    Geometry is synthetic; this function does not derive terrain shadows from DEM.
    """
    validate_image(image)
    strength = number(strength, "strength", maximum=1)
    num_shadows = integer(num_shadows, "num_shadows")
    rng = generator(seed)
    if vertices is not None:
        polygon = np.asarray(vertices, dtype=np.float64)
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or not np.isfinite(polygon).all() or (polygon < 0).any() or (polygon > 1).any()):
            raise ValueError("vertices must contain >= 3 normalized (x, y) points")
        if cv2.contourArea(polygon.astype(np.float32)) == 0:
            raise ValueError("vertices must form a polygon with positive area")
        if num_shadows != 1:
            raise ValueError("An explicit polygon requires num_shadows=1")
        polygons = [polygon]
    else:
        polygons = []
        for _ in range(num_shadows):
            top, bottom = rng.uniform(0.15, 0.6, 2)
            polygon = np.array([[0, 0], [top, 0], [bottom, 1], [0, 1]])
            for _ in range(int(rng.integers(4))):
                polygon = np.column_stack((1 - polygon[:, 1], polygon[:, 0]))
            polygons.append(polygon)
    if strength == 0 or num_shadows == 0:
        return image.copy()

    h, w = image.shape[:2]
    shadow = np.zeros((h, w), dtype=np.uint8)
    for polygon in polygons:
        pixels = np.rint(polygon * (w - 1, h - 1)).astype(np.int32)
        cv2.fillPoly(shadow, [pixels], color=1, lineType=cv2.LINE_8)
    multiplier = 1 - strength * shadow.astype(np.float64)
    if image.ndim == 3:
        multiplier = multiplier[..., None]
    return restore_image(unit_image(image) * multiplier, image)
