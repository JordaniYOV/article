"""Pure NumPy candidate filtering; cannot see crater annotations or reference masks."""
import math
import numpy as np


def union_candidates(candidates, shape, cfg):
    combined = np.zeros(shape, dtype=bool)
    accepted = 0
    for item in candidates:
        mask = np.asarray(item["segmentation"], dtype=bool)
        if mask.shape != shape:
            raise ValueError("SAM candidate is not on the original pixel grid")
        area = int(mask.sum())
        if area < cfg["min_area"] or area > np.prod(shape) * cfg["max_area_fraction"]:
            continue
        y, x = np.nonzero(mask)
        height, width = int(y.max() - y.min() + 1), int(x.max() - x.min() + 1)
        if min(height, width) / max(height, width) < cfg["min_axis_ratio"]:
            continue
        # Four-neighbour perimeter, including borders. This is a grid heuristic, not a rim classifier.
        padded = np.pad(mask.astype(np.int8), 1)
        perimeter = int(np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum())
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter else 0.
        if circularity < cfg["min_circularity"]:
            continue
        combined |= mask
        accepted += 1
    return combined, accepted
