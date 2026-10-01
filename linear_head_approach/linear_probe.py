"""Small frozen-feature spatial ridge probe; NumPy/Pillow load only on use.

Feature extraction is intentionally separate. Only train arrays may enter fit().
"""
from __future__ import annotations

from dataclasses import dataclass


def resize_features(feature_map, size: tuple[int, int]):
    """Bilinearly resize H×W×C tokens to a fixed spatial grid."""
    import numpy as np

    array = np.asarray(feature_map, dtype=np.float64)
    if array.ndim != 3 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError("Expected a finite H×W×C feature map")
    height, width, _ = array.shape
    out_h, out_w = size
    if out_h < 1 or out_w < 1:
        raise ValueError("Grid dimensions must be positive")
    ys = np.clip((np.arange(out_h) + 0.5) * height / out_h - 0.5, 0, height - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * width / out_w - 0.5, 0, width - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, height - 1), np.minimum(x0 + 1, width - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    return ((1 - wy) * (1 - wx) * array[y0[:, None], x0[None, :]]
            + (1 - wy) * wx * array[y0[:, None], x1[None, :]]
            + wy * (1 - wx) * array[y1[:, None], x0[None, :]]
            + wy * wx * array[y1[:, None], x1[None, :]])


def mask_fractions(mask, size: tuple[int, int]):
    """Area-average a binary crater mask onto the probe grid."""
    import numpy as np
    from PIL import Image

    array = np.asarray(mask)
    if array.ndim != 2 or not np.isin(array, [0, 1]).all():
        raise ValueError("Expected a binary 0/1 crater mask")
    image = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(image.resize((size[1], size[0]), Image.Resampling.BOX), dtype=np.float64)


@dataclass(frozen=True)
class SpatialRidgeProbe:
    coefficients: object
    intercept: float
    mean: object
    scale: object
    grid: tuple[int, int]
    regularization: float

    def predict_grid(self, feature_map):
        import numpy as np

        features = resize_features(feature_map, self.grid)
        if features.shape[-1] != len(self.coefficients):
            raise ValueError("Feature width changed after fitting")
        score = ((features - self.mean) / self.scale) @ self.coefficients + self.intercept
        return np.clip(score, 0, 1)


def fit(feature_maps, masks, *, grid: tuple[int, int] = (24, 24), regularization: float = 1.0):
    """Fit one linear head to frozen train features and binary masks only."""
    import numpy as np

    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError("Regularization must be positive")
    if len(feature_maps) != len(masks) or not feature_maps:
        raise ValueError("Fit requires matching non-empty feature and mask lists")
    x_rows, y_rows = [], []
    width = None
    for features, mask in zip(feature_maps, masks):
        spatial = resize_features(features, grid)
        if width is None:
            width = spatial.shape[-1]
        elif spatial.shape[-1] != width:
            raise ValueError("Inconsistent feature width")
        x_rows.append(spatial.reshape(-1, width))
        y_rows.append(mask_fractions(mask, grid).reshape(-1))
    x = np.concatenate(x_rows)
    y = np.concatenate(y_rows)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x = (x - mean) / scale
    # Each source image contributes the same number of grid cells. Weighting
    # prevents tiny positive crater regions from vanishing in the background.
    prevalence = max(float(y.mean()), 1e-4)
    positive_weight = min(10.0, max(1.0, (1.0 - prevalence) / prevalence))
    weights = 1.0 + (positive_weight - 1.0) * y
    total = weights.sum()
    x_center = (weights[:, None] * x).sum(axis=0) / total
    y_center = float((weights * y).sum() / total)
    centered = x - x_center
    matrix = centered.T @ (weights[:, None] * centered)
    matrix.flat[::width + 1] += regularization * len(feature_maps)
    rhs = centered.T @ (weights * (y - y_center))
    coefficients = np.linalg.solve(matrix, rhs)
    intercept = y_center - float(x_center @ coefficients)
    return SpatialRidgeProbe(coefficients, intercept, mean, scale, grid, regularization)


def image_scores(probability_grid, mask, threshold: float) -> dict[str, float]:
    """Score one image at native mask resolution; pixels are not sample units."""
    import numpy as np
    from PIL import Image

    target = np.asarray(mask)
    if target.ndim != 2 or not np.isin(target, [0, 1]).all():
        raise ValueError("Expected a binary 0/1 crater mask")
    prob = np.asarray(probability_grid, dtype=np.float32)
    if prob.ndim != 2 or not np.isfinite(prob).all() or not (0 < threshold < 1):
        raise ValueError("Invalid probability grid or threshold")
    resized = np.asarray(Image.fromarray(prob, mode="F").resize(
        (target.shape[1], target.shape[0]), Image.Resampling.BILINEAR))
    prediction = resized >= threshold
    truth = target == 1
    intersection = int(np.logical_and(prediction, truth).sum())
    union = int(np.logical_or(prediction, truth).sum())
    total_positive = int(prediction.sum() + truth.sum())
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / total_positive) if total_positive else 1.0,
        "all_foreground_iou": float(truth.mean()),
    }


def image_iou(probability_grid, mask, threshold: float) -> float:
    return image_scores(probability_grid, mask, threshold)["iou"]


def grouped_mean_interval(values, groups, *, replicates: int = 2000, seed: int = 42):
    """Bootstrap source groups, never pixels, for one model's mean score."""
    import numpy as np

    if len(values) != len(groups) or len(values) < 2:
        raise ValueError("At least two grouped image scores are required")
    if not np.isfinite(values).all():
        raise ValueError("Scores must be finite")
    by_group: dict[str, list[float]] = {}
    for value, group in zip(values, groups):
        by_group.setdefault(str(group), []).append(float(value))
    if len(by_group) < 2 or replicates < 100:
        raise ValueError("At least two groups and 100 replicates are required")
    arrays = [np.asarray(scores, dtype=np.float64) for scores in by_group.values()]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        sample = np.concatenate([arrays[index] for index in selected])
        estimates.append(float(sample.mean()))
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def paired_group_difference(scores_a: dict, scores_b: dict, groups: dict[str, str]) -> dict:
    """Compare two models on identical test images, resampling scene groups."""
    import numpy as np

    if set(scores_a) != set(scores_b) or set(scores_a) != set(groups) or len(scores_a) < 2:
        raise ValueError("Paired comparison requires identical nontrivial sample IDs")
    sample_ids = sorted(scores_a)
    differences = [float(scores_a[sample_id]["iou"] - scores_b[sample_id]["iou"])
                   for sample_id in sample_ids]
    if not np.isfinite(differences).all():
        raise ValueError("Paired IoU differences must be finite")
    return {
        "samples": len(sample_ids),
        "mean_iou_difference_a_minus_b": float(np.mean(differences)),
        "group_bootstrap_95pct": grouped_mean_interval(
            differences, [groups[sample_id] for sample_id in sample_ids]),
        "per_image_difference": dict(zip(sample_ids, differences)),
    }


def choose_threshold(probe: SpatialRidgeProbe, feature_maps, masks) -> tuple[float, float]:
    """Choose a deterministic threshold using validation images only."""
    import numpy as np

    if len(feature_maps) != len(masks) or not feature_maps:
        raise ValueError("Validation features and masks must match")
    grids = [probe.predict_grid(features) for features in feature_maps]
    scores = [(float(t), float(np.mean([image_iou(grid, mask, float(t))
                                         for grid, mask in zip(grids, masks)])))
              for t in np.arange(0.1, 0.91, 0.05)]
    return max(scores, key=lambda item: (item[1], -abs(item[0] - 0.5)))
