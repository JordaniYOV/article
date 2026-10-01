"""Synthetic achromatic veiling glare with a halo and optional lens ghosts."""

import numpy as np

from ._common import generator, integer, number, restore_image, unit_image, validate_image


def apply_lens_flare(
    image: np.ndarray,
    strength: float = 0.3,
    seed: int | None = 0,
    *,
    center: tuple[float, float] | None = None,
    radius: float = 0.3,
    num_ghosts: int = 3,
) -> np.ndarray:
    """Add a smooth glow, halo and ghosts along the source-to-image-center axis.

    center is normalized (x, y) in [0, 1]; None samples a seeded location.
    radius is the main glow sigma as a fraction of the shorter image side.
    strength in [0, 1] blends towards white; channel order does not matter.
    This is a stress-test overlay, not a calibrated optical model.
    """
    validate_image(image)
    strength = number(strength, "strength", maximum=1)
    radius = number(radius, "radius")
    if radius == 0:
        raise ValueError("radius must be > 0")
    num_ghosts = integer(num_ghosts, "num_ghosts")
    rng = generator(seed)
    if center is None:
        cx, cy = rng.uniform(0, 1, 2)
    else:
        if len(center) != 2:
            raise ValueError("center must be normalized (x, y)")
        cx, cy = (number(value, "center coordinate", maximum=1) for value in center)
    if strength == 0:
        return image.copy()

    h, w = image.shape[:2]
    cx, cy = cx * (w - 1), cy * (h - 1)
    y, x = np.mgrid[:h, :w]
    sigma = max(radius * min(h, w), 0.5)
    distance = np.hypot(x - cx, y - cy)
    profile = np.exp(-0.5 * (distance / sigma) ** 2)
    profile += 0.15 * np.exp(-0.5 * ((distance - 1.7 * sigma) / (0.15 * sigma)) ** 2)
    middle = ((w - 1) / 2, (h - 1) / 2)
    for index in range(num_ghosts):
        position = 0.5 + 1.5 * (index + 1) / (num_ghosts + 1)
        gx = cx + position * (middle[0] - cx)
        gy = cy + position * (middle[1] - cy)
        ghost_sigma = max(sigma * rng.uniform(0.08, 0.22), 0.5)
        ghost = np.exp(-0.5 * ((x - gx) ** 2 + (y - gy) ** 2) / ghost_sigma ** 2)
        profile += rng.uniform(0.1, 0.3) * ghost
    alpha = strength * np.clip(profile, 0, 1)
    if image.ndim == 3:
        alpha = alpha[..., None]
    values = unit_image(image)
    return restore_image(values + alpha * (1 - values), image)
