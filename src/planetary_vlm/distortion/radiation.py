"""Synthetic bright particle-hit points and thin streaks; no dose simulation."""

import cv2
import numpy as np

from ._common import generator, integer, number, restore_image, unit_image, validate_image


def add_radiation_artifacts(
    image: np.ndarray,
    num_hits: int = 20,
    seed: int | None = 0,
    *,
    max_streak_length: int = 5,
    strength: float = 1.0,
) -> np.ndarray:
    """Add num_hits randomly placed one-pixel-wide bright events.

    strength is added intensity in [0, 1]; clipping simulates saturation.
    max_streak_length=1 gives points only. Hits may overlap or be clipped by
    the image edge, so num_hits is not the number of modified pixels.
    """
    validate_image(image)
    num_hits = integer(num_hits, "num_hits")
    max_streak_length = integer(max_streak_length, "max_streak_length", minimum=1)
    strength = number(strength, "strength", maximum=1)
    rng = generator(seed)
    if num_hits == 0 or strength == 0:
        return image.copy()

    h, w = image.shape[:2]
    events = np.zeros((h, w), dtype=np.float32)
    for _ in range(num_hits):
        x, y = int(rng.integers(w)), int(rng.integers(h))
        length = int(rng.integers(1, max_streak_length + 1))
        angle = rng.uniform(0, 2 * np.pi)
        end = (x + round((length - 1) * np.cos(angle)),
               y + round((length - 1) * np.sin(angle)))
        cv2.line(events, (x, y), end, float(strength), thickness=1, lineType=cv2.LINE_8)
    if image.ndim == 3:
        events = events[..., None]
    return restore_image(unit_image(image) + events, image)
