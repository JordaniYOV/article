"""Fixed-order composition of the four synthetic image corruptions."""

import numpy as np

from ._common import seed_sequence
from .geometry import apply_synchronous_shear
from .lens_flare import apply_lens_flare
from .radiation import add_radiation_artifacts
from .shadows import apply_hard_shadows


def spoil_image(
    image: np.ndarray,
    dem_mask: np.ndarray | None = None,
    seg_mask: np.ndarray | None = None,
    *,
    max_shift_px: float = 3.0,
    control_spacing_px: int = 32,
    scan_axis: str = "both",
    num_hits: int = 20,
    max_streak_length: int = 5,
    flare_strength: float = 0.3,
    shadow_strength: float = 0.6,
    seed: int | None = 0,
    seg_ignore_label: int = 255,
    dem_nodata: float = np.nan,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply shear -> shadows -> lens flare -> particle hits to one image.

    Only geometric jitter changes the selected target map. Each random effect has a
    separate seed stream, so changing one effect's draw count cannot move
    another effect. Zero shift/strength/hit count disables that stage.
    """
    shadow_seed, flare_seed, radiation_seed, geometry_seed = [
        int(child.generate_state(1)[0]) for child in seed_sequence(seed).spawn(4)
    ]
    result, dem, seg = apply_synchronous_shear(
        image, max_shift_px, dem_mask, seg_mask,
        control_spacing_px=control_spacing_px, scan_axis=scan_axis, seed=geometry_seed,
        seg_ignore_label=seg_ignore_label, dem_nodata=dem_nodata,
    )
    result = apply_hard_shadows(result, shadow_strength, shadow_seed)
    result = apply_lens_flare(result, flare_strength, flare_seed)
    result = add_radiation_artifacts(
        result, num_hits, radiation_seed, max_streak_length=max_streak_length,
    )
    return result, dem, seg
