"""NumPy/OpenCV synthetic robustness transforms with explicit seeded parameters."""

from .geometry import apply_synchronous_shear
from .image_spoil import spoil_image
from .lens_flare import apply_lens_flare
from .radiation import add_radiation_artifacts
from .shadows import apply_hard_shadows

DISTORTION_VERSION = "2.0"

__all__ = [
    "DISTORTION_VERSION", "apply_synchronous_shear", "add_radiation_artifacts",
    "apply_lens_flare", "apply_hard_shadows", "spoil_image",
]
