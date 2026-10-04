"""Ordered distortion chains with stable per-source, per-effect seeds."""

import hashlib

PARAMETERS = {
    "shear": {"max_shift_px", "control_spacing_px", "scan_axis"},
    "radiation": {"num_hits", "max_streak_length", "strength"},
    "lens_flare": {"strength", "center", "radius", "num_ghosts"},
    "hard_shadows": {"strength", "vertices", "num_shadows"},
}


def validate_parameters(parameters):
    if set(parameters) - PARAMETERS.keys():
        raise ValueError("Unknown distortion parameter group")
    for effect, values in parameters.items():
        if set(values) - PARAMETERS[effect]:
            raise ValueError(f"Unknown parameters for {effect}")
    # Validate using the actual implementations, including bounds and geometry.
    import numpy as np
    apply_chain(np.zeros((256, 256), dtype=np.uint8), list(parameters), parameters, 0, "validation")


def effect_seed(seed, source_id, effect):
    return int.from_bytes(hashlib.sha256(f"{seed}:{source_id}:{effect}".encode()).digest()[:4], "big")


def variants(protocol):
    result = [("clean", [])] if protocol["include_clean"] else []
    result.extend(("+".join(chain), chain) for chain in protocol["distortions"])
    return result


def apply_chain(image, chain, parameters, seed, source_id, mask=None):
    from planetary_vlm.distortion import (
        apply_synchronous_shear, add_radiation_artifacts, apply_lens_flare, apply_hard_shadows)
    for effect in chain:
        kwargs = {**parameters.get(effect, {}), "seed": effect_seed(seed, source_id, effect)}
        if effect == "shear":
            if mask is None:
                image = apply_synchronous_shear(image, image_only=True, **kwargs)[0]
            else:
                image, _, mask = apply_synchronous_shear(image, seg_mask=mask, **kwargs)
        elif effect == "radiation":
            image = add_radiation_artifacts(image, **kwargs)
        elif effect == "lens_flare":
            image = apply_lens_flare(image, **kwargs)
        elif effect == "hard_shadows":
            image = apply_hard_shadows(image, **kwargs)
        else:
            raise ValueError(f"Unknown effect: {effect}")
    return image, mask
