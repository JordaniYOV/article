"""Resolve hash-identical provenance retained after retiring the v3 image copy."""
from .orbital_questions import file_hash, orbital_path


def retained_orbital_path(root, relative, expected_sha256=None):
    original = orbital_path(root, relative)
    candidates = [original]
    old = "data_orbital/orbital_300_v3/"
    if relative.startswith(old) and not original.exists():
        suffix = relative[len(old):]
        candidates.append(orbital_path(root, "data_orbital/provenance/orbital_300_v3/" + suffix))
        # Images may be deduplicated against v4, but only by their original hash.
        # Manifests must retain their original bytes, never use the v4 manifest.
        if expected_sha256 is not None:
            candidates.append(orbital_path(root, "data_orbital/orbital_300_v4/" + suffix))
    for candidate in candidates:
        if candidate.is_file():
            if expected_sha256 is not None and file_hash(candidate) != expected_sha256:
                if candidate == original:
                    raise ValueError("Original orbital asset hash changed: " + relative)
                continue
            return candidate
    raise FileNotFoundError("Missing retained orbital provenance: " + relative)
