"""Audited legacy-code equivalence for completed export/inference ONLY.

Original checkpoints/identities are never rewritten. Training resume keeps its
original exact-code guard. Unknown code and environment changes still fail.
"""
import hashlib
import itertools
from pathlib import Path

from .common import fingerprint, read_json, sha


def code_manifest(package, normalize_newlines=False):
    result = {}
    for path in sorted(Path(package).glob("*.py")):
        content = path.read_bytes()
        if normalize_newlines:
            content = content.replace(b"\r\n", b"\n")
        result[path.name] = hashlib.sha256(content).hexdigest()
    return result


def matches_historical_hash(expected, files):
    """Only exact historical manifests; account for Windows Git LF/CRLF checkout."""
    names = sorted(files)
    variants = [tuple(dict.fromkeys(files[name])) for name in names]
    if not names or len(names) > 16 or any(not values or len(values) > 2 for values in variants):
        raise ValueError("Invalid audited historical code manifest")
    return any(fingerprint(dict(zip(names, values))) == expected for values in itertools.product(*variants))


def code_compatibility(identity, package=None, policy_path=None):
    package = Path(package) if package is not None else Path(__file__).parent
    current = fingerprint(code_manifest(package))
    training = identity["code"]
    if identity["feature_identity"].get("code") != training:
        raise ValueError("Feature/training code identities disagree; cannot attest compatibility")
    if current == training:
        return {"mode": "exact", "training_code": training, "runtime_code": current}
    policy_path = Path(policy_path) if policy_path is not None else package / "compatibility_revisions.json"
    if not policy_path.is_file():
        raise ValueError("Training code changed and audited compatibility policy is missing; restore the exact training checkout")
    policy = read_json(policy_path)
    if (policy.get("schema") != "frozen-heads-code-compatibility-v1"
            or code_manifest(package, True) != policy.get("runtime_sha256")):
        raise ValueError("Current code differs from the audited compatible runtime; restore the exact training checkout")
    encoder = identity["feature_identity"]["encoder"]
    for profile in policy["profiles"]:
        if encoder in profile["encoders"] and matches_historical_hash(training, profile["file_sha256"]):
            return {"mode": "audited_compatibility", "profile": profile["id"],
                    "historical_git_revision": profile["git_revision"], "training_code": training,
                    "runtime_code": current, "policy_sha256": sha(policy_path),
                    "scope": "completed_export_and_frozen_inference_only", "reason": policy["reason"]}
    raise ValueError(
        f"Training code {training[:12]} is not an audited compatible {encoder} revision "
        f"(current {current[:12]}). Restore the exact training checkout; do not edit identity.json."
    )


def verify_deployment_code(deployment):
    """The worker must honor the SAME compatibility attestation as publish."""
    checked = code_compatibility(deployment["training_identity"])
    recorded = deployment.get("code_compatibility")
    if (recorded is not None and recorded != checked) or (checked["mode"] != "exact" and recorded is None):
        raise ValueError("Runtime compatibility attestation differs; republish the original unchanged checkpoint")
    return checked
