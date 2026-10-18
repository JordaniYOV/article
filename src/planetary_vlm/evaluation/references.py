"""Reference admission is independent of whether a model produced an answer."""

READY = "REFERENCE_READY"
CANDIDATE = "CANDIDATE_REFERENCE"
UNKNOWN = "UNKNOWN_REFERENCE"


def reference_status(record, *, has_reference, require_explicit=False):
    """Return (status, reason); never convert missing/candidate annotations to NO.

    A plain GroundTruth record is a caller-declared reference for compatibility.
    Rich source manifests can explicitly withhold admission. `scored=False`
    means 'not yet evaluated', not 'annotation rejected', and is not a gate.
    """
    declared = record.get("reference_status")
    if declared is not None and declared not in {READY, CANDIDATE, UNKNOWN}:
        raise ValueError("Unknown reference_status")
    review = record.get("review_status")
    if review is not None and review not in {"accepted", "pending", "pending_second_review", "rejected"}:
        raise ValueError("Unknown review_status")
    admitted = record.get("scoring_eligible")
    if admitted is not None and type(admitted) is not bool:
        raise ValueError("scoring_eligible must be a boolean")
    withheld = review in {"pending", "pending_second_review", "rejected"} or admitted is False
    if declared == READY and (withheld or not has_reference):
        raise ValueError("REFERENCE_READY contradicts reference data or admission flags")
    if declared in {CANDIDATE, UNKNOWN}:
        return declared, "reference_not_admitted"
    if withheld:
        return CANDIDATE if has_reference else UNKNOWN, "reference_review_pending_or_withheld"
    if not has_reference:
        return UNKNOWN, "reference_missing"
    if require_explicit and declared != READY and review != "accepted" and admitted is not True:
        return CANDIDATE, "explicit_crater_reference_admission_required"
    return READY, "declared_reference"
