"""Versioned label/evidence format checks, never an evidence-truth judge."""

import re

from .parser import parse_answer, validate_answers

SCHEMA = "label-evidence-two-lines-v1"
_FORMAT = re.compile(r"ANSWER:[ \t]*([^\r\n]+)(?:\r?\nEVIDENCE:[ \t]*([^\r\n]*))?")


def parse_label_evidence(raw_response, allowed_answers):
    """Only explicit fields; never search prose for a favorable label.

    A recognizable ANSWER alone can supply a label, but not a valid two-line
    response. Nonempty evidence is not proof of correctness or consistency.
    Unlisted abstentions remain invalid labels, never NO/NONE ground truth.
    """
    validate_answers(allowed_answers)
    if not isinstance(raw_response, str):
        raise ValueError("raw_response must be a string")
    match = _FORMAT.fullmatch(raw_response.strip())
    answer_text = match.group(1).strip() if match else None
    evidence = match.group(2).strip() if match and match.group(2) is not None else None
    label = parse_answer(answer_text, allowed_answers) if answer_text else None
    return {"schema": SCHEMA, "parsed_answer": label, "answer_field": answer_text,
        "label_valid": label is not None, "format_valid": label is not None and bool(evidence),
        "evidence": evidence, "evidence_review_status": "NOT_REVIEWED" if evidence else "NOT_PROVIDED",
        "unlisted_abstention": label is None and answer_text is not None
            and answer_text.upper() in {"UNSURE", "UNKNOWN", "UNCERTAIN", "CANNOT DETERMINE"}}
