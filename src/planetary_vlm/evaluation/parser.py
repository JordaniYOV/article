"""Deterministic closed-form parser. Never extract labels from prose."""


def normalize_label(text: str) -> str:
    """Ignore case and leading/trailing/repeated Unicode whitespace only."""
    if not isinstance(text, str):
        raise ValueError("label must be a string")
    return " ".join(text.split()).casefold()


def validate_answers(allowed_answers: list[str] | tuple[str, ...]) -> None:
    if not isinstance(allowed_answers, (list, tuple)):
        raise ValueError("allowed_answers must be a list or tuple")
    normalized = [normalize_label(label) for label in allowed_answers]
    if any(not label for label in normalized):
        raise ValueError("allowed_answers cannot contain blank labels")
    if len(set(normalized)) != len(normalized):
        raise ValueError("allowed_answers contain ambiguous normalized labels")


def parse_answer(raw_response: str, allowed_answers: list[str] | tuple[str, ...]) -> str | None:
    """Return the canonical supplied label or None for invalid/free-form output."""
    validate_answers(allowed_answers)
    normalized = normalize_label(raw_response)
    return next((label for label in allowed_answers
                 if normalize_label(label) == normalized), None)
