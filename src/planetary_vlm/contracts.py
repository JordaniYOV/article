"""Minimal in-memory boundary; a versioned on-disk manifest is still pending."""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ModelRequest:
    """Model-visible data only: never add targets, masks or outcome telemetry."""

    request_id: str
    image_paths: tuple[str, ...]
    prompt: str
    allowed_answers: tuple[str, ...]
    max_new_tokens: int = 32
    do_sample: bool = False

    def __post_init__(self) -> None:
        if not self.request_id or not self.image_paths or not self.prompt.strip():
            raise ValueError("request_id, image_paths and prompt are required")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True)
class GroundTruth:
    """Evaluation-only target, joined by request_id after inference."""

    request_id: str
    answer: str
    source_sample_id: str
    group_id: str
    derivation_version: str


@dataclass(frozen=True)
class ModelResponse:
    """Keep raw output; parsing invalidity is separate from runtime failure."""

    request_id: str
    raw_response: str
    status: Literal["ok", "error", "unsupported"]
    elapsed_seconds: float
    error_message: str | None = None


class ModelAdapter(Protocol):
    def predict(self, request: ModelRequest) -> ModelResponse:
        """Run a single model request without access to GroundTruth."""
        ...
