"""Model-independent boundary. Ground truth belongs to evaluation only."""

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
    modality_paths: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if type(self.do_sample) is not bool:
            raise ValueError("do_sample must be a boolean")
        if not isinstance(self.image_paths, tuple) or any(
            not isinstance(path, str) or not path for path in self.image_paths
        ):
            raise ValueError("image_paths must contain nonempty strings")
        if not isinstance(self.allowed_answers, tuple) or any(
            not isinstance(label, str) or not label.strip() for label in self.allowed_answers
        ):
            raise ValueError("allowed_answers must contain nonempty labels")
        normalized = [" ".join(label.split()).casefold() for label in self.allowed_answers]
        if len(set(normalized)) != len(normalized):
            raise ValueError("allowed_answers must be distinct after normalization")
        if not isinstance(self.modality_paths, tuple):
            raise ValueError("modality_paths must be a tuple of name/path pairs")
        for pair in self.modality_paths:
            if (not isinstance(pair, tuple) or len(pair) != 2
                    or any(not isinstance(item, str) or not item for item in pair)):
                raise ValueError("modality_paths requires nonempty name/path pairs")
        if len({name for name, _ in self.modality_paths}) != len(self.modality_paths):
            raise ValueError("duplicate modality names")


@dataclass(frozen=True)
class GroundTruth:
    """Evaluation-only target, joined by request_id after inference."""

    request_id: str
    answer: str
    source_sample_id: str
    group_id: str
    derivation_version: str
    task_id: str = "default"
    condition_id: str = "clean"


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

    def close(self) -> None:
        """Release backend resources before the next model is loaded."""
        ...
