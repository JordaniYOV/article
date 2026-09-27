"""Fixture-only backend; responses are not scientific results."""
from collections.abc import Mapping
from typing import Any
from planetary_vlm.contracts import ModelRequest, ModelResponse


class MockAdapter:
    def __init__(self, options: Mapping[str, Any]):
        self.responses = dict(options.get("responses", {}))
        self.default_response = str(options.get("default_response", "UNKNOWN"))

    def predict(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(request.request_id, str(self.responses.get(request.request_id, self.default_response)), "ok", 0.0)

    def close(self) -> None:
        pass
