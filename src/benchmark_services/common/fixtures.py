"""Explicit non-scientific service smoke fixtures; never read targets."""

import json

from planetary_vlm.contracts import ModelResponse
from .segmentation import save_maps


class ServiceFixture:
    def __init__(self, backend, options):
        self.backend, self.options = backend, options

    def initialize(self):
        return {"fixture": True, "mode": "service_mock", "backend": self.backend}

    def predict(self, request):
        if self.backend == "ibm_imp":
            import numpy as np
            logits = np.zeros((2, 256, 256), dtype=np.float32)
            logits[0] = 1
            raw = json.dumps(save_maps(self.options["artifact_dir"], request.request_id, logits))
        else:
            raw = "MOCK_OUTPUT"
        return ModelResponse(request.request_id, raw, "ok", 0)

    def close(self):
        pass
