"""Small internal HTTP contract: identifiers only, never labels or images."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

from .files import digest
from . import settings as paths

PROTOCOL = "planetary-model-service-v1"
BACKENDS = {"spacellava": ("spacellava", "mock"), "ibm": ("ibm_imp",), "heads": ("frozen_crater",)}
TERMINAL = {"completed", "cancelled", "interrupted", "blocked", "failed"}


class JobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
    protocol_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


def store_identity(settings):
    return digest(["shared-sqlite-v1", str(paths.WORKSPACE_ROOT), str(settings.database_path)])


def service_for_backend(backend):
    for service, backends in BACKENDS.items():
        if backend in backends:
            return service
    raise ValueError(f"Unsupported service backend: {backend}")
