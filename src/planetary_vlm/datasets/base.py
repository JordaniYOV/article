"""Shared dataset lifecycle, paths and immutable experiment versions."""
from abc import ABC, abstractmethod
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class BaseData(ABC):
    """Domain-independent interface; imports do not load image/GPU backends."""
    domain: str

    def __init__(self, root: str | Path = "data_rover"):
        self.root = Path(root).resolve()

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def evidence(self) -> Path:
        return self.root / "source_audit_v1"

    def new_version(self, destination: str | Path) -> Path:
        destination = Path(destination).resolve()
        if destination.exists():
            raise FileExistsError(f"Version already exists; choose a new destination: {destination}")
        if destination == self.root or self.root.is_relative_to(destination):
            raise ValueError("Output must not contain the source data root")
        return destination

    @abstractmethod
    def download(self, **options):
        """Explicit network operation, never triggered by prepare or inference."""

    @abstractmethod
    def prepare(self, *, output, specification=None, samples=None):
        """Offline preparation with evaluation targets isolated from requests."""
