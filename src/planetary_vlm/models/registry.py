"""Lazy backend dispatch."""
from collections.abc import Mapping
from typing import Any

_CUSTOM_ADAPTERS = {}


def register_adapter(backend: str, factory) -> None:
    """Register a custom factory without changing the runner or built-in names."""
    if not backend or backend in {"mock", "hf_vlm", "hf_text", "seleno_native"} or backend in _CUSTOM_ADAPTERS:
        raise ValueError(f"Backend already exists or is invalid: {backend}")
    if not callable(factory):
        raise TypeError("factory must be callable")
    _CUSTOM_ADAPTERS[backend] = factory


def create_adapter(backend: str, options: Mapping[str, Any]):
    if backend in _CUSTOM_ADAPTERS:
        return _CUSTOM_ADAPTERS[backend](options)
    if backend == "mock":
        from .mock import MockAdapter
        return MockAdapter(options)
    if backend in {"hf_vlm", "hf_text"}:
        from .huggingface import HuggingFaceAdapter
        return HuggingFaceAdapter(options, visual=backend == "hf_vlm")
    if backend == "seleno_native":
        from .seleno import SelenoNativeAdapter
        return SelenoNativeAdapter(options)
    raise ValueError(f"Unknown model backend: {backend}")
