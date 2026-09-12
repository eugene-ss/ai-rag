"""Shared exceptions for optional backends."""

from __future__ import annotations


class MissingBackendError(ImportError):
    """Raised when an optional backend extra is not installed."""

    def __init__(self, backend: str, extra: str) -> None:
        super().__init__(
            f"{backend} requires the '{extra}' extra. Install with: uv sync --extra {extra}"
        )
        self.backend = backend
        self.extra = extra
