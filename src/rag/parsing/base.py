from __future__ import annotations

from typing import Protocol

from rag.schemas import Document


class Parser(Protocol):
    """Extracts normalized text from a Document."""

    mime_types: tuple[str, ...]

    def parse(self, document: Document) -> Document: ...
