from __future__ import annotations

from typing import Protocol

from rag.schemas import Chunk, Document


class Chunker(Protocol):
    """Splits a Document into Chunks."""

    name: str
    version: str

    def chunk(
        self,
        document: Document,
        *,
        index_version: str,
        size: int = 512,
        overlap: int = 64,
    ) -> list[Chunk]: ...
