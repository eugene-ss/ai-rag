from __future__ import annotations

from rag.chunking.recursive import RecursiveChunker
from rag.exceptions import MissingBackendError
from rag.schemas import Chunk, Document


class SemanticChunker:
    """Stub semantic chunker. Falls back or raises if forced without extras."""

    name = "semantic"
    version = "1.0"

    def __init__(self, *, require_backend: bool = False) -> None:
        self.require_backend = require_backend
        self._fallback = RecursiveChunker()

    def chunk(
        self,
        document: Document,
        *,
        index_version: str,
        size: int = 512,
        overlap: int = 64,
    ) -> list[Chunk]:
        if self.require_backend:
            raise MissingBackendError("SemanticChunker", "rerank")
        chunks = self._fallback.chunk(
            document, index_version=index_version, size=size, overlap=overlap
        )
        for c in chunks:
            c.metadata["chunker"] = self.name
            c.metadata["chunker_version"] = self.version
            c.metadata["semantic_fallback"] = True
        return chunks
