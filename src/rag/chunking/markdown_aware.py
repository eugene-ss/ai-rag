from __future__ import annotations

import re

from rag.chunking.recursive import RecursiveChunker
from rag.schemas import Chunk, Document, derive_chunk_id


class MarkdownAwareChunker:
    """Splits markdown primarily on headings, then recursively within sections."""

    name = "markdown_aware"
    version = "1.0"

    def __init__(self) -> None:
        self._fallback = RecursiveChunker()

    def chunk(
        self,
        document: Document,
        *,
        index_version: str,
        size: int = 800,
        overlap: int = 80,
    ) -> list[Chunk]:
        text = document.text or ""
        if not text:
            return []
        sections = re.split(r"(?=^#{1,6}\s)", text, flags=re.M)
        sections = [s for s in sections if s.strip()]
        if not sections:
            return self._fallback.chunk(
                document, index_version=index_version, size=size, overlap=overlap
            )

        chunks: list[Chunk] = []
        ordinal = 0
        cursor = 0
        for section in sections:
            pieces = (
                [section] if len(section) <= size else self._fallback._split(section, size)  # noqa: SLF001
            )
            for piece in pieces:
                start = text.find(piece, cursor)
                if start < 0:
                    start = cursor
                end = start + len(piece)
                chunks.append(
                    Chunk(
                        chunk_id=derive_chunk_id(
                            document.doc_id, ordinal, self.version, index_version
                        ),
                        doc_id=document.doc_id,
                        ordinal=ordinal,
                        text=piece.strip(),
                        span=(start, end),
                        acl=document.acl,
                        index_version=index_version,
                        metadata={
                            "chunker": self.name,
                            "chunker_version": self.version,
                        },
                    )
                )
                ordinal += 1
                cursor = end
        return chunks
