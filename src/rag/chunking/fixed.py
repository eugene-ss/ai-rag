from __future__ import annotations

from rag.schemas import Chunk, Document, derive_chunk_id


class FixedChunker:
    """Fixed-size character windows with overlap."""

    name = "fixed"
    version = "1.0"

    def chunk(
        self,
        document: Document,
        *,
        index_version: str,
        size: int = 512,
        overlap: int = 64,
    ) -> list[Chunk]:
        text = document.text or ""
        if not text:
            return []
        step = max(size - overlap, 1)
        chunks: list[Chunk] = []
        ordinal = 0
        for start in range(0, len(text), step):
            end = min(start + size, len(text))
            piece = text[start:end]
            if not piece.strip():
                if end >= len(text):
                    break
                continue
            chunks.append(
                Chunk(
                    chunk_id=derive_chunk_id(document.doc_id, ordinal, self.version, index_version),
                    doc_id=document.doc_id,
                    ordinal=ordinal,
                    text=piece,
                    span=(start, end),
                    acl=document.acl,
                    index_version=index_version,
                    metadata={"chunker": self.name, "chunker_version": self.version},
                )
            )
            ordinal += 1
            if end >= len(text):
                break
        return chunks
