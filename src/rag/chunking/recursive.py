from __future__ import annotations

from rag.schemas import Chunk, Document, derive_chunk_id

_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


class RecursiveChunker:
    """Recursively splits on natural separators until size fits."""

    name = "recursive"
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
        pieces = self.split_text(text, size)
        chunks: list[Chunk] = []
        cursor = 0
        for ordinal, piece in enumerate(pieces):
            start = text.find(piece, cursor)
            if start < 0:
                start = cursor
            end = start + len(piece)
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
            cursor = max(end - overlap, end)
        return chunks

    def split_text(
        self, text: str, size: int, separators: tuple[str, ...] = _SEPARATORS
    ) -> list[str]:
        """Split text to fit `size`, preferring the largest natural boundary."""
        if len(text) <= size:
            return [text] if text.strip() else []
        sep = separators[0] if separators else ""
        rest = separators[1:] if separators else ()
        if sep == "":
            return [text[i : i + size] for i in range(0, len(text), size)]
        parts = text.split(sep)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else current + sep + part
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > size:
                    chunks.extend(self.split_text(part, size, rest))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks
