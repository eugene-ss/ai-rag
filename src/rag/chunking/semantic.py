from __future__ import annotations

from rag.chunking.recursive import RecursiveChunker
from rag.embedding.base import Embedder
from rag.schemas import Chunk, Document, derive_chunk_id


class SemanticChunker:
    """Splits on embedding-similarity troughs between adjacent sentences.

    Needs an embedder. Without one it falls back to recursive splitting rather
    than failing, so a misconfiguration degrades quality instead of breaking
    ingestion.
    """

    name = "semantic"
    version = "1.0"

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        breakpoint_percentile: float = 0.75,
    ) -> None:
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self._fallback = RecursiveChunker()

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
        if self.embedder is None:
            chunks = self._fallback.chunk(
                document, index_version=index_version, size=size, overlap=overlap
            )
            for c in chunks:
                c.metadata.update(
                    {
                        "chunker": self.name,
                        "chunker_version": self.version,
                        "semantic_fallback": True,
                    }
                )
            return chunks
        return self._semantic_chunks(document, text, index_version=index_version, size=size)

    def _semantic_chunks(
        self,
        document: Document,
        text: str,
        *,
        index_version: str,
        size: int,
    ) -> list[Chunk]:
        assert self.embedder is not None
        sentences = _split_sentences(text)
        if len(sentences) < 2:
            return self._fallback.chunk(document, index_version=index_version, size=size)

        vectors = self.embedder.embed(sentences)
        distances = [1.0 - _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        threshold = _percentile(distances, self.breakpoint_percentile)

        groups: list[list[str]] = [[sentences[0]]]
        for i, sentence in enumerate(sentences[1:]):
            too_long = sum(len(s) for s in groups[-1]) + len(sentence) > size
            if distances[i] >= threshold or too_long:
                groups.append([sentence])
            else:
                groups[-1].append(sentence)

        chunks: list[Chunk] = []
        cursor = 0
        for ordinal, group in enumerate(groups):
            body = " ".join(group).strip()
            if not body:
                continue
            start = text.find(group[0], cursor)
            if start < 0:
                start = cursor
            end = start + len(body)
            cursor = end
            chunks.append(
                Chunk(
                    chunk_id=derive_chunk_id(document.doc_id, ordinal, self.version, index_version),
                    doc_id=document.doc_id,
                    ordinal=ordinal,
                    text=body,
                    span=(start, end),
                    acl=document.acl,
                    index_version=index_version,
                    metadata={"chunker": self.name, "chunker_version": self.version},
                )
            )
        return chunks


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(percentile * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]
