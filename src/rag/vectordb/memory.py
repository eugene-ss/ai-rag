from __future__ import annotations

import math

from rag.schemas import Chunk, Principal, ScoredChunk
from rag.security.acl import is_allowed


class MemoryVectorStore:
    """In-process cosine similarity store for tests and demos."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}
        self._aliases: dict[str, str] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            msg = "chunks and vectors length mismatch"
            raise ValueError(msg)
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._items[chunk.chunk_id] = (chunk, vector)

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        principal: Principal,
        index_version: str | None = None,
    ) -> list[ScoredChunk]:
        scored: list[tuple[float, Chunk]] = []
        for chunk, stored in self._items.values():
            if index_version and chunk.index_version != index_version:
                continue
            if not is_allowed(principal, chunk.acl):
                continue
            score = _cosine(vector, stored)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[ScoredChunk] = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            results.append(ScoredChunk(chunk=chunk, score=score, rank=rank, retriever="dense"))
        return results

    def delete(self, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            self._items.pop(cid, None)

    def resolve_alias(self, alias: str) -> str | None:
        return self._aliases.get(alias)

    def set_alias(self, alias: str, index_version: str) -> None:
        self._aliases[alias] = index_version

    def count(self) -> int:
        return len(self._items)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
