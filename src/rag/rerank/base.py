from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.schemas import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieved chunks before generation."""

    def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]: ...
