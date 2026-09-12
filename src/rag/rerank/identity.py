from __future__ import annotations

from rag.schemas import ScoredChunk


class IdentityReranker:
    """Pass-through reranker: preserves order, updates retriever label."""

    def rerank(self, query: str, candidates: list[ScoredChunk], *, top_k: int) -> list[ScoredChunk]:
        _ = query
        results: list[ScoredChunk] = []
        for rank, item in enumerate(candidates[:top_k], start=1):
            results.append(
                ScoredChunk(
                    chunk=item.chunk,
                    score=item.score,
                    rank=rank,
                    retriever="rerank",
                )
            )
        return results
