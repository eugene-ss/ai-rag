from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.schemas import Chunk, Principal, ScoredChunk


@runtime_checkable
class LexicalIndex(Protocol):
    """Sparse / lexical index with ACL-aware search."""

    def upsert(self, chunks: list[Chunk]) -> None: ...

    def search(
        self,
        query: str,
        *,
        top_k: int,
        principal: Principal,
        index_version: str | None = None,
    ) -> list[ScoredChunk]: ...

    def delete(self, chunk_ids: list[str]) -> None: ...
