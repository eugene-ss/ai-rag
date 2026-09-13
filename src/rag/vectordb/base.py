from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.schemas import Chunk, Principal, ScoredChunk


@runtime_checkable
class VectorStore(Protocol):
    """Dense vector index with ACL-aware search."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        principal: Principal,
        index_version: str | None = None,
    ) -> list[ScoredChunk]: ...

    def delete(self, chunk_ids: list[str]) -> None: ...

    def resolve_alias(self, alias: str) -> str | None:
        """Index version currently published under `alias`, if any."""
        ...

    def set_alias(self, alias: str, index_version: str) -> None:
        """Publish `index_version` under `alias`. The atomic cutover."""
        ...

    def count(self, index_version: str | None = None) -> int:
        """Number of indexed vectors, optionally for one version.

        Readiness must count the version the alias publishes; counting whatever
        this process was built with can report ready while serving nothing.
        """
        ...
