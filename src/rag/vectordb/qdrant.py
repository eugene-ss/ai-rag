from __future__ import annotations

from rag.exceptions import MissingBackendError
from rag.schemas import Chunk, Principal, ScoredChunk


class QdrantVectorStore:
    """Lazy Qdrant adapter. Requires the 'qdrant' extra."""

    def __init__(self, url: str = "http://localhost:6333", collection: str = "docs") -> None:
        try:
            import qdrant_client  # noqa: F401
        except ImportError as exc:
            raise MissingBackendError("QdrantVectorStore", "qdrant") from exc
        self.url = url
        self.collection = collection

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise MissingBackendError("QdrantVectorStore", "qdrant")

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        principal: Principal,
        index_version: str | None = None,
    ) -> list[ScoredChunk]:
        raise MissingBackendError("QdrantVectorStore", "qdrant")

    def delete(self, chunk_ids: list[str]) -> None:
        raise MissingBackendError("QdrantVectorStore", "qdrant")

    def resolve_alias(self, alias: str) -> str | None:
        raise MissingBackendError("QdrantVectorStore", "qdrant")

    def set_alias(self, alias: str, index_version: str) -> None:
        raise MissingBackendError("QdrantVectorStore", "qdrant")
