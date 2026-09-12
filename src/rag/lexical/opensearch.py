from __future__ import annotations

from rag.exceptions import MissingBackendError
from rag.schemas import Chunk, Principal, ScoredChunk


class OpenSearchLexicalIndex:
    """Lazy OpenSearch adapter. Requires the 'opensearch' extra."""

    def __init__(self, url: str = "http://localhost:9200", index: str = "docs") -> None:
        try:
            import opensearchpy  # noqa: F401
        except ImportError as exc:
            raise MissingBackendError("OpenSearchLexicalIndex", "opensearch") from exc
        self.url = url
        self.index = index

    def upsert(self, chunks: list[Chunk]) -> None:
        raise MissingBackendError("OpenSearchLexicalIndex", "opensearch")

    def search(
        self,
        query: str,
        *,
        top_k: int,
        principal: Principal,
        index_version: str | None = None,
    ) -> list[ScoredChunk]:
        raise MissingBackendError("OpenSearchLexicalIndex", "opensearch")

    def delete(self, chunk_ids: list[str]) -> None:
        raise MissingBackendError("OpenSearchLexicalIndex", "opensearch")
