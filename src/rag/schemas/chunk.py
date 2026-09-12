from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.schemas.principal import AclTags
from rag.utils.hashing import short_id


def derive_chunk_id(
    doc_id: str,
    ordinal: int,
    chunker_version: str,
    index_version: str,
) -> str:
    """Deterministic chunk id so reindexing is idempotent and diffable."""
    return short_id(doc_id, str(ordinal), chunker_version, index_version)


class Chunk(BaseModel):
    """Indexed text unit with ACL and index version."""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    span: tuple[int, int]
    acl: AclTags
    index_version: str
    embedding_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """Chunk with retrieval score and provenance."""

    chunk: Chunk
    score: float
    rank: int
    retriever: Literal["dense", "lexical", "fused", "rerank"]
