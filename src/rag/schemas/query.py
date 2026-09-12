from __future__ import annotations

from pydantic import BaseModel, Field

from rag.schemas.chunk import ScoredChunk


class QueryResult(BaseModel):
    """Output of hybrid retrieval before generation."""

    query: str
    rewritten: list[str] = Field(default_factory=list)
    results: list[ScoredChunk] = Field(default_factory=list)
    index_version: str
    trace_id: str
    latency_ms: float = 0.0
    cache_hit: bool = False
