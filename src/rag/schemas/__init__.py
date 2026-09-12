"""Explicit contracts for Document, Chunk, QueryResult, Answer, and Agent."""

from rag.schemas.agent import (
    AgentAnswer,
    AgentStep,
    AgentTrace,
    StopReason,
    ToolCall,
    ToolResult,
    Verdict,
)
from rag.schemas.answer import Answer, Citation, Usage
from rag.schemas.chunk import Chunk, ScoredChunk, derive_chunk_id
from rag.schemas.document import Document
from rag.schemas.principal import AclTags, Principal
from rag.schemas.query import QueryResult
from rag.schemas.source import Source, SourceKind

__all__ = [
    "AclTags",
    "AgentAnswer",
    "AgentStep",
    "AgentTrace",
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "Principal",
    "QueryResult",
    "ScoredChunk",
    "Source",
    "SourceKind",
    "StopReason",
    "ToolCall",
    "ToolResult",
    "Usage",
    "Verdict",
    "derive_chunk_id",
]
