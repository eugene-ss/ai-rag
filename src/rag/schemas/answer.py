from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Pointer from answer span back to a retrieved chunk."""

    chunk_id: str
    doc_id: str
    quote: str
    score: float = 0.0


class Usage(BaseModel):
    """Token / cost accounting for a generation call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = "echo"
    prompt_version: str = "v1"


class Answer(BaseModel):
    """Grounded generation result, including explicit refusal."""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    trace_id: str = ""
