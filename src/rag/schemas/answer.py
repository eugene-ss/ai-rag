from __future__ import annotations

from pydantic import BaseModel, Field

from rag.schemas.source import SourceKind


class Citation(BaseModel):
    """Pointer from an answer back to the evidence that supports it.

    Originally chunk-shaped (`chunk_id`/`doc_id`). Extended additively for
    multi-source agent answers: `kind`, `ref`, `url`, `title` default so
    existing clients and tests keep working.
    """

    chunk_id: str = ""
    doc_id: str = ""
    quote: str = ""
    score: float = 0.0
    kind: SourceKind = SourceKind.CHUNK
    ref: str = ""
    url: str | None = None
    title: str = ""


class Usage(BaseModel):
    """Token and cost accounting for a generation call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = "echo"
    prompt_version: str = "v1"

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            model=other.model or self.model,
            prompt_version=other.prompt_version or self.prompt_version,
        )


class Answer(BaseModel):
    """Grounded generation result, including explicit refusal.

    A refusal is a normal, successful response: `refused=True` with an empty
    `citations` list. Callers must never treat it as an error.

    Refusal reasons used by the system today:
    - empty_query / no_retrieved_context / low_confidence_score
    - zero_resolvable_citations / llm_unavailable
    - agent_budget_exhausted / agent_no_grounding / tool_unavailable
    """

    text: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    trace_id: str = ""
    index_version: str = ""
    latency_ms: float = 0.0
    cached: bool = False
