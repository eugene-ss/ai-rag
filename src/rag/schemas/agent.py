"""Contracts for the agentic plan/act/critique loop.

`AgentAnswer` subclasses `Answer` so `/query` stays response-compatible: callers
that only know `Answer` keep working; agent-aware callers can read the trace.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.schemas.answer import Answer, Usage
from rag.schemas.source import Source


class StopReason(StrEnum):
    """Why the agent loop terminated."""

    ANSWERED = "answered"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CRITIC_GROUNDED = "critic_grounded"
    CRITIC_EXHAUSTED = "critic_exhausted"
    NO_GROUNDING = "no_grounding"
    TOOL_UNAVAILABLE = "tool_unavailable"
    FALLBACK_PIPELINE = "fallback_pipeline"
    REFUSED = "refused"
    ERROR = "error"


class ToolCall(BaseModel):
    """A tool invocation requested by the planning model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Outcome of one tool call, including any sources it produced."""

    call_id: str
    name: str
    ok: bool = True
    content: str = ""
    sources: list[Source] = Field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0


class AgentStep(BaseModel):
    """One plan → (optional tools) → (optional draft) cycle."""

    index: int
    kind: Literal["plan", "act", "critique", "finalize", "fallback"] = "plan"
    thought: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    draft: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0


class Verdict(BaseModel):
    """Critic judgement over a draft answer and its sources.

    The critic never sees the agent's chain of thought — only the draft and the
    sources — so it cannot rubber-stamp the agent's own reasoning.
    """

    sufficient: bool
    grounded: bool
    reasons: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class AgentTrace(BaseModel):
    """Internal reasoning trace. Not exposed in API responses by default."""

    steps: list[AgentStep] = Field(default_factory=list)
    stop_reason: StopReason = StopReason.ANSWERED
    sources: list[Source] = Field(default_factory=list)
    critique_rounds: int = 0
    total_tool_calls: int = 0
    self_corrections: int = 0
    fallback_used: bool = False


class AgentAnswer(Answer):
    """Answer plus agent metadata.

    Subclassing keeps `/query` response-compatible: unknown fields are ignored
    by clients that only deserialize `Answer`.
    """

    mode: Literal["agent"] = "agent"
    stop_reason: StopReason = StopReason.ANSWERED
    steps_taken: int = 0
    tool_calls: int = 0
    tools_used: list[str] = Field(default_factory=list)
    self_corrections: int = 0
    fallback_used: bool = False
    # Trace is omitted from the public response unless explicitly requested.
    trace: AgentTrace | None = None
