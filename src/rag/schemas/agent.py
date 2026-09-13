"""Contracts for the agentic plan/act/critique loop.

`AgentAnswer` subclasses `Answer` so `/query` stays response-compatible: callers
that only know `Answer` keep working; agent-aware callers can read the trace.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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


#: Stop reasons under which the agent produced a cited answer under its own
#: power. Single source of truth: the runtime decides whether to keep a draft,
#: `/query` decides whether to cache it, and the eval metrics decide whether to
#: score it as a success — all three must agree or the system contradicts itself.
#:
#: `CRITIC_EXHAUSTED` belongs here because the critique budget gates the critic,
#: not the agent: running it out still leaves a grounded answer the caller got.
ANSWERABLE_STOPS = frozenset(
    {
        StopReason.ANSWERED,
        StopReason.CRITIC_GROUNDED,
        StopReason.CRITIC_EXHAUSTED,
    }
)


class Budget(BaseModel):
    """Hard caps for one agent turn. Caller-supplied values are clamped by the server.

    Lives in `schemas` rather than `agent` because three layers need it — the
    API request model, `Settings`, and the runtime — and only the contract layer
    can be imported by all three without a cycle.

    Two kinds of dimension, deliberately distinguished by `BudgetTracker`:
    `max_steps` / `max_tool_calls` / `max_tokens` / `max_cost_usd` /
    `max_wall_clock_seconds` are *terminal* — hitting one ends the turn.
    `max_critique_rounds` is a *capability gate*: spending it stops further
    critique, it does not stop the agent.
    """

    model_config = ConfigDict(frozen=True)

    max_steps: int = 6
    max_tool_calls: int = 10
    max_critique_rounds: int = 2
    max_tokens: int = 20_000
    max_cost_usd: float = 0.10
    max_wall_clock_seconds: float = 30.0

    def clamp(self, ceiling: Budget) -> Budget:
        """Never raise a caller budget above the server ceiling."""
        return Budget(
            max_steps=min(self.max_steps, ceiling.max_steps),
            max_tool_calls=min(self.max_tool_calls, ceiling.max_tool_calls),
            max_critique_rounds=min(self.max_critique_rounds, ceiling.max_critique_rounds),
            max_tokens=min(self.max_tokens, ceiling.max_tokens),
            max_cost_usd=min(self.max_cost_usd, ceiling.max_cost_usd),
            max_wall_clock_seconds=min(self.max_wall_clock_seconds, ceiling.max_wall_clock_seconds),
        )


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
    # Why the agent stepped aside, on a response that is not itself a refusal.
    # Distinct from `refusal_reason`, which only ever accompanies refused=True.
    degraded_reason: str | None = None
    # The critic's last judgement. Surfaced so a caller can see that an answer
    # was returned without critic approval instead of having to infer it.
    critic_verdict: Verdict | None = None
    # Trace is omitted from the public response unless explicitly requested.
    trace: AgentTrace | None = None
