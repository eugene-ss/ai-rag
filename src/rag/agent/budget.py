"""Per-turn budgets that keep an unbounded agent loop safe in a request path.

An agent without a budget is a cost-amplification attack waiting to happen.
Every chargeable resource — steps, tool calls, tokens, dollars, wall clock —
is tracked; the first exhausted dimension stops the loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag.schemas.agent import StopReason
from rag.schemas.answer import Usage


@dataclass(frozen=True)
class Budget:
    """Hard caps for one agent turn. Caller-supplied values are clamped by the server."""

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
            max_wall_clock_seconds=min(
                self.max_wall_clock_seconds, ceiling.max_wall_clock_seconds
            ),
        )


@dataclass
class BudgetTracker:
    """Mutable spend counter for one agent turn."""

    budget: Budget
    steps: int = 0
    tool_calls: int = 0
    critique_rounds: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)
    usage_total: Usage = field(default_factory=Usage)

    def charge_step(self) -> None:
        self.steps += 1

    def charge_tool_calls(self, n: int = 1) -> None:
        self.tool_calls += n

    def charge_critique(self) -> None:
        self.critique_rounds += 1

    def charge_usage(self, usage: Usage) -> None:
        self.tokens += usage.total_tokens
        self.cost_usd += usage.cost_usd
        self.usage_total = self.usage_total + usage

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def exhausted(self) -> StopReason | None:
        """Return the stop reason for the first exhausted dimension, else None."""
        if self.steps >= self.budget.max_steps:
            return StopReason.BUDGET_EXHAUSTED
        if self.tool_calls >= self.budget.max_tool_calls:
            return StopReason.BUDGET_EXHAUSTED
        if self.critique_rounds >= self.budget.max_critique_rounds:
            return StopReason.CRITIC_EXHAUSTED
        if self.tokens >= self.budget.max_tokens:
            return StopReason.BUDGET_EXHAUSTED
        if self.cost_usd >= self.budget.max_cost_usd:
            return StopReason.BUDGET_EXHAUSTED
        if self.elapsed_seconds() >= self.budget.max_wall_clock_seconds:
            return StopReason.BUDGET_EXHAUSTED
        return None

    def remaining_tool_calls(self) -> int:
        return max(0, self.budget.max_tool_calls - self.tool_calls)

    def can_critique(self) -> bool:
        return self.critique_rounds < self.budget.max_critique_rounds
