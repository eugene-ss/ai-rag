"""Per-turn budget accounting for the agent loop.

An agent without a budget is a cost-amplification attack waiting to happen.
Every chargeable resource — steps, tool calls, tokens, dollars, wall clock — is
tracked, and the first exhausted *terminal* dimension stops the loop.

The terminal/gate distinction matters. Steps, tool calls, tokens, cost and wall
clock are terminal: spending one means the turn must end. Critique rounds are
not — spending them means "stop asking the critic", and the agent must still be
allowed to finish. Collapsing the two is how `max_critique_rounds=0` came to
mean "never invoke the planner at all".

`Budget` itself lives in `rag.schemas.agent`, because `Settings` and the API
request model both need it and neither may import the agent package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag.schemas.agent import Budget, StopReason
from rag.schemas.answer import Usage

__all__ = ["Budget", "BudgetTracker"]


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

    def remaining_seconds(self) -> float:
        """Wall clock left in this turn. Never negative."""
        return max(0.0, self.budget.max_wall_clock_seconds - self.elapsed_seconds())

    def terminal_stop(self) -> StopReason | None:
        """Stop reason for the first exhausted *terminal* dimension, else None.

        Critique rounds are deliberately absent: they gate the critic, not the
        turn. See the module docstring.
        """
        if self.steps >= self.budget.max_steps:
            return StopReason.BUDGET_EXHAUSTED
        if self.tool_calls >= self.budget.max_tool_calls:
            return StopReason.BUDGET_EXHAUSTED
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
