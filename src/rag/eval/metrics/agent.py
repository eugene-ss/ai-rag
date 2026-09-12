"""Agent-specific evaluation metrics.

CI gates on steps and cost, not just recall: a change that doubles average
steps is a real regression even when answer quality holds.
"""

from __future__ import annotations

from rag.schemas.agent import AgentAnswer, StopReason


def task_success(answer: AgentAnswer, *, require_citations: bool = True) -> float:
    """1.0 when the agent answered without falling back or refusing."""
    if answer.refused:
        return 0.0
    if answer.fallback_used:
        return 0.0
    if answer.stop_reason not in {StopReason.ANSWERED, StopReason.CRITIC_GROUNDED}:
        return 0.0
    if require_citations and not answer.citations:
        return 0.0
    return 1.0


def step_efficiency(steps_taken: int, *, budget_steps: int) -> float:
    """1.0 when the turn used a single step; 0.0 when the full budget is spent."""
    if budget_steps <= 1:
        return 1.0 if steps_taken <= 1 else 0.0
    unused = max(budget_steps - steps_taken, 0)
    return unused / (budget_steps - 1)


def tool_selection_precision(predicted: list[str], expected: list[str]) -> float:
    """Fraction of expected tools that were actually called (order-insensitive)."""
    if not expected:
        return 1.0
    predicted_set = set(predicted)
    hits = sum(1 for name in expected if name in predicted_set)
    return hits / len(expected)


def self_correction_rate(self_corrections: int, steps_taken: int) -> float:
    if steps_taken <= 0:
        return 0.0
    return min(self_corrections / steps_taken, 1.0)


def cost_per_correct(cost_usd: float, success: float) -> float:
    """Cost attributed only to successful answers; failed answers contribute 0."""
    if success <= 0:
        return 0.0
    return cost_usd / success
