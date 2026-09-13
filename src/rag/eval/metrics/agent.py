"""Agent-specific evaluation metrics.

CI gates on steps and cost, not just recall: a change that doubles average
steps is a real regression even when answer quality holds.
"""

from __future__ import annotations

from rag.schemas.agent import ANSWERABLE_STOPS, AgentAnswer


def task_success(answer: AgentAnswer, *, require_citations: bool = True) -> float:
    """1.0 when the agent answered without falling back or refusing."""
    if answer.refused:
        return 0.0
    if answer.fallback_used:
        return 0.0
    if answer.stop_reason not in ANSWERABLE_STOPS:
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
    """Harmonic mean of tool precision and recall, order- and count-insensitive.

    Recall alone (the fraction of expected tools that were called) is blind to
    over-calling: an agent that invokes every tool it has scores a perfect 1.0
    while burning the budget. Precision alone is blind to an agent that calls
    one correct tool and skips the rest. The gate needs both, so this is F1.

    An empty `expected` means the example does not pin tool choice; scoring it
    would penalise examples that simply did not specify.
    """
    if not expected:
        return 1.0
    predicted_set = set(predicted)
    if not predicted_set:
        return 0.0
    expected_set = set(expected)
    hits = len(predicted_set & expected_set)
    if hits == 0:
        return 0.0
    precision = hits / len(predicted_set)
    recall = hits / len(expected_set)
    return 2 * precision * recall / (precision + recall)


def self_correction_rate(self_corrections: int, steps_taken: int) -> float:
    if steps_taken <= 0:
        return 0.0
    return min(self_corrections / steps_taken, 1.0)


def cost_per_correct(cost_usd: float, success: float) -> float:
    """Cost attributed only to successful answers; failed answers contribute 0."""
    if success <= 0:
        return 0.0
    return cost_usd / success
