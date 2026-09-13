from __future__ import annotations

import pytest

from rag.eval.metrics import answer_relevance, faithfulness, mrr, ndcg_at_k, recall_at_k
from rag.eval.metrics.agent import (
    cost_per_correct,
    self_correction_rate,
    step_efficiency,
    task_success,
    tool_selection_precision,
)
from rag.schemas.agent import AgentAnswer, StopReason
from rag.schemas.answer import Citation


def test_recall_at_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = ["c", "e"]
    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert recall_at_k(retrieved, relevant, 3) == 0.5


def test_mrr() -> None:
    assert mrr(["a", "b", "c"], ["c"]) == 1 / 3
    assert mrr(["a", "b"], ["z"]) == 0.0


def test_ndcg_at_k() -> None:
    score = ndcg_at_k(["a", "b", "c"], ["a", "c"], 3)
    assert 0.0 < score <= 1.0


def test_faithfulness_and_relevance() -> None:
    answer = "hybrid retrieval uses dense and lexical search"
    ctx = ["Hybrid retrieval combines dense vector search with lexical BM25."]
    assert faithfulness(answer, ctx) > 0.3
    assert answer_relevance(answer, "what is hybrid retrieval") > 0.2


# --- agent metrics ----------------------------------------------------------


def _answer(**overrides: object) -> AgentAnswer:
    defaults: dict[str, object] = {
        "text": "Grounded. [r1]",
        "citations": [Citation(ref="r1", chunk_id="c1")],
        "stop_reason": StopReason.ANSWERED,
    }
    return AgentAnswer(**{**defaults, **overrides})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "stop_reason",
    [StopReason.ANSWERED, StopReason.CRITIC_GROUNDED, StopReason.CRITIC_EXHAUSTED],
)
def test_task_success_counts_every_answerable_stop(stop_reason: StopReason) -> None:
    """CRITIC_GROUNDED and CRITIC_EXHAUSTED are answers the caller actually got.

    Scoring them as failures would make the metric disagree with the response,
    and would silently zero out the CI gate the moment the runtime stops
    normalising its stop reasons.
    """
    assert task_success(_answer(stop_reason=stop_reason)) == 1.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"refused": True},
        {"fallback_used": True},
        {"stop_reason": StopReason.NO_GROUNDING},
        {"stop_reason": StopReason.FALLBACK_PIPELINE},
        {"citations": []},
    ],
)
def test_task_success_is_zero_for_degraded_turns(overrides: dict[str, object]) -> None:
    assert task_success(_answer(**overrides)) == 0.0


def test_task_success_can_ignore_citations() -> None:
    assert task_success(_answer(citations=[]), require_citations=False) == 1.0


def test_step_efficiency_scales_between_one_step_and_the_full_budget() -> None:
    assert step_efficiency(1, budget_steps=6) == 1.0
    assert step_efficiency(6, budget_steps=6) == 0.0
    assert step_efficiency(3, budget_steps=6) == pytest.approx(0.6)
    # Overspending cannot produce a negative score.
    assert step_efficiency(9, budget_steps=6) == 0.0
    # A single-step budget has no range to scale over.
    assert step_efficiency(1, budget_steps=1) == 1.0
    assert step_efficiency(2, budget_steps=1) == 0.0


def test_tool_selection_penalises_both_missing_and_extra_calls() -> None:
    assert tool_selection_precision(["retrieval_search"], ["retrieval_search"]) == 1.0
    # Regression: the old recall-only formula scored this 1.0, so an agent that
    # sprayed every tool it had looked perfect while burning the budget.
    sprayed = tool_selection_precision(
        ["retrieval_search", "web_search", "sql_query"], ["retrieval_search"]
    )
    assert sprayed == pytest.approx(0.5)
    # Half the expected tools called, nothing extra.
    assert tool_selection_precision(["a"], ["a", "b"]) == pytest.approx(2 / 3)
    assert tool_selection_precision(["wrong"], ["right"]) == 0.0
    assert tool_selection_precision([], ["right"]) == 0.0
    # Examples that do not pin tool choice must not be penalised.
    assert tool_selection_precision(["anything"], []) == 1.0
    # Repeated calls to the same tool are one selection decision.
    assert tool_selection_precision(["a", "a"], ["a"]) == 1.0


def test_self_correction_rate_is_bounded() -> None:
    assert self_correction_rate(0, 4) == 0.0
    assert self_correction_rate(1, 4) == 0.25
    assert self_correction_rate(9, 4) == 1.0
    assert self_correction_rate(1, 0) == 0.0


def test_cost_per_correct_charges_nothing_to_failed_answers() -> None:
    assert cost_per_correct(0.02, 1.0) == 0.02
    assert cost_per_correct(0.02, 0.0) == 0.0
