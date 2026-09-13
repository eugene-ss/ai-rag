"""Budget semantics, wall-clock enforcement, and bounded prompt growth.

Each test here pins a defect that shipped once and would otherwise reappear
silently, because the symptom in every case is a degraded answer rather than a
crash: the agent still returns something, it is just wrong, slow or expensive.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from rag.agent import AgentRuntime, Budget, Critic, RetrievalTool, ToolRegistry
from rag.agent.budget import BudgetTracker
from rag.agent.evidence import EvidenceLedger, render_sources
from rag.backends import BackendContext
from rag.llm.chat import ChatCompletion, Message, ToolSpec
from rag.llm.echo_chat import EchoChatLLM, ScriptedTurn
from rag.pipelines import offline as offline_mod
from rag.pipelines import online as online_mod
from rag.schemas import AclTags, Principal
from rag.schemas.agent import StopReason, ToolCall
from rag.schemas.answer import Usage
from rag.schemas.source import Source, SourceKind
from rag.settings import Settings

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
TENANT = "acme"
GROUP = "engineering"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        auth_trust_headers=True,
        cache_enabled=False,
        semantic_cache_enabled=False,
        index_registry_file=tmp_path / "index_versions.yaml",
        agent_enabled=True,
        agent_allow_egress=False,
        agent_include_trace=True,
    )


@pytest.fixture
def principal() -> Principal:
    return Principal(subject="alice", tenant=TENANT, groups=frozenset({GROUP}))


@pytest.fixture
def pipeline(settings: Settings):
    ctx = BackendContext.from_settings(settings)
    acl = AclTags(tenant=TENANT, allow_groups=frozenset({GROUP}))
    offline_mod.from_context(ctx).run(CORPUS, index_version="v1", acl=acl)
    return online_mod.from_context(ctx)


def _chunk_id(pipeline, principal: Principal) -> str:
    result = pipeline.retrieve_only("hybrid retrieval", principal=principal)
    assert result.results
    return result.results[0].chunk.chunk_id


def _approving_critic() -> Critic:
    return Critic(
        EchoChatLLM(
            script=[
                ScriptedTurn(
                    content=(
                        '{"sufficient": true, "grounded": true, "reasons": ["cited"], '
                        '"missing": [], "confidence": 0.9}'
                    )
                )
            ]
        )
    )


# --- terminal dimensions vs capability gates --------------------------------


def test_spent_critique_budget_does_not_stop_the_turn() -> None:
    """`max_critique_rounds=0` must gate the critic, not brick the agent.

    When both kinds of budget were checked in one `exhausted()` call, a zero
    critique budget made the loop terminate before the planner was ever invoked,
    so every request fell back to the plain pipeline. Configuring "no critic"
    silently disabled the agent.
    """
    tracker = BudgetTracker(budget=Budget(max_steps=4, max_critique_rounds=0))

    assert tracker.terminal_stop() is None
    assert tracker.can_critique() is False

    tracker.charge_step()
    assert tracker.terminal_stop() is None


@pytest.mark.parametrize(
    ("budget", "charge"),
    [
        (Budget(max_steps=1), "step"),
        (Budget(max_tool_calls=1), "tool"),
        (Budget(max_tokens=256), "tokens"),
        (Budget(max_cost_usd=0.01), "cost"),
    ],
)
def test_each_terminal_dimension_stops_the_turn(budget: Budget, charge: str) -> None:
    tracker = BudgetTracker(budget=budget)
    assert tracker.terminal_stop() is None

    if charge == "step":
        tracker.charge_step()
    elif charge == "tool":
        tracker.charge_tool_calls(1)
    elif charge == "tokens":
        tracker.charge_usage(Usage(total_tokens=budget.max_tokens))
    else:
        tracker.charge_usage(Usage(cost_usd=budget.max_cost_usd))

    assert tracker.terminal_stop() is StopReason.BUDGET_EXHAUSTED


@pytest.mark.asyncio
async def test_agent_answers_with_zero_critique_budget(pipeline, principal: Principal) -> None:
    """End-to-end counterpart: a criticless agent still produces a cited answer."""
    chunk_id = _chunk_id(pipeline, principal)
    runtime = AgentRuntime(
        chat_llm=EchoChatLLM(
            script=[
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(id="t1", name="retrieval_search", arguments={"query": "hybrid"})
                    ]
                ),
                ScriptedTurn(content=f"Dense and lexical search are fused. [{chunk_id}]"),
            ]
        ),
        tools=ToolRegistry([RetrievalTool(pipeline)]),
        pipeline=pipeline,
        critic=_approving_critic(),
        budget=Budget(max_steps=5, max_tool_calls=5, max_critique_rounds=0),
    )

    answer = await runtime.run("What is hybrid retrieval?", principal=principal)

    assert answer.fallback_used is False
    assert answer.refused is False
    assert answer.citations
    assert answer.stop_reason is StopReason.CRITIC_EXHAUSTED
    # The critic was never consulted, so there is no verdict to surface.
    assert answer.critic_verdict is None
    assert answer.self_corrections == 0


# --- wall clock -------------------------------------------------------------


class _SlowChatLLM:
    """A planner that hangs, standing in for a wedged upstream provider."""

    model_name = "slow"

    def __init__(self, delay: float = 30.0) -> None:
        self.delay = delay
        self.calls = 0

    async def chat(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> ChatCompletion:
        self.calls += 1
        await asyncio.sleep(self.delay)
        raise AssertionError("the deadline should have cancelled this call")


@pytest.mark.asyncio
async def test_wall_clock_deadline_cancels_a_hung_planner(pipeline, principal: Principal) -> None:
    """The budget is sampled between iterations, so a single slow call needs a deadline.

    Without the outer `asyncio.timeout`, one wedged provider call overruns
    `max_wall_clock_seconds` indefinitely while holding a request slot and a
    per-tenant semaphore — a self-inflicted denial of service.
    """
    llm = _SlowChatLLM(delay=30.0)
    runtime = AgentRuntime(
        chat_llm=llm,
        tools=ToolRegistry([RetrievalTool(pipeline)]),
        pipeline=pipeline,
        critic=_approving_critic(),
        budget=Budget(max_steps=5, max_wall_clock_seconds=0.2),
    )

    started = time.perf_counter()
    answer = await runtime.run("slow question", principal=principal)
    elapsed = time.perf_counter() - started

    assert llm.calls == 1
    # Generously bounded: the point is "roughly the deadline", not 30 seconds.
    assert elapsed < 5.0
    assert answer.stop_reason is StopReason.FALLBACK_PIPELINE
    assert answer.degraded_reason == "agent_budget_exhausted"


def test_remaining_seconds_never_goes_negative() -> None:
    tracker = BudgetTracker(budget=Budget(max_wall_clock_seconds=0.0))
    assert tracker.remaining_seconds() == 0.0
    assert tracker.terminal_stop() is StopReason.BUDGET_EXHAUSTED


# --- evidence ledger --------------------------------------------------------


def _source(ref: str, *, score: float, quote: str = "passage text") -> Source:
    return Source(
        ref=ref,
        kind=SourceKind.CHUNK,
        doc_id="d1",
        quote=quote,
        score=score,
    )


def test_ledger_deduplicates_repeated_tool_results() -> None:
    """Repeating the same tool call must not grow the prompt.

    The runtime used to append every result to the history *and* re-render the
    accumulated sources into a new user message each iteration, so token cost
    grew quadratically in step count on a loop that learned nothing.
    """
    ledger = EvidenceLedger()

    first = ledger.add([_source("a", score=0.9), _source("b", score=0.5)])
    repeat = ledger.add([_source("a", score=0.9), _source("b", score=0.5)])

    assert [s.ref for s in first] == ["a", "b"]
    assert repeat == [], "a repeated tool call adds no new evidence"
    assert len(ledger) == 2


def test_ledger_keeps_the_strongest_observation_of_a_ref() -> None:
    ledger = EvidenceLedger()
    ledger.add([_source("a", score=0.4, quote="weak")])
    ledger.add([_source("a", score=0.9, quote="strong")])
    ledger.add([_source("a", score=0.6, quote="middling")])

    assert len(ledger) == 1
    assert ledger.sources()[0].quote == "strong"


def test_ledger_orders_sources_strongest_first() -> None:
    ledger = EvidenceLedger()
    ledger.add([_source("low", score=0.1), _source("high", score=0.9), _source("mid", score=0.5)])

    assert [s.ref for s in ledger.sources()] == ["high", "mid", "low"]


def test_render_stays_within_the_char_budget_and_says_what_it_dropped() -> None:
    """A wide top_k must not push the planner prompt past the context window."""
    sources = [_source(f"r{i}", score=1.0 - i / 100, quote="x" * 400) for i in range(50)]

    rendered = render_sources(sources, max_chars=1500)

    assert len(rendered) <= 1600, "budget is a bound, not a suggestion"
    # Truncation drops the weakest evidence, and says so rather than silently
    # presenting a partial corpus as the whole of it.
    assert "r0" in rendered
    assert "r49" not in rendered
    assert "omitted to stay within the prompt budget" in rendered


def test_render_handles_the_empty_case() -> None:
    assert render_sources([]) == "(none)"


@pytest.mark.asyncio
async def test_prompt_does_not_regrow_when_the_agent_repeats_a_tool_call(
    pipeline, principal: Principal
) -> None:
    """Prompt growth must be flat across identical tool calls, not linear."""
    chunk_id = _chunk_id(pipeline, principal)
    sizes: list[int] = []

    class _MeasuringLLM:
        model_name = "measuring"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(
            self, messages: list[Message], *, tools: list[ToolSpec] | None = None
        ) -> ChatCompletion:
            sizes.append(sum(len(m.content or "") for m in messages))
            self.calls += 1
            # Three identical retrievals, then a cited draft.
            if self.calls <= 3:
                return ChatCompletion(
                    message=Message(role="assistant", content=""),
                    tool_calls=[
                        ToolCall(
                            id=f"t{self.calls}",
                            name="retrieval_search",
                            arguments={"query": "hybrid"},
                        )
                    ],
                    usage=Usage(model="measuring"),
                )
            return ChatCompletion(
                message=Message(role="assistant", content=f"Fused retrieval. [{chunk_id}]"),
                usage=Usage(model="measuring"),
            )

    runtime = AgentRuntime(
        chat_llm=_MeasuringLLM(),
        tools=ToolRegistry([RetrievalTool(pipeline)]),
        pipeline=pipeline,
        critic=_approving_critic(),
        budget=Budget(max_steps=6, max_tool_calls=6),
    )

    answer = await runtime.run("What is hybrid retrieval?", principal=principal)

    assert answer.citations
    assert len(sizes) >= 4
    # Each repeated call adds one tool message and nothing else. The old code
    # also re-rendered the full ledger every step, so growth compounded.
    per_step_growth = [sizes[i + 1] - sizes[i] for i in range(len(sizes) - 1)]
    assert all(growth > 0 for growth in per_step_growth[:3])
    assert max(per_step_growth) <= min(per_step_growth) * 2, (
        f"prompt growth must be flat across identical calls, got {per_step_growth}"
    )
