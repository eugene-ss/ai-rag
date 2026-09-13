"""What the registry offers the planner, and what it refuses to run.

The planner can only avoid a tool it was never offered. Advertising a tool that
cannot work and relying on its description to warn the model off costs a step
and a tool call per attempt — budget the turn never gets back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from rag.agent import AgentRuntime, Budget, Critic, RetrievalTool, ToolRegistry, WebSearchTool
from rag.agent.budget import BudgetTracker
from rag.agent.tools import GraphQueryTool
from rag.agent.tools.mixin import ToolMixin
from rag.backends import BackendContext, build_tool_registry
from rag.cache.memory import MemoryCache
from rag.llm.chat import ChatCompletion, Message, ToolSpec
from rag.llm.echo_chat import EchoChatLLM, ScriptedTurn
from rag.pipelines import offline as offline_mod
from rag.pipelines import online as online_mod
from rag.schemas import AclTags, Principal
from rag.schemas.agent import ToolCall, ToolResult
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


class _DownTool(ToolMixin):
    """A tool whose backend is unreachable — available today, not tomorrow."""

    name = "flaky_search"
    description = "Search a backend that may be down."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.runs = 0

    @property
    def available(self) -> bool:
        return self.healthy

    async def run(
        self,
        arguments: dict[str, Any],
        *,
        principal: Principal,
        budget: BudgetTracker,
    ) -> ToolResult:
        _ = arguments, principal, budget
        self.runs += 1
        return ToolResult(call_id="", name=self.name, ok=True, content="ok")


# --- what gets advertised ---------------------------------------------------


def test_unimplemented_tools_are_never_advertised(pipeline, principal: Principal) -> None:
    """Regression: graph_query used to appear in every production planner prompt.

    It sets `requires_egress = False`, so the egress gate did not hide it, and
    the only deterrent was the phrase "Currently unavailable" in its English
    description — which a model is free to ignore at the cost of a step.
    """
    registry = ToolRegistry([RetrievalTool(pipeline), WebSearchTool(), GraphQueryTool()])

    offered = {spec.name for spec in registry.specs(principal=principal, allow_egress=False)}
    assert offered == {"retrieval_search"}

    # An egress grant does not resurrect an unimplemented tool.
    with_egress = {spec.name for spec in registry.specs(principal=principal, allow_egress=True)}
    assert with_egress == {"retrieval_search"}


def test_default_production_registry_offers_only_working_tools(
    settings: Settings, pipeline, principal: Principal
) -> None:
    """The wiring the API actually uses, not a hand-built registry."""
    registry = build_tool_registry(settings, pipeline, cache=MemoryCache())

    # All three are registered — the stubs keep the surface honest for later.
    assert registry.get("web_search") is not None
    assert registry.get("graph_query") is not None
    # Only the one that works is offered.
    assert [s.name for s in registry.specs(principal=principal, allow_egress=False)] == [
        "retrieval_search"
    ]


def test_a_tool_whose_backend_goes_down_stops_being_offered(principal: Principal) -> None:
    """Availability is a runtime state, which is why a description cannot carry it."""
    tool = _DownTool(healthy=True)
    registry = ToolRegistry([tool])

    assert [s.name for s in registry.specs(principal=principal)] == ["flaky_search"]

    tool.healthy = False
    assert registry.specs(principal=principal) == []


# --- what gets executed -----------------------------------------------------


@pytest.mark.asyncio
async def test_naming_an_unadvertised_tool_costs_nothing_to_run(principal: Principal) -> None:
    """The gate is not only in the schema: execute() refuses too.

    A model can name a tool it was never shown, and a backend can fall over
    between `specs()` and the call.
    """
    tool = _DownTool(healthy=False)
    registry = ToolRegistry([tool])
    tracker = BudgetTracker(budget=Budget())

    result = await registry.execute(
        ToolCall(id="c1", name="flaky_search", arguments={"query": "x"}),
        principal=principal,
        budget=tracker,
    )

    assert result.ok is False
    assert result.error == "tool_unavailable:flaky_search"
    assert tool.runs == 0, "an unavailable tool must not be executed"


@pytest.mark.asyncio
async def test_egress_denial_outranks_unavailability(principal: Principal) -> None:
    """`web_search` is both egress-gated and unimplemented; the security gate reports first.

    Deliberate ordering: a caller without an egress grant learns only that
    egress was denied, not whether the tool behind it exists.
    """
    registry = ToolRegistry([WebSearchTool()])

    result = await registry.execute(
        ToolCall(id="c1", name="web_search", arguments={"query": "x"}),
        principal=principal,
        budget=BudgetTracker(budget=Budget()),
        allow_egress=False,
    )

    assert result.error == "egress_denied"


@pytest.mark.asyncio
async def test_a_turn_spends_no_budget_on_an_unavailable_tool(
    pipeline, principal: Principal
) -> None:
    """End to end: the planner cannot burn steps discovering a tool does not work.

    Before the availability gate, two speculative `graph_query` calls cost a
    third of a `max_steps=6` turn, and failed results are deliberately not
    cached, so the next request repeated the mistake.
    """
    chunk_id = pipeline.retrieve_only("hybrid retrieval", principal=principal).results[0]
    ref = chunk_id.chunk.chunk_id
    offered: list[list[str]] = []

    class _RecordingLLM:
        model_name = "recording"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(
            self, messages: list[Message], *, tools: list[ToolSpec] | None = None
        ) -> ChatCompletion:
            offered.append(sorted(t.name for t in tools or []))
            self.calls += 1
            return ChatCompletion(
                message=Message(role="assistant", content=f"Fused retrieval. [{ref}]"),
            )

    runtime = AgentRuntime(
        chat_llm=_RecordingLLM(),
        tools=ToolRegistry([RetrievalTool(pipeline), WebSearchTool(), GraphQueryTool()]),
        pipeline=pipeline,
        critic=Critic(
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
        ),
        budget=Budget(max_steps=6, max_tool_calls=6),
    )

    answer = await runtime.run("What is hybrid retrieval?", principal=principal)

    assert offered, "the planner must have been invoked"
    for names in offered:
        assert names == ["retrieval_search"]
    assert answer.tool_calls == 0
    assert answer.citations
