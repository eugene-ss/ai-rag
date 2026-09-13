"""Tool-result cache scoping, and the refusal/degradation distinction.

Both are correctness-of-contract issues rather than crashes: the wrong cache key
serves stale or cross-tenant evidence, and a misplaced `refusal_reason` teaches
callers to treat a perfectly good answer as a refusal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.agent import AgentRuntime, Budget, Critic, RetrievalTool, ToolRegistry
from rag.agent.tools.cache import ToolResultCache, tool_result_cache_key
from rag.backends import BackendContext
from rag.cache.memory import MemoryCache
from rag.llm.echo_chat import EchoChatLLM, ScriptedTurn
from rag.pipelines import offline as offline_mod
from rag.pipelines import online as online_mod
from rag.schemas import AclTags, Principal
from rag.schemas.agent import StopReason, ToolCall, ToolResult

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
TENANT = "acme"
GROUP = "engineering"

ARGS = {"query": "hybrid retrieval", "top_k": 5}


@pytest.fixture
def principal() -> Principal:
    return Principal(subject="alice", tenant=TENANT, groups=frozenset({GROUP}))


def _key(**overrides: object) -> str:
    base: dict[str, object] = {
        "tool": "retrieval_search",
        "arguments": ARGS,
        "principal": Principal(subject="alice", tenant=TENANT, groups=frozenset({GROUP})),
        "index_version": "v1",
    }
    return tool_result_cache_key(**{**base, **overrides})  # type: ignore[arg-type]


def test_cache_key_is_stable_for_the_same_request() -> None:
    assert _key() == _key()
    # Argument order must not matter; the payload is canonicalised.
    assert _key(arguments={"top_k": 5, "query": "hybrid retrieval"}) == _key()


def test_cache_key_separates_index_versions() -> None:
    """Retrieval results are only valid for the corpus that produced them.

    Without the version in the key, promoting v2 keeps serving v1 passages for
    the whole cache TTL — a reindex that appears to have had no effect.
    """
    assert _key(index_version="v1") != _key(index_version="v2")


def test_cache_key_separates_tenants_and_groups() -> None:
    other_tenant = Principal(subject="alice", tenant="other", groups=frozenset({GROUP}))
    other_groups = Principal(subject="alice", tenant=TENANT, groups=frozenset({"finance"}))

    assert _key(principal=other_tenant) != _key()
    assert _key(principal=other_groups) != _key()


def test_cache_key_ignores_the_subject_within_one_acl_scope() -> None:
    """Two principals with identical reach share a cache entry.

    Keying on ACL scope rather than identity is what makes the cache worth
    having: what a caller may see is fully determined by tenant plus groups, so
    per-subject keys fragment the cache without adding any isolation.
    """
    bob = Principal(subject="bob", tenant=TENANT, groups=frozenset({GROUP}))
    assert _key(principal=bob) == _key()


def test_cache_key_separates_tools_and_arguments() -> None:
    assert _key(tool="web_search") != _key()
    assert _key(arguments={"query": "something else", "top_k": 5}) != _key()


def _result(content: str = "passage") -> ToolResult:
    return ToolResult(call_id="c1", name="retrieval_search", ok=True, content=content)


def test_cached_results_do_not_cross_index_versions(principal: Principal) -> None:
    cache = ToolResultCache(MemoryCache())
    cache.set(
        tool="retrieval_search",
        arguments=ARGS,
        principal=principal,
        result=_result("v1 passage"),
        index_version="v1",
    )

    hit = cache.get(
        tool="retrieval_search", arguments=ARGS, principal=principal, index_version="v1"
    )
    miss = cache.get(
        tool="retrieval_search", arguments=ARGS, principal=principal, index_version="v2"
    )

    assert hit is not None
    assert hit.content == "v1 passage"
    assert miss is None


def test_failed_results_are_not_cached(principal: Principal) -> None:
    """Caching a failure turns a transient outage into a sticky one for the TTL."""
    cache = ToolResultCache(MemoryCache())
    cache.set(
        tool="retrieval_search",
        arguments=ARGS,
        principal=principal,
        result=ToolResult(call_id="c1", name="retrieval_search", ok=False, error="timeout"),
        index_version="v1",
    )

    assert (
        cache.get(tool="retrieval_search", arguments=ARGS, principal=principal, index_version="v1")
        is None
    )


def test_cached_results_drop_the_call_id(principal: Principal) -> None:
    """`call_id` binds a result to one planner call; the registry rebinds it."""
    cache = ToolResultCache(MemoryCache())
    cache.set(
        tool="retrieval_search",
        arguments=ARGS,
        principal=principal,
        result=_result(),
        index_version="v1",
    )

    hit = cache.get(
        tool="retrieval_search", arguments=ARGS, principal=principal, index_version="v1"
    )
    assert hit is not None
    assert hit.call_id == ""


def test_disabled_cache_never_stores_or_serves(principal: Principal) -> None:
    backing = MemoryCache()
    cache = ToolResultCache(backing, enabled=False)
    cache.set(
        tool="retrieval_search",
        arguments=ARGS,
        principal=principal,
        result=_result(),
        index_version="v1",
    )

    assert (
        cache.get(tool="retrieval_search", arguments=ARGS, principal=principal, index_version="v1")
        is None
    )
    # And nothing was written behind its back.
    assert backing.get(_key()) is None


# --- refusal vs degradation -------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_answer_reports_degradation_not_refusal(tmp_path: Path) -> None:
    """A successful fallback is an answer, so `refusal_reason` must stay empty.

    The agent used to stamp its own reason for stepping aside into
    `refusal_reason`, which is the field that means "this response is a refusal".
    Clients keying off it hid a perfectly good pipeline answer from the user.
    """
    from rag.settings import Settings

    settings = Settings(
        env="test",
        auth_trust_headers=True,
        cache_enabled=False,
        semantic_cache_enabled=False,
        index_registry_file=tmp_path / "index_versions.yaml",
        agent_enabled=True,
        agent_allow_egress=False,
    )
    ctx = BackendContext.from_settings(settings)
    acl = AclTags(tenant=TENANT, allow_groups=frozenset({GROUP}))
    offline_mod.from_context(ctx).run(CORPUS, index_version="v1", acl=acl)
    pipeline = online_mod.from_context(ctx)
    principal = Principal(subject="alice", tenant=TENANT, groups=frozenset({GROUP}))

    runtime = AgentRuntime(
        chat_llm=EchoChatLLM(
            script=[
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(id="t1", name="retrieval_search", arguments={"query": "hybrid"})
                    ]
                ),
                # Never cites, so the agent can never ground its own answer.
                ScriptedTurn(content="An answer with no citation markers at all."),
            ]
        ),
        tools=ToolRegistry([RetrievalTool(pipeline)]),
        pipeline=pipeline,
        critic=Critic(
            EchoChatLLM(
                script=[
                    ScriptedTurn(
                        content=(
                            '{"sufficient": false, "grounded": false, "reasons": ["uncited"], '
                            '"missing": ["citations"], "confidence": 0.1}'
                        )
                    )
                ]
            )
        ),
        budget=Budget(max_steps=3, max_tool_calls=3, max_critique_rounds=1),
    )

    answer = await runtime.run("What is hybrid retrieval?", principal=principal)

    assert answer.fallback_used is True
    assert answer.stop_reason is StopReason.FALLBACK_PIPELINE
    # The two fields carry different facts and must not be conflated.
    assert answer.degraded_reason == "agent_budget_exhausted"
    if answer.refused:
        assert answer.refusal_reason is not None
    else:
        assert answer.refusal_reason is None
        assert answer.citations, "a non-refusal answer must be grounded"
