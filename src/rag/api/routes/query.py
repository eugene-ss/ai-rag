from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, SerializeAsAny

from rag.agent.budget import Budget
from rag.api.deps import AppState, get_principal, get_state
from rag.cache.keys import agent_cache_params, cache_key
from rag.observability.logging import get_logger
from rag.query.route import QueryMode, should_use_agent
from rag.schemas import Answer, Principal
from rag.schemas.agent import AgentAnswer

router = APIRouter(tags=["query"])
log = get_logger("api.query")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000, description="Natural-language question")
    mode: QueryMode = Field(
        default="auto",
        description=(
            "auto: elevate COMPLEX routes to the agent; "
            "fast: always OnlinePipeline; "
            "agent: force the bounded agent when enabled"
        ),
    )
    max_steps: int | None = Field(default=None, ge=1, le=32)
    max_tool_calls: int | None = Field(default=None, ge=1, le=64)
    max_critique_rounds: int | None = Field(default=None, ge=0, le=8)
    max_tokens: int | None = Field(default=None, ge=256, le=200_000)
    max_cost_usd: float | None = Field(default=None, ge=0.0, le=10.0)
    max_wall_clock_seconds: float | None = Field(default=None, ge=1.0, le=300.0)

    def budget_override(self, ceiling: Budget) -> Budget:
        """Caller budgets are clamped by the server ceiling — never raised."""
        requested = Budget(
            max_steps=self.max_steps if self.max_steps is not None else ceiling.max_steps,
            max_tool_calls=(
                self.max_tool_calls if self.max_tool_calls is not None else ceiling.max_tool_calls
            ),
            max_critique_rounds=(
                self.max_critique_rounds
                if self.max_critique_rounds is not None
                else ceiling.max_critique_rounds
            ),
            max_tokens=self.max_tokens if self.max_tokens is not None else ceiling.max_tokens,
            max_cost_usd=(
                self.max_cost_usd if self.max_cost_usd is not None else ceiling.max_cost_usd
            ),
            max_wall_clock_seconds=(
                self.max_wall_clock_seconds
                if self.max_wall_clock_seconds is not None
                else ceiling.max_wall_clock_seconds
            ),
        )
        return requested.clamp(ceiling)


class QueryResponse(BaseModel):
    # SerializeAsAny keeps AgentAnswer subclass fields in the JSON response.
    answer: SerializeAsAny[Answer]


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question over the indexed corpus",
    description=(
        "Runs the online pipeline or, for COMPLEX / mode=agent requests, the "
        "bounded agent runtime. Returns `refused=true` with no citations when "
        "the corpus does not support an answer. Only chunks the caller's "
        "principal may access are ever retrieved."
    ),
)
async def query(
    body: QueryRequest,
    state: Annotated[AppState, Depends(get_state)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> QueryResponse:
    use_agent = should_use_agent(
        body.query,
        mode=body.mode,
        agent_enabled=state.settings.agent_enabled and state.agent is not None,
    )
    if use_agent and state.agent is not None:
        answer = await _run_agent(body, state=state, principal=principal)
    else:
        answer = await asyncio.to_thread(state.pipeline.answer, body.query, principal=principal)
    return QueryResponse(answer=answer)


async def _run_agent(
    body: QueryRequest,
    *,
    state: AppState,
    principal: Principal,
) -> Answer:
    assert state.agent is not None
    ceiling = state.settings.agent_budget()
    budget = body.budget_override(ceiling)
    tool_names = [t.name for t in state.agent.tools.allowed_for(
        principal, allow_egress=state.agent.allow_egress
    )]
    params = agent_cache_params(
        mode="agent",
        allow_egress=state.agent.allow_egress,
        tool_names=tool_names,
        budget={
            "max_steps": budget.max_steps,
            "max_tool_calls": budget.max_tool_calls,
            "max_critique_rounds": budget.max_critique_rounds,
            "max_tokens": budget.max_tokens,
            "max_cost_usd": budget.max_cost_usd,
            "max_wall_clock_seconds": budget.max_wall_clock_seconds,
        },
    )
    # Merge retrieval params so agent and pipeline caches never collide.
    retrieval = state.settings.retrieval_config().cache_params()
    scoped_params: dict[str, Any] = {**retrieval, "agent": params}
    index_version = state.pipeline.active_index_version()
    key = cache_key(
        query=body.query,
        principal=principal,
        index_version=index_version,
        prompt_version=state.agent.prompt_version,
        retrieval_params=scoped_params,
    )

    if state.settings.cache_enabled:
        raw = state.pipeline.cache.get(key)
        if raw is not None:
            try:
                cached = AgentAnswer.model_validate_json(raw)
                return cached.model_copy(update={"cached": True})
            except Exception:
                try:
                    return Answer.model_validate_json(raw).model_copy(update={"cached": True})
                except Exception:
                    log.warning("discarding corrupt agent cache entry")

    async with state.agent_semaphore_for(principal.tenant):
        answer = await state.agent.run(body.query, principal=principal, budget=budget)

    if (
        state.settings.cache_enabled
        and not answer.refused
        and isinstance(answer, AgentAnswer)
        and answer.stop_reason.value == "answered"
    ):
        state.pipeline.cache.set(
            key,
            answer.model_dump_json(),
            ttl_seconds=state.settings.cache_ttl_seconds,
        )
    return answer
