from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from rag.api.deps import AppState, get_principal, get_state
from rag.schemas import Answer, Principal

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000, description="Natural-language question")


class QueryResponse(BaseModel):
    answer: Answer


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question over the indexed corpus",
    description=(
        "Runs the online pipeline: rewrite, hybrid retrieve, rerank, generate, "
        "cite. Returns `refused=true` with no citations when the corpus does not "
        "support an answer. Only chunks the caller's principal may access are "
        "ever retrieved."
    ),
)
def query(
    body: QueryRequest,
    state: Annotated[AppState, Depends(get_state)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> QueryResponse:
    answer = state.pipeline.answer(body.query, principal=principal)
    return QueryResponse(answer=answer)
