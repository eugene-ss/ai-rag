from __future__ import annotations

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from rag.api.deps import get_state, resolve_principal
from rag.schemas import Answer

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class QueryResponse(BaseModel):
    answer: Answer


@router.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    authorization: str | None = Header(default=None),
    x_tenant: str | None = Header(default=None),
    x_groups: str | None = Header(default=None),
    x_subject: str | None = Header(default=None),
) -> QueryResponse:
    state = get_state()
    principal = resolve_principal(
        authorization=authorization,
        x_tenant=x_tenant,
        x_groups=x_groups,
        x_subject=x_subject,
    )
    answer = state.pipeline.answer(body.query, principal=principal)
    return QueryResponse(answer=answer)
