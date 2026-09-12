from __future__ import annotations

from fastapi import APIRouter

from rag.api.deps import get_state

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    state = get_state()
    return {
        "status": "ok",
        "index_version": state.settings.index_version,
        "collection_alias": state.settings.collection_alias,
    }
