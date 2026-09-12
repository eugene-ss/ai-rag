from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, status

from rag.api.deps import AppState, get_state
from rag.observability.metrics import METRICS

router = APIRouter(tags=["operations"])


@router.get("/health", summary="Liveness: process is up")
def health(state: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
    return {
        "status": "ok",
        "service": state.settings.service_name,
        "env": state.settings.env,
    }


@router.get("/ready", summary="Readiness: dependencies usable and an index is live")
def ready(state: Annotated[AppState, Depends(get_state)], response: Response) -> dict[str, Any]:
    """Readiness fails until an index is actually queryable.

    Distinct from liveness so a rolling deploy does not send traffic to a
    replica whose alias has not resolved yet.
    """
    report = state.pipeline.readiness()
    if not report["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if report["ready"] else "not_ready",
        "authenticator": state.authenticator.name,
        **report,
    }


@router.get("/metrics", summary="In-process counters and histograms")
def metrics(state: Annotated[AppState, Depends(get_state)], response: Response) -> dict[str, Any]:
    """Snapshot of process-local metrics.

    Intentionally not Prometheus text format: wire a real exporter for that.
    Disable with RAG_METRICS_ENABLED=false if the endpoint should not be public.
    """
    if not state.settings.metrics_enabled:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {"detail": "metrics disabled"}
    return METRICS.snapshot()
