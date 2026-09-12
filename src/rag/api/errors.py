from __future__ import annotations

from fastapi import Request, status
from fastapi.responses import JSONResponse

from rag.observability.logging import get_logger
from rag.observability.metrics import METRICS
from rag.observability.tracing import current_trace_id

log = get_logger("api.errors")


class RagError(Exception):
    """Base class for errors that map to a client-visible HTTP response."""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "bad_request"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnauthorizedError(RagError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"


class BackendUnavailableError(RagError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "backend_unavailable"


def _payload(code: str, message: str, trace_id: str) -> dict[str, object]:
    return {"error": {"code": code, "message": message, "trace_id": trace_id}}


async def rag_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RagError)
    trace_id = getattr(request.state, "trace_id", "") or current_trace_id()
    METRICS.incr("api_errors", code=exc.code)
    headers = {"WWW-Authenticate": "Bearer"} if isinstance(exc, UnauthorizedError) else None
    return JSONResponse(
        status_code=exc.status_code,
        content=_payload(exc.code, exc.message, trace_id),
        headers=headers,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Never leak internals: log the detail, return an opaque 500."""
    trace_id = getattr(request.state, "trace_id", "") or current_trace_id()
    METRICS.incr("api_errors", code="internal")
    log.exception("unhandled error trace=%s path=%s", trace_id, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_payload("internal_error", "Internal server error", trace_id),
    )
