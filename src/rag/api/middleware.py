from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.status import HTTP_504_GATEWAY_TIMEOUT

from rag.observability.logging import get_logger
from rag.observability.metrics import METRICS
from rag.observability.tracing import current_trace_id, reset_trace

log = get_logger("api.access")

TRACE_HEADER = "X-Request-Id"


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Bound every request by `request_timeout_seconds`.

    The agent budget and the LLM retry policy each bound a *part* of a request,
    and multiplying them out exceeds any latency objective worth publishing. This
    is the one place that bounds the whole thing, so a stuck dependency releases
    the connection instead of holding a worker.
    """

    def __init__(self, app: Callable[..., Awaitable[None]], *, timeout_seconds: float) -> None:
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if self.timeout_seconds <= 0:
            return await call_next(request)
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except (TimeoutError, asyncio.CancelledError):
            METRICS.incr("http_timeouts", path=request.url.path)
            log.warning(
                "%s %s -> timeout after %.1fs trace=%s",
                request.method,
                request.url.path,
                self.timeout_seconds,
                current_trace_id(),
            )
            return JSONResponse(
                status_code=HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "error": "request_timeout",
                    "detail": f"request exceeded {self.timeout_seconds:.0f}s",
                    "trace_id": current_trace_id(),
                },
            )


class TraceMiddleware(BaseHTTPMiddleware):
    """Assign a trace id per request, echo it, and log one access line.

    Accepts an inbound X-Request-Id so traces stitch together across services.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        trace_id = reset_trace(request.headers.get(TRACE_HEADER))
        request.state.trace_id = trace_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            METRICS.observe("http_latency_ms", elapsed, path=request.url.path)
            log.warning(
                "%s %s -> exception in %.1fms trace=%s",
                request.method,
                request.url.path,
                elapsed,
                trace_id,
            )
            raise

        elapsed = (time.perf_counter() - start) * 1000
        response.headers[TRACE_HEADER] = trace_id
        METRICS.observe("http_latency_ms", elapsed, path=request.url.path)
        METRICS.incr(
            "http_requests",
            path=request.url.path,
            status=str(response.status_code),
        )
        log.info(
            "%s %s -> %s in %.1fms trace=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            trace_id,
        )
        return response
