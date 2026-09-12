from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from rag.observability.logging import get_logger
from rag.observability.metrics import METRICS
from rag.observability.tracing import reset_trace

log = get_logger("api.access")

TRACE_HEADER = "X-Request-Id"


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
