from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.deps import AppState, build_state
from rag.api.errors import RagError, rag_error_handler, unhandled_error_handler
from rag.api.middleware import TimeoutMiddleware, TraceMiddleware
from rag.api.routes import health, query
from rag.observability.logging import configure_logging, get_logger
from rag.settings import Settings, get_settings

log = get_logger("api")

DESCRIPTION = """
Online query path for the RAG system.

This service is read-only with respect to the index. Ingestion, chunking,
embedding, and indexing run in the offline `jobs/` process, never here.
"""


def create_app(
    *,
    settings: Settings | None = None,
    state: AppState | None = None,
) -> FastAPI:
    """Build the ASGI app.

    Injecting `state` lets tests supply a pre-built index without touching
    global process state.
    """
    resolved = settings or (state.settings if state else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved.log_level, resolved.log_dir, json_output=resolved.log_json)
        app.state.rag = state or build_state(settings=resolved)
        log.info(
            "api ready env=%s auth=%s vector=%s lexical=%s",
            resolved.env,
            app.state.rag.authenticator.name,
            resolved.vector_backend,
            resolved.lexical_backend,
        )
        try:
            yield
        finally:
            log.info("api shutting down")

    application = FastAPI(
        title="Production RAG API",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )

    # Added first so it runs innermost of the two: the trace middleware still
    # logs and labels the 504 it produces.
    application.add_middleware(
        TimeoutMiddleware,
        timeout_seconds=resolved.request_timeout_seconds,
    )
    application.add_middleware(TraceMiddleware)
    if resolved.cors_allow_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
        )

    application.add_exception_handler(RagError, rag_error_handler)
    application.add_exception_handler(Exception, unhandled_error_handler)

    application.include_router(health.router)
    application.include_router(query.router)
    return application


app = create_app()
