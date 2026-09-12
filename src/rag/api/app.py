from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.api.deps import init_state
from rag.api.errors import RagError, rag_error_handler
from rag.api.routes import health, query
from rag.observability.logging import configure_logging
from rag.settings import get_settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_state(settings=settings)
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="Production RAG API",
        description="Online query path only. Indexing lives in jobs/.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_exception_handler(RagError, rag_error_handler)  # type: ignore[arg-type]
    application.include_router(health.router)
    application.include_router(query.router)
    return application


app = create_app()
