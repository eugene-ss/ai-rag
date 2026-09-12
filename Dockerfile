# syntax=docker/dockerfile:1

# ---- builder ----------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Real backend clients: the compose stack runs Qdrant, OpenSearch, and Redis.
# Drop extras you do not deploy to keep the image smaller.
ARG EXTRAS="--extra qdrant --extra opensearch --extra redis --extra openai"

# Dependency layer: cached until pyproject/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev ${EXTRAS}

# Project layer. README.md is required because pyproject declares it as readme.
COPY README.md ./
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev ${EXTRAS}

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

# curl is used by the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 rag

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    RAG_CONFIGS_ROOT=/app/configs \
    # configs/ is read-only in the container, so keep the registry writable.
    RAG_INDEX_REGISTRY_FILE=/tmp/index_versions.yaml

COPY --from=builder --chown=rag:rag /app/.venv /app/.venv
COPY --from=builder --chown=rag:rag /app/src /app/src
COPY --from=builder --chown=rag:rag /app/configs /app/configs
COPY --chown=rag:rag main.py ./

USER rag
EXPOSE 8000

# Liveness only. Readiness (index resolved, cache reachable) is /ready.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Serves the online query path only. Indexing runs via the `worker` service.
CMD ["uvicorn", "rag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
