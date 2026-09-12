# syntax=docker/dockerfile:1

# ---- builder ----------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layer: cached until pyproject/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 10001 rag
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder --chown=rag:rag /app/.venv /app/.venv
COPY --from=builder --chown=rag:rag /app/src /app/src
COPY --from=builder --chown=rag:rag /app/configs /app/configs
COPY --chown=rag:rag main.py ./

USER rag
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Serves the online query path only. Indexing runs via the `worker` service.
CMD ["uvicorn", "rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
