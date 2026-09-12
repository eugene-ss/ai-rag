# Production RAG skeleton

default: check

sync:
    uv sync --extra dev

test:
    uv run pytest -q

lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

typecheck:
    uv run mypy src/rag

check: lint typecheck test

serve:
    uv run uvicorn rag.api.app:app --reload --host 0.0.0.0 --port 8000

ingest:
    uv run rag ingest --source data/raw --index-version {{env_var("RAG_INDEX_VERSION", "v1")}}

eval:
    uv run rag eval --dataset tests/fixtures/golden.jsonl

reindex:
    uv run rag reindex --source data/raw --index-version {{env_var("RAG_INDEX_VERSION", "v1")}}

fmt:
    uv run ruff format src tests
    uv run ruff check --fix src tests
