# Production RAG — task runner. Run `just` to see everything.

default:
    @just --list

# --- setup -------------------------------------------------------------------

# Install runtime + dev dependencies.
sync:
    uv sync --extra dev

# Install everything, including real backend clients.
sync-all:
    uv sync --extra dev --extra all

# Create .env from the template if it does not exist yet.
env:
    @test -f .env || (cp .env.example .env && echo "created .env")

# --- quality gates -----------------------------------------------------------

test:
    uv run pytest

test-cov:
    uv run pytest --cov --cov-report=term-missing

lint:
    uv run ruff check .
    uv run ruff format --check .

typecheck:
    uv run mypy src/rag main.py

# Everything CI runs.
check: lint typecheck test

fmt:
    uv run ruff format .
    uv run ruff check --fix .

# --- online path -------------------------------------------------------------

# Serve the API with autoreload (dev).
serve:
    uv run python main.py serve --reload

# Serve without reload, closer to production.
serve-prod workers="2":
    uv run uvicorn rag.api.app:app --host 0.0.0.0 --port 8000 --workers {{workers}}

# --- offline jobs ------------------------------------------------------------

# Ingest and index a source directory.
ingest source="data/raw" version=env_var_or_default("RAG_INDEX_VERSION", "v1"):
    uv run rag ingest --source {{source}} --index-version {{version}}

# Build a new version without sending traffic to it.
build-index source="data/raw" version="v2":
    uv run rag reindex --source {{source}} --index-version {{version}} --no-activate

# Publish a built version (also used to roll back).
activate version:
    uv run rag activate --index-version {{version}}

reindex source="data/raw" version=env_var_or_default("RAG_INDEX_VERSION", "v1"):
    uv run rag reindex --source {{source}} --index-version {{version}}

versions:
    uv run rag versions

# Evaluate against the golden set, indexing the demo corpus first.
eval:
    uv run rag eval --dataset tests/fixtures/golden.jsonl --index-source tests/fixtures/corpus

# Same, but fail if retrieval quality regresses. Use this in CI.
eval-gate min_recall="0.8":
    uv run rag eval --dataset tests/fixtures/golden.jsonl \
        --index-source tests/fixtures/corpus --min-recall {{min_recall}}

# Hermetic agent gate: EchoChatLLM, fail on step/cost regressions.
eval-agent:
    uv run rag eval-agent \
        --dataset tests/fixtures/golden_agent.jsonl \
        --index-source tests/fixtures/corpus \
        --max-avg-steps 4 --max-avg-cost 0.05 --min-success-rate 1.0

# --- demo --------------------------------------------------------------------

# Zero-dependency end-to-end demo: index the sample corpus, then evaluate it.
demo:
    @just ingest tests/fixtures/corpus v1
    @just eval

# --- containers --------------------------------------------------------------

build:
    docker compose build

# Start the full stack: api + qdrant + opensearch + redis.
up:
    docker compose up -d --wait

down:
    docker compose down

# Remove volumes too: wipes all indexed data.
clean:
    docker compose down -v

logs service="api":
    docker compose logs -f {{service}}

# One-shot indexing job inside the stack (offline profile, never the API).
docker-reindex version="v1":
    docker compose run --rm -e RAG_INDEX_VERSION={{version}} worker

# Evaluate inside the stack against your curated set in data/golden/.
docker-eval dataset="/data/golden/golden.jsonl":
    docker compose run --rm worker python main.py eval --dataset {{dataset}}

# Verify the compose file and Dockerfile parse.
docker-validate:
    docker compose config --quiet
    docker build --target runtime -t ai-rag:validate .
