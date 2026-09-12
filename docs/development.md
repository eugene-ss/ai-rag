# Development

## Setup

```bash
just sync        # uv sync --extra dev
just env         # create .env from the template
just check       # lint + typecheck + tests
just demo        # index the sample corpus and evaluate, no external services
```

Python 3.11+ and [uv](https://docs.astral.sh/uv/). `just` is a convenience; every
recipe is a plain command you can run directly.

## Layout

```
src/rag/
  schemas/       contracts: Document, Chunk, QueryResult, Answer, Principal
  settings.py    all configuration, one place
  backends.py    which implementation each Protocol gets
  ingestion/     source connectors            ─┐
  parsing/       mime-type dispatch            │ offline only
  chunking/      strategy per document type    │
  embedding/     vector providers             ─┘
  vectordb/      dense index adapters         ─┐
  lexical/       BM25 adapters                 │ shared state
  index_registry.py  version bookkeeping      ─┘
  query/         rewrite, route, multi-query, HyDE  ─┐
  retrieval/     hybrid + RRF + ACL pushdown         │ online only
  rerank/        identity, cross-encoder             │
  prompts/       versioned templates                 │
  generation/    grounded answers, citations, refusal│
  cache/         exact + semantic                   ─┘
  llm/           provider adapters + resilience
  security/      auth, ACLs, PII
  observability/ traces, metrics, cost, logging
  eval/          golden sets, metrics, report
  pipelines/     offline.py, online.py
  api/           FastAPI (online path only)
  jobs/          CLI, reindex, scheduled eval
```

Two properties to preserve:

1. **Every backend is a `typing.Protocol`** with at least one dependency-free
   implementation. That is what makes the test suite hermetic.
2. **`api/` must not be able to *reach* `ingestion`, `parsing`, `chunking`,
   `eval`, or `jobs`** — transitively, not just directly.
   `tests/test_architecture_boundaries.py` builds the import graph and fails the
   build otherwise. This is why `rag/pipelines/__init__.py` re-exports nothing:
   a convenience re-export there pulls every parser into the API process.

## Adding a backend

The pattern is the same for every extension point: implement the protocol, use a
lazy import for the third-party client, register it in `backends.py`, add a
settings value.

```python
# src/rag/vectordb/pgvector.py
class PgVectorStore:
    def __init__(self, *, dsn: str, collection: str, dimensions: int) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise MissingBackendError("PgVectorStore", "pgvector") from exc
        ...

    def search(self, vector, *, top_k, principal, index_version=None):
        # ACL must be part of the SQL WHERE clause, not a post-filter.
        ...
```

```python
# src/rag/backends.py
def build_vector_store(settings, *, index_version=None) -> VectorStore:
    if settings.vector_backend == "pgvector":
        from rag.vectordb.pgvector import PgVectorStore
        return PgVectorStore(dsn=settings.pg_dsn, ...)
```

Then add `"pgvector"` to the `VectorBackend` literal, the extra to
`pyproject.toml`, and the module to the mypy `ignore_missing_imports` overrides.

Non-negotiable for a retrieval backend: **ACL filtering happens inside the query**.
See [retrieval.md](retrieval.md#acl-pushdown) for why post-filtering is both a
correctness and a security problem.

### Other extension points

| Add a… | Implement | Register |
|---|---|---|
| Source connector | `SourceConnector` | `ingestion.register("s3", factory)` |
| Parser | `Parser` | `parsing.register("application/docx", parser)` |
| Chunker | `Chunker` | `chunking.register("layout", chunker)` |
| Reranker | `Reranker` | `backends.build_reranker` |
| LLM / embedder | `LLMClient` / `Embedder` | `backends.build_llm` / `build_embedder` |
| Authenticator | `Authenticator` | pass to `build_state(authenticator=...)` |
| Cache | `Cache` | `backends.build_cache` |

New LLM adapters should **not** implement their own retries — wrap them in
`ResilientLLM` so timeout, backoff, and fallback behave the same everywhere.

## Testing

```bash
just test                     # 60 tests, no external services
just test-cov                 # with coverage
uv run pytest tests/test_acl.py -v
uv run pytest -k semantic_cache
```

`tests/conftest.py` provides the shared fixtures: `settings` (isolated index
registry per test), `context` (in-memory `BackendContext`), `principal`,
`outsider`, `acl`.

| Test file | Guards |
|---|---|
| `test_architecture_boundaries.py` | the offline/online separation |
| `test_acl.py` | ACL logic across dense, lexical, and fused paths |
| `test_semantic_cache.py` | no cross-tenant hits, even at threshold 0.0 |
| `test_cache_keys.py` | ACL and version are part of every key |
| `test_rrf.py` | exact fusion arithmetic |
| `test_pipelines.py` | end-to-end offline and online, alias swap, rollback |
| `test_api.py` | HTTP contract, auth, readiness, refusals |
| `test_llm_resilience.py` | timeout, backoff, model fallback |
| `test_eval_metrics.py` | metric implementations |

What a new test should be worth: prefer one that would have caught a real bug.
Every ACL fix gets a regression test before it counts as fixed.

## Style

`ruff` (line length 100) and `mypy --strict` both gate CI.

- Type everything. `disallow_untyped_defs` is on.
- Assign message strings before `raise` (ruff EM/TRY convention).
- Prefer `pathlib` over `os.path`.
- Comments explain *why*, not *what*. If a line needs a comment to say what it
  does, rename something instead.
- Docstrings on modules and public functions, especially where a decision is
  non-obvious.

```bash
just fmt      # format + autofix
just lint
just typecheck
```

## CI

`.github/workflows/ci.yml` runs three jobs:

| Job | Checks |
|---|---|
| `quality` | ruff, mypy, pytest on Python 3.11 and 3.13 |
| `quality-gate` | golden-set evaluation with `--min-recall 0.8` |
| `container` | `docker compose config`, image build, `/health` up and `/ready` correctly 503 without an index |

The matrix runs the lowest and highest supported Python versions on purpose: a
3.12-only syntax feature slipped in once and only the 3.11 job would have caught
it.

## Common pitfalls

**In-memory backends do not cross processes.** `rag ingest` in one shell is
invisible to an API in another. Use `--index-source` for a self-contained eval, or
run the Docker stack.

**`data/**` is gitignored,** so `rg` and `glob` skip it by default. Use
`ls`/`--no-ignore` when looking for files under `data/`.

**Fused scores are small.** RRF output tops out near `0.03` with two legs, so a
"reasonable-looking" refusal threshold like `0.15` refuses everything.

**`Settings` is cached** via `lru_cache`. Call `reset_settings_cache()` in a test
that patches the environment.
