# Production RAG Skeleton

Two systems. One contract. Never mix them.

## Architecture

```
Offline (jobs/):  Ingest → Parse → Chunk → Embed → Index → Evaluate
Online  (api/):   Query Rewrite → Hybrid Retrieve → Rerank → Generate → Cite → Refuse
```

The online path only reads from `VectorStore` / `LexicalIndex`. Ingestion and
indexing never run inside API request handlers. A boundary test enforces this.

## Layout

| Path | Role |
|------|------|
| `data/` | Raw / interim / processed / golden — never mixed with application code |
| `schemas/` | Explicit contracts: Document, Chunk, QueryResult, Answer |
| `ingestion/` + `parsing/` + `chunking/` | Document-type strategies, not one magic splitter |
| `vectordb/` + `lexical/` | Dense + lexical retrieval |
| `retrieval/` + `rerank/` | Hybrid fusion and ranking before the LLM speaks |
| `query/` | Rewrite, route, multi-query, HyDE |
| `cache/` | ACL-aware exact and semantic answer caches |
| `prompts/` | Versioned templates, never buried in app code |
| `llm/` | LLM clients with retries, timeouts, model fallback |
| `generation/` | Grounded answers, citations, refusal |
| `eval/` | Golden datasets, faithfulness, Recall@K |
| `observability/` | Traces, latency, cost, quality |
| `security/` | Auth, chunk-level ACLs, PII redaction |
| `api/` | Online query path only |
| `jobs/` | Reindexing and scheduled evaluations |
| `utils/` | Shared text and hashing helpers |
| `logs/` | Structured logs (gitignored) |

## Quick start

```bash
uv sync --extra dev
just test          # green with zero external services
just serve         # FastAPI on :8000
just ingest        # offline pipeline over data/raw
just eval          # run golden-set evaluation
```

`main.py` is the single entry point if you prefer it over `just`:

```bash
python main.py serve
python main.py ingest --source data/raw
python main.py reindex --index-version v2
python main.py eval
```

`serve` execs uvicorn without importing the jobs modules, so the API process
never loads ingestion or indexing code.

## Containers

```bash
just build        # docker compose build
just up           # api + qdrant + opensearch + redis
just docker-reindex   # one-shot offline indexing job
```

The `api` service serves queries only; indexing runs as the separate `worker`
service under the `jobs` profile.

Default installs use in-memory stubs (`HashEmbedder`, `MemoryVectorStore`,
`BM25MemoryIndex`, `EchoLLM`). Optional extras:

```bash
uv sync --extra qdrant --extra opensearch --extra rerank --extra openai
```

## Caching

Two layers, both scoped by an ACL fingerprint plus index and prompt version:

- **Exact** (`cache/keys.py`, `cache/memory.py`) — hash of the normalized query and scope.
- **Semantic** (`cache/semantic.py`) — nearest-neighbour match for paraphrases,
  bucketed by that same scope, so a hit can never cross a tenant, group, index
  version, or prompt version boundary.

## LLM resilience

`ResilientLLM` wraps any `LLMClient` with per-attempt timeouts, exponential
backoff with jitter, and ordered model fallback. Each model exhausts its retries
before the next fallback is tried; when all are spent it raises
`LLMUnavailableError` rather than returning a silently degraded answer.

## Rules that matter

1. Never mix raw data with application code.
2. Hybrid search + reranking happens before generation.
3. Evaluation and tracing are part of the product.
4. Index versions, caching, and ACLs make RAG scalable.
5. Never embed or retrieve a document a user isn't allowed to access.

The difference between a RAG demo and a production RAG system isn't the LLM.
