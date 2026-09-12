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
| `cache/` | ACL-aware answer cache |
| `prompts/` | Versioned templates, never buried in app code |
| `generation/` | Grounded answers, citations, refusal |
| `eval/` | Golden datasets, faithfulness, Recall@K |
| `observability/` | Traces, latency, cost, quality |
| `security/` | Auth, chunk-level ACLs, PII redaction |
| `api/` | Online query path only |
| `jobs/` | Reindexing and scheduled evaluations |

## Quick start

```bash
uv sync --extra dev
just test          # green with zero external services
just serve         # FastAPI on :8000
just ingest        # offline pipeline over data/raw
just eval          # run golden-set evaluation
```

Default installs use in-memory stubs (`HashEmbedder`, `MemoryVectorStore`,
`BM25MemoryIndex`, `EchoLLM`). Optional extras:

```bash
uv sync --extra qdrant --extra opensearch --extra rerank --extra openai
```

## Rules that matter

1. Never mix raw data with application code.
2. Hybrid search + reranking happens before generation.
3. Evaluation and tracing are part of the product.
4. Index versions, caching, and ACLs make RAG scalable.
5. Never embed or retrieve a document a user isn't allowed to access.

The difference between a RAG demo and a production RAG system isn't the LLM.
