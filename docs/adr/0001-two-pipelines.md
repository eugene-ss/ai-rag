# ADR 0001 — Offline indexing and online serving are separate systems

**Status:** accepted · **Date:** 2026-01

## Context

The natural way to build RAG is one application: an endpoint that can also ingest,
because ingestion needs the same embedder and the same store. It is fewer moving
parts and it demos well.

It fails in production for reasons that are structural rather than accidental:

- The two workloads have latency budgets three orders of magnitude apart —
  milliseconds versus minutes.
- They fail differently. A bad document should cost one document, not one user's
  request; and vice versa.
- They scale on different signals. Requests per second has nothing to do with
  corpus size.
- Once indexing code is importable from a request handler, someone eventually
  calls it there. Then a request triggers a reindex, and the incident is
  latency, cost, and a corrupted index at once.

## Decision

Two pipelines joined by a contract, and no other coupling.

- Offline (`pipelines/offline.py`, driven by `jobs/`): ingest, parse, chunk,
  embed, index, publish, evaluate. The only writer.
- Online (`pipelines/online.py`, served by `api/`): rewrite, retrieve, rerank,
  generate, cite, refuse. Read-only.
- The contract is `schemas/`: `Document`, `Chunk`, `QueryResult`, `Answer`,
  `Principal`, `AclTags`.

Enforced, not documented: `tests/test_architecture_boundaries.py` builds the
transitive import graph of the `rag` package and fails if anything under `rag.api`
can reach `rag.ingestion`, `rag.parsing`, `rag.chunking`, `rag.eval`, or
`rag.jobs`. `main.py serve` does not import the jobs CLI at all.

The check has to be transitive. A direct-import version of it passed while
`pipelines/__init__.py` re-exported both pipelines, which loaded every parser and
chunker into the API process — the exact outcome this ADR exists to prevent.
Convenience re-exports in shared `__init__` files are the usual way this boundary
erodes.

## Consequences

Good:

- An indexer cannot end up in a request path — the build breaks first.
- Each side scales and deploys independently; the API image needs no parsers.
- Reindexing cannot affect request latency.
- Both sides are testable in isolation.

Costs, accepted:

- Two deployment units (an API Deployment and a CronJob) instead of one.
- Shared code must live in a neutral module (`schemas`, `settings`, `backends`,
  `observability`), which takes more discipline than importing across the line.
- No "index this document now" endpoint. Adding one means enqueueing a job, not
  calling the pipeline inline.

## Alternatives considered

**One process, ingestion behind a feature flag.** The import boundary is what
prevents the mistake; a flag does not.

**Background thread in the API process.** Indexing competes with requests for CPU
and memory in the same address space, and a crash during indexing takes the API
replica with it.
