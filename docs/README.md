# Documentation

Start with [architecture.md](architecture.md) — it contains the diagrams and
explains the one decision everything else follows from: offline indexing and
online serving are separate systems joined by a contract.

## By topic

| Document | Read it when you want to |
|---|---|
| [architecture.md](architecture.md) | Understand the whole system and see the diagrams |
| [agentic.md](agentic.md) | Bounded agent as a third subsystem |
| [contracts.md](contracts.md) | Know exactly what `Document`, `Chunk`, `Answer` guarantee |
| [offline-pipeline.md](offline-pipeline.md) | Change ingestion, parsing, chunking, or indexing |
| [online-pipeline.md](online-pipeline.md) | Change query handling, generation, or refusal |
| [retrieval.md](retrieval.md) | Tune hybrid search, RRF, or reranking |
| [caching.md](caching.md) | Reason about cache hits, TTLs, and tenant isolation |
| [security.md](security.md) | Wire real auth, or review the threat model |
| [evaluation.md](evaluation.md) | Add golden examples or gate a release on quality |
| [observability.md](observability.md) | Debug a slow or wrong answer in production |
| [configuration.md](configuration.md) | Look up a setting or environment variable |
| [api.md](api.md) | Call the HTTP API |
| [deployment.md](deployment.md) | Deploy, scale, or size the stack |
| [operations.md](operations.md) | Reindex, roll back, or handle an incident |
| [development.md](development.md) | Add a backend, chunker, or connector |
| [adr/](adr/) | Understand why a decision was made |

## By task

- **Get it running:** [../README.md#quickstart](../README.md#quickstart)
- **Ship a new corpus version:** [operations.md#reindex-and-promote](operations.md#reindex-and-promote)
- **Answers are wrong:** [evaluation.md](evaluation.md), then [retrieval.md#tuning](retrieval.md#tuning)
- **Answers are slow:** [online-pipeline.md#latency-budget](online-pipeline.md#latency-budget), then [caching.md](caching.md)
- **Move off the stubs to real backends:** [deployment.md#choosing-backends](deployment.md#choosing-backends)
- **A tenant saw the wrong document:** [security.md#incident-response](security.md#incident-response)

## Conventions in these docs

Diagrams are [Mermaid](https://mermaid.js.org/) in fenced blocks, rendered by
GitHub and most IDEs. Code references point at real files; if a snippet and the
code disagree, the code is correct and the doc is a bug.
