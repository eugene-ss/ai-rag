# Architecture

A production RAG system is **two systems joined by a contract**, not one
application with a vector database attached.

| | Offline pipeline | Online pipeline |
|---|---|---|
| Runs as | `jobs/` (CLI, cron, worker container) | `api/` (FastAPI process) |
| Trigger | schedule or explicit operator action | user request |
| Latency budget | minutes to hours | milliseconds |
| Writes to the index | yes | **never** |
| Failure blast radius | one index version | one request |

The contract between them is `src/rag/schemas/`: `Document`, `Chunk`,
`QueryResult`, `Answer`, `Principal`, `AclTags`. Either side can be rewritten
without touching the other as long as the contract holds.

`tests/test_architecture_boundaries.py` builds the **transitive** import graph of
the `rag` package and fails the build if anything under `rag.api` can reach
`rag.ingestion`, `rag.parsing`, `rag.chunking`, `rag.eval`, or `rag.jobs` by any
path. Importing `rag.api.app` loads zero offline modules.

Transitivity is the part that matters. A direct-import check passes while
`pipelines/__init__.py` quietly re-exports both pipelines and pulls every parser
and chunker into the API process — which is exactly the bug the strengthened test
found. The separation is enforced, not merely intended; otherwise an indexer
eventually ships inside a request handler, and with it the latency, cost, and
incidents.

---

## 1. System context

```mermaid
flowchart LR
    subgraph sources["Data sources"]
        FS["Local filesystem<br/>data/raw"]
        S3["S3 / SharePoint / Confluence<br/>(add a SourceConnector)"]
    end

    subgraph offline["OFFLINE — jobs/ (write path)"]
        direction TB
        JOB["rag ingest / reindex"]
    end

    subgraph stores["Shared state"]
        direction TB
        VDB[("Vector index<br/>Qdrant")]
        LEX[("Lexical index<br/>OpenSearch BM25")]
        REG[["Index registry<br/>configs/index_versions.yaml"]]
        CACHE[("Answer cache<br/>Redis")]
    end

    subgraph online["ONLINE — api/ (read path)"]
        direction TB
        API["FastAPI<br/>POST /query"]
    end

    LLM["LLM + embedding provider"]
    USER["Client application"]
    OPS["Operators / dashboards"]

    FS --> JOB
    S3 -.-> JOB
    JOB -->|"upsert, then flip alias"| VDB
    JOB --> LEX
    JOB --> REG
    JOB -->|embeddings| LLM

    USER -->|"Bearer token"| API
    API -->|"read only"| VDB
    API --> LEX
    API --> CACHE
    API -->|generation| LLM
    API -.->|"resolve alias"| REG
    API --> OPS
    JOB -->|"eval reports"| OPS

    classDef off fill:#e8f0fe,stroke:#3b6fd4,stroke-width:1px
    classDef on fill:#e9f7ef,stroke:#2f9e5f,stroke-width:1px
    class offline,JOB off
    class online,API on
```

The only coupling between the two processes is shared state. Neither imports
the other.

---

## 2. Module map

Directory layout mirrors the pipeline stages, so a stack trace tells you which
stage failed.

```mermaid
flowchart TB
    subgraph contracts["schemas/ — the contract"]
        SCH["Document · Chunk · ScoredChunk<br/>QueryResult · Answer · Citation<br/>Principal · AclTags"]
    end

    subgraph offlinemods["Offline modules"]
        ING["ingestion/<br/>connectors, checksums, mime"]
        PAR["parsing/<br/>text · markdown · html · pdf"]
        CHU["chunking/<br/>fixed · recursive · markdown_aware · semantic"]
        EMB["embedding/<br/>hash · openai"]
        IDX["vectordb/ + lexical/<br/>write side"]
    end

    subgraph onlinemods["Online modules"]
        QRY["query/<br/>rewrite · route · multi_query · hyde"]
        RET["retrieval/<br/>hybrid + RRF + ACL pushdown"]
        RER["rerank/<br/>identity · cross_encoder"]
        PRO["prompts/<br/>versioned templates"]
        GEN["generation/<br/>grounded · citations · refusal"]
        CAC["cache/<br/>exact · semantic"]
    end

    subgraph cross["Cross-cutting"]
        SEC["security/<br/>auth · ACL · PII"]
        OBS["observability/<br/>traces · metrics · cost"]
        EVA["eval/<br/>golden sets · metrics · report"]
        CFG["settings.py + backends.py<br/>configs/"]
    end

    ING --> PAR --> CHU --> EMB --> IDX
    QRY --> RET --> RER --> GEN
    CAC -.->|"short-circuit"| GEN
    PRO --> GEN
    contracts --- offlinemods
    contracts --- onlinemods
    SEC -.-> offlinemods
    SEC -.-> onlinemods
    OBS -.-> offlinemods
    OBS -.-> onlinemods
    CFG -.-> offlinemods
    CFG -.-> onlinemods
    EVA -.-> onlinemods
```

Every box is a `typing.Protocol` with at least one dependency-free
implementation, so the whole system runs and is testable without a single
external service.

---

## 3. Offline pipeline

```mermaid
flowchart LR
    A["Ingest<br/><small>connector, checksum, mime, ACL stamp</small>"]
    B["Parse<br/><small>per mime type, PII redaction</small>"]
    C["Chunk<br/><small>strategy per document type</small>"]
    D["Embed<br/><small>model name stamped on chunk</small>"]
    E["Index<br/><small>write to docs__vN</small>"]
    F["Publish<br/><small>flip alias docs_live → vN</small>"]
    G["Evaluate<br/><small>golden set gate</small>"]

    A --> B --> C --> D --> E --> F --> G
    G -->|"regression"| H["Roll back:<br/>rag activate --index-version vN-1"]
```

Properties that make this safe to run repeatedly:

- **Deterministic chunk ids.** `short_id(doc_id, ordinal, chunker_version,
  index_version)` — re-running produces identical ids, so indexing is
  idempotent rather than duplicating content.
- **Build-then-swap.** Writes go to `docs__vN` while `docs_live` still points at
  `vN-1`. Readers never observe a half-built index, and rollback is another
  alias flip.
- **Embedding model stamped per chunk.** Changing the embedder changes
  `Chunk.embedding_model`, which makes a silent mixed-model index detectable.
- **One bad document cannot fail a reindex.** Parse failures are counted and
  skipped.

Details: [offline-pipeline.md](offline-pipeline.md).

---

## 4. Online pipeline

```mermaid
flowchart TB
    Q["POST /query"] --> AUTH{"Authenticate"}
    AUTH -->|"401"| DENY["Reject"]
    AUTH -->|"Principal"| CK["Exact cache<br/><small>sha256(scope + normalized query)</small>"]
    CK -->|hit| OUT
    CK -->|miss| SC["Semantic cache<br/><small>cosine ≥ 0.95, within ACL scope</small>"]
    SC -->|hit| OUT
    SC -->|miss| ROUTE{"Route"}

    ROUTE -->|empty| REF["Refuse<br/>empty_query"]
    ROUTE -->|chitchat| SMALL["Short reply, no retrieval"]
    ROUTE -->|retrieval| VAR["Query variants<br/><small>rewrite · multi-query · HyDE</small>"]

    VAR --> DEN["Dense search<br/><small>ACL pushed into Qdrant</small>"]
    VAR --> LEXS["BM25 search<br/><small>ACL pushed into OpenSearch</small>"]
    DEN --> RRF["RRF fusion"]
    LEXS --> RRF
    RRF --> ACL2["Re-check ACLs<br/><small>defense in depth</small>"]
    ACL2 --> RR["Rerank → rerank_top_k"]
    RR --> THR{"Top score ≥<br/>refusal threshold?"}
    THR -->|no| REF2["Refuse<br/>low_confidence_score"]
    THR -->|yes| GEN["Generate<br/><small>versioned prompt + context</small>"]
    GEN --> CIT{"Citations<br/>resolve?"}
    CIT -->|none| REF3["Refuse<br/>zero_resolvable_citations"]
    CIT -->|yes| RED["Redact PII"]
    RED --> WR["Cache<br/><small>answers only, never refusals</small>"]
    WR --> OUT["Answer + citations + trace_id"]
```

Retrieval and reranking always happen **before** the model is asked anything.
The model never chooses what to retrieve, and it only ever sees chunks the
caller is allowed to see.

Refusal is a normal return value — `refused=true`, empty `citations`, HTTP 200 —
not an exception. Details: [online-pipeline.md](online-pipeline.md).

---

## 5. Request sequence

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant M as TraceMiddleware
    participant A as Authenticator
    participant P as OnlinePipeline
    participant K as Cache
    participant V as Qdrant
    participant L as OpenSearch
    participant G as LLM

    C->>M: POST /query (Bearer, X-Request-Id?)
    M->>M: assign/join trace id
    M->>A: authenticate(credentials)
    A-->>M: Principal(subject, tenant, groups)
    M->>P: answer(query, principal)
    P->>K: get(cache_key(scope, query))
    K-->>P: miss
    par Hybrid retrieval
        P->>V: search(vector, acl filter)
        V-->>P: dense hits
    and
        P->>L: search(text, acl filter)
        L-->>P: lexical hits
    end
    P->>P: RRF fuse → re-check ACL → rerank
    alt no chunk above threshold
        P-->>M: Answer(refused, reason)
    else
        P->>G: complete(prompt with context)
        G-->>P: text + usage
        P->>P: resolve citations, redact PII
        P->>K: set(key, answer, ttl)
        P-->>M: Answer + citations
    end
    M-->>C: 200 + X-Request-Id
```

Both retrieval legs run concurrently, so hybrid search costs roughly the slower
leg rather than their sum.

---

## 6. ACL enforcement layers

> Never embed or retrieve a document a user isn't allowed to access.

```mermaid
flowchart TB
    L1["1 · Ingest<br/>every chunk carries AclTags(tenant, allow_groups, classification)"]
    L2["2 · Authenticate<br/>Principal comes from a verified token, never from headers"]
    L3["3 · Pushdown<br/>tenant + group filters evaluated inside Qdrant / OpenSearch"]
    L4["4 · Post-fusion recheck<br/>is_allowed() again after RRF"]
    L5["5 · Cache scope<br/>ACL fingerprint is part of every cache key and bucket"]
    L6["6 · Generation<br/>the prompt can only contain chunks that survived 1–5"]

    L1 --> L2 --> L3 --> L4 --> L5 --> L6
```

Layer 3 alone would be enough if backends were perfect. Layer 4 exists because
a filter regression in one adapter must not become a data leak. Layer 5 exists
because a cache is the easiest way to leak across tenants: identical text asked
by two tenants must never share an entry. `tests/test_semantic_cache.py` proves
a cross-tenant miss even with the similarity threshold forced to `0.0`.

Details and threat model: [security.md](security.md).

---

## 7. Index version lifecycle

```mermaid
stateDiagram-v2
    [*] --> building: rag reindex --index-version v2
    building --> built: all chunks written
    building --> failed: job error
    failed --> [*]: v1 still live, no impact
    built --> active: alias flip (rag activate)
    active --> retired: a newer version is activated
    retired --> active: rollback (rag activate --index-version v1)
    active --> [*]
```

Online readers resolve `docs_live` through the alias on every request rather
than reading a configured version name, so promotion and rollback need no
redeploy and no restart. Runbooks: [operations.md](operations.md).

---

## 8. Deployment topology

```mermaid
flowchart TB
    subgraph edge["Edge"]
        LB["Load balancer<br/><small>/health liveness · /ready readiness</small>"]
    end

    subgraph apis["API replicas — stateless, scale horizontally"]
        A1["api 1"]
        A2["api 2"]
        A3["api N"]
    end

    subgraph jobsx["Jobs — scheduled, never in the request path"]
        W1["reindex worker"]
        W2["nightly eval"]
    end

    subgraph data["Stateful services"]
        QD[("Qdrant")]
        OS[("OpenSearch")]
        RD[("Redis")]
    end

    LB --> A1 & A2 & A3
    A1 & A2 & A3 --> QD & OS & RD
    W1 --> QD & OS
    W2 --> QD & OS
```

API replicas hold no durable state, which is why the cache must be Redis rather
than in-process once there is more than one replica: `MemoryCache` would give
each replica a different view of the same question. Details:
[deployment.md](deployment.md).

---

## 9. Where each rule lives in the code

| Rule | Enforced by |
|---|---|
| Never mix raw data with application code | `data/**` gitignored except layout; `configs/` separate; `.env` never committed; images mount `./data:ro` |
| Hybrid search + reranking before generation | `retrieval/hybrid.py` → `rerank/` → `generation/grounded.py`, in that fixed order in `pipelines/online.py` |
| Evaluation and tracing are part of the product | `eval/` with a CI gate (`--min-recall`); `observability/tracing.py` spans on every stage; trace id on every answer |
| Index versions, caching, ACLs make it scalable | `index_registry.py` + alias swap; `cache/` exact + semantic; `security/acl.py` |
| Never embed or retrieve an unauthorized document | ACLs stamped at ingest, pushed into both backends, rechecked after fusion, folded into cache keys |

## Further reading

- [contracts.md](contracts.md) — the schemas both pipelines depend on
- [retrieval.md](retrieval.md) — RRF, BM25, and reranking in detail
- [caching.md](caching.md) — exact and semantic cache design
- [evaluation.md](evaluation.md) — metrics and quality gates
- [observability.md](observability.md) — traces, metrics, cost
- [configuration.md](configuration.md) — every setting
- [operations.md](operations.md) — runbooks
- [adr/](adr/) — why the tricky decisions were made
