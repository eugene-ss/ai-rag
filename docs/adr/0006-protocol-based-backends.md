# ADR 0006 — Protocols with dependency-free default implementations

**Status:** accepted · **Date:** 2026-01

## Context

A RAG system needs a vector database, a lexical index, an embedding provider, an
LLM, and a cache. Depending on all of them directly means:

- the test suite needs Docker, network access, and API keys
- tests are slow and flaky, so they get skipped, so they stop protecting anything
- swapping a provider touches every call site
- a contributor cannot run the project without an account somewhere

## Decision

Every backend is a `typing.Protocol`, and every protocol ships an in-process
implementation with no third-party dependency:

| Protocol | Stub | Real |
|---|---|---|
| `VectorStore` | `MemoryVectorStore` | `QdrantVectorStore` |
| `LexicalIndex` | `BM25MemoryIndex` | `OpenSearchLexicalIndex` |
| `Embedder` | `HashEmbedder` | `OpenAIEmbedder` |
| `LLMClient` | `EchoLLM` | `OpenAIChatClient` |
| `Cache` | `MemoryCache` | `RedisCache` |
| `Reranker` | `IdentityReranker` | `CrossEncoderReranker` |
| `Authenticator` | `TrustedHeaderAuthenticator` | `StaticTokenAuthenticator` |

Supporting rules:

- Third-party imports are **lazy**, inside `__init__`, raising
  `MissingBackendError` with the name of the missing extra.
- `backends.py` is the only place that maps configuration to implementation;
  `BackendContext` builds all of them once per process.
- Protocols, not ABCs: an adapter needs no import from this project to satisfy one.
- `Settings` refuses the `hash` embedder and `echo` LLM in production, so the stubs
  cannot be deployed by accident.

## Consequences

Good:

- The full suite (60 tests) runs in ~2 seconds with no services and no keys.
- `git clone && just check` works on a fresh machine.
- Retrieval, fusion, ACL, cache, and refusal logic are tested deterministically —
  `HashEmbedder` gives identical vectors every run.
- Swapping a provider is one environment variable.
- The API image can install only the extras it actually uses.

Costs, accepted:

- The stubs are not semantically useful. `HashEmbedder` has no notion of meaning,
  so tests verify *plumbing*, not answer quality — that is what `eval/` is for.
- The real adapters are exercised by the Docker stack and by production, not by
  unit tests. A Qdrant filter regression would not be caught locally, which is
  precisely why the post-fusion ACL recheck exists.
- Two implementations of every protocol to maintain.
- Lazy imports move a missing dependency from import time to construction time.
  Mitigated by a clear error naming the extra to install.

## Alternatives considered

**Depend on Qdrant and OpenSearch directly, use testcontainers.** Higher fidelity,
much slower, and requires Docker for every contributor — tests that take minutes
get run less often.

**Mock the clients in tests.** Mocks assert that code calls a library in a
particular way, not that the behaviour is right. `MemoryVectorStore` actually
retrieves and actually enforces ACLs.

**Abstract base classes.** Force adapters to inherit from this project's types,
which makes wrapping a third-party client awkward for no benefit.
