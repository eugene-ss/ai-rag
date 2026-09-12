# HTTP API

The API serves the **online query path only**. There is no ingestion, indexing,
or evaluation endpoint, by design — see
[adr/0001-two-pipelines.md](adr/0001-two-pipelines.md).

Interactive docs: `http://localhost:8000/docs` (OpenAPI at `/openapi.json`).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query` | answer a question over the indexed corpus |
| `GET` | `/health` | liveness — is the process alive |
| `GET` | `/ready` | readiness — can it answer a query |
| `GET` | `/metrics` | in-process counters and histograms |

## Authentication

A bearer token, mapped to a `Principal` server-side:

```
Authorization: Bearer <token>
```

In local development with `RAG_AUTH_TRUST_HEADERS=true` you may instead send
`X-Tenant`, `X-Groups`, `X-Subject`. These headers are **ignored** by every other
authenticator, and the process refuses to start with header trust enabled in
production. See [security.md](security.md#authentication).

`X-Request-Id` is optional on any request; it is echoed back and used as the
trace id, so client and server logs correlate.

## POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H 'Authorization: Bearer <token>' \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: my-correlation-id' \
  -d '{"query": "What is hybrid retrieval?"}'
```

Request:

| Field | Type | Constraints |
|---|---|---|
| `query` | string | 1–4000 characters |
| `mode` | string | `auto` (default), `fast`, or `agent` |
| `max_steps` | int? | optional agent budget; clamped by server ceiling |
| `max_tool_calls` | int? | optional agent budget |
| `max_critique_rounds` | int? | optional agent budget |
| `max_tokens` | int? | optional agent budget |
| `max_cost_usd` | float? | optional agent budget |
| `max_wall_clock_seconds` | float? | optional agent budget |

`mode=auto` elevates only `COMPLEX` routes to the agent. `fast` always uses the
deterministic online pipeline. `agent` forces the bounded agent when enabled.
Caller budgets are never raised above the server ceiling. See
[agentic.md](agentic.md).

Response `200`:

```json
{
  "answer": {
    "text": "Hybrid retrieval combines dense and lexical search [a1b2c3d4].",
    "citations": [
      {
        "chunk_id": "a1b2c3d4e5f6",
        "doc_id": "hybrid_retrieval",
        "quote": "Dense retrieval alone is not enough...",
        "score": 0.0323
      }
    ],
    "refused": false,
    "refusal_reason": null,
    "usage": {
      "prompt_tokens": 412,
      "completion_tokens": 88,
      "total_tokens": 500,
      "cost_usd": 0.00012,
      "model": "gpt-4o-mini",
      "prompt_version": "v1"
    },
    "trace_id": "my-correlation-id",
    "index_version": "v1",
    "latency_ms": 842.3,
    "cached": false
  }
}
```

### Refusals are 200, not errors

```json
{
  "answer": {
    "text": "I don't know based on the available documents.",
    "citations": [],
    "refused": true,
    "refusal_reason": "no_retrieved_context",
    "trace_id": "...",
    "index_version": "v1"
  }
}
```

**Branch on `refused`, never on the status code.** The request succeeded; the
system declined to answer because the corpus (as visible to this caller) does not
support one. Reasons are listed in [contracts.md](contracts.md#answer).

A caller with no access to matching content receives exactly this — never a 403,
and never a hint that content exists. Existence disclosure is itself a leak.

### Errors

| Status | Code | Cause |
|---|---|---|
| `401` | `unauthorized` | missing or invalid token (`WWW-Authenticate: Bearer`) |
| `422` | — | validation failure (empty or over-long `query`) |
| `500` | `internal_error` | unexpected failure; details are logged, not returned |
| `503` | `backend_unavailable` | a dependency is unusable |

```json
{"error": {"code": "unauthorized", "message": "invalid_token", "trace_id": "…"}}
```

The `trace_id` in an error body is the same id in the logs. Quote it in a bug
report.

## GET /health

Liveness. Cheap, no dependency checks — wire it to a liveness probe.

```json
{"status": "ok", "service": "rag-api", "env": "production"}
```

## GET /ready

Readiness. `200` when the alias resolves and the index is populated, `503`
otherwise. Wire it to a readiness probe so a replica whose index has not been
published never receives traffic.

```json
{
  "status": "ready",
  "ready": true,
  "checks": {"index_alias": true, "index_populated": true, "cache": true},
  "index_version": "v1",
  "collection_alias": "docs_live",
  "chunk_count": 12043,
  "authenticator": "static_token"
}
```

## GET /metrics

JSON snapshot of process-local counters and histograms. Per-replica and reset on
restart. Returns `404` when `RAG_METRICS_ENABLED=false`. Metric list:
[observability.md](observability.md#metrics).

## Client guidance

- **Send `X-Request-Id`** and log it with your own request id. It is the entire
  debugging story.
- **Handle refusals as a normal outcome.** Show the user that the corpus does not
  cover their question rather than an error.
- **Render citations.** An answer without visible sources is unverifiable, and
  `citations` is exactly what makes it checkable.
- **Set a client timeout above the generation budget** (a few seconds), and
  retry `503` with backoff. Do not retry `401` or `422`.
- **Treat `cached: true` as informational** — a cached answer is equally valid.
