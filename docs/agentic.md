# Agentic RAG

The agent is a **third subsystem** alongside the offline and online pipelines. It
does not replace `OnlinePipeline`. Most queries stay on the fixed-cost
deterministic path; the agent is reserved for multi-hop / comparative questions
and is always free to degrade back.

```
POST /query
   ├─ mode=fast            → OnlinePipeline (always)
   ├─ mode=agent           → AgentRuntime (if enabled)
   └─ mode=auto (default)  → COMPLEX route → AgentRuntime
                              else OnlinePipeline
```

## Why it is gated

Agentic-by-default would be the wrong default. An unbounded plan/act loop on the
request path is a cost-amplification attack. The agent is therefore:

1. **Router-gated** — only `COMPLEX` (or explicit `mode=agent`) enters it.
2. **Budget-bounded** — steps, tool calls, critique rounds, tokens, dollars, wall clock.
3. **Fallback-capable** — model/tool failure or no grounding falls back to `OnlinePipeline.answer()`.

## Loop

```
budget left?
  → ChatLLM plans (tool calls or draft)
  → tools run concurrently under a semaphore (principal injected by runtime)
  → critic judges draft + sources only (never the chain of thought)
  → sufficient + grounded → finalize with explicit citations
  → else feed critic.missing back as the next instruction
```

Finalization requires **explicit** citation markers. The single-pass permissive
fallback that cites the top chunk is deliberately disabled in agent mode.

## Tools

| Tool | Status | Egress | Offered to the planner |
|---|---|---|---|
| `retrieval_search` | Implemented — wraps `OnlinePipeline.retrieve_only` | No | Yes |
| `web_search` | Declared stub (`MissingBackendError`) | Yes | No — `available = False` |
| `graph_query` | Declared stub (`MissingBackendError`) | No | No — `available = False` |

Non-negotiable properties:

- **Principal is runtime-injected**, never chosen by the model.
- **Arguments are schema-validated** before execution (rejects unexpected fields).
- **Egress tools are hidden** unless `agent_allow_egress` is granted.
- **Unavailable tools are hidden**, always. A planner can only avoid a tool it was
  never offered; a call it makes anyway costs a step and a tool call before
  failing. `available` is also how a tool whose backend is down withdraws itself —
  a runtime state no description can express. `execute()` re-checks it, because a
  model can name a tool it was never shown.

Tool results are delivered as `role=tool` messages. The system prompt states that
tool content is data, never instructions. A dedicated injection test asserts that
a poisoned document instructing an egress call cannot execute it.

## Budgets

Server ceiling comes from settings (`RAG_AGENT_MAX_*`). Callers may send lower
caps on `/query`; the server **clamps**, never raises.

```python
Budget(
    max_steps=6,
    max_tool_calls=10,
    max_critique_rounds=2,
    max_tokens=20_000,
    max_cost_usd=0.10,
    max_wall_clock_seconds=30.0,
)
```

Per-tenant concurrency is also capped (`RAG_AGENT_MAX_CONCURRENT_PER_TENANT`).

## Caching

Answer-cache scope for agent turns includes mode, tool allow-list, and budget
caps — otherwise two agent configurations collide on the same query.

A separate **tool-result cache** keys on
`(tool, canonical args, ACL fingerprint, index_version)`. That is where the
savings live: agents re-retrieve similar things within and across turns. The
index version is part of the key because retrieval results are only valid for
the corpus that produced them — without it, promoting a new index keeps serving
passages from the old one for the whole cache TTL.

Traces are not returned in API responses unless `agent_include_trace` is on.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `RAG_AGENT_ENABLED` | `true` in dev | Production refuses `echo` chat when enabled |
| `RAG_CHAT_LLM_BACKEND` | `echo` | Must be a real model in production with agent on |
| `RAG_AGENT_DEFAULT_MODE` | `auto` | `auto` \| `fast` \| `agent` |
| `RAG_AGENT_ALLOW_EGRESS` | `false` | Required false outside audited egress setups |
| `RAG_AGENT_MAX_STEPS` | `6` | Server ceiling |
| `RAG_AGENT_TOOL_CACHE_ENABLED` | `true` | Tool-result cache |

## Evaluation

Deterministic agent eval uses `EchoChatLLM` scripts and gates on **steps and
cost**, not only answer quality:

```bash
just eval-agent
# or
uv run rag eval-agent --dataset tests/fixtures/golden_agent.jsonl \
  --max-avg-steps 3 --max-avg-cost 0.05 --min-trajectory-rate 1.0
```

Real-model agent eval is a separate nightly job.

### Trajectory expectations

A golden example may declare the *shape* a correct trajectory takes, not only
the answer:

| Field | Meaning |
|---|---|
| `retrieval_rounds` | Retrieval rounds the question genuinely needs. `>1` marks it multi-hop. |
| `expect_self_correction` | The agent must recover from a rejected draft rather than degrade. |

`TrajectoryOK` in the report is the fraction of examples whose turn met those
expectations, and `--min-trajectory-rate` gates it. This is separate from
`Success` on purpose: a multi-hop example answered in a single round is
*cheaper* than expected, so every cost gate stays green while the behaviour the
example exists to prove never happened.

## Related

- [ADR 0007 — Agent as a third subsystem](adr/0007-agent-as-third-subsystem.md)
- [ADR 0005 — Refusal as a feature](adr/0005-refusal-as-a-feature.md)
- [architecture.md](architecture.md) · [api.md](api.md) · [security.md](security.md)
