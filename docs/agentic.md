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

| Tool | Status | Egress |
|---|---|---|
| `retrieval_search` | Implemented — wraps `OnlinePipeline.retrieve_only` | No |
| `web_search` | Declared stub (`MissingBackendError`) | Yes |
| `graph_query` | Declared stub (`MissingBackendError`) | Yes |

Non-negotiable properties:

- **Principal is runtime-injected**, never chosen by the model.
- **Arguments are schema-validated** before execution (rejects unexpected fields).
- **Egress tools are hidden** unless `agent_allow_egress` is granted.

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

A separate **tool-result cache** keys on `(tool, canonical args, ACL fingerprint)`.
That is where the savings live: agents re-retrieve similar things within and
across turns.

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
  --max-avg-steps 4 --max-avg-cost 0.05
```

Real-model agent eval is a separate nightly job.

## Related

- [ADR 0007 — Agent as a third subsystem](adr/0007-agent-as-third-subsystem.md)
- [ADR 0005 — Refusal as a feature](adr/0005-refusal-as-a-feature.md)
- [architecture.md](architecture.md) · [api.md](api.md) · [security.md](security.md)
