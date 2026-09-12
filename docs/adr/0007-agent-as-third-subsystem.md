# ADR 0007 — Agent as a third subsystem

**Status:** accepted · **Date:** 2026-09

## Context

Multi-hop and comparative questions need more than one retrieve→generate pass.
Folding an unbounded plan/act loop into `OnlinePipeline` would destroy the
property the system is built on: a fixed-cost, read-only request path.

Making the agent the default path would amplify cost and turn prompt injection
from a wrong-answer problem into an exfiltration problem (retrieved text can
influence tool calls).

## Decision

Add a bounded reasoning agent as a **third subsystem** that *composes*
`OnlinePipeline`, gated by the router and hard budgets:

1. **Compose, don't replace.** `RetrievalTool` calls `retrieve_only`, never
   nested `answer()`. Fallback ladder ends at `OnlinePipeline.answer()`.
2. **Router + mode.** Default `mode=auto` only elevates `COMPLEX` routes.
   `fast` stays deterministic; `agent` forces the loop when enabled.
3. **Hard budgets.** Steps, tool calls, critique rounds, tokens, dollars, wall
   clock. Caller budgets are clamped by the server ceiling.
4. **Independent critic.** Sees draft + sources only — never the planner's
   chain of thought. Cannot call tools.
5. **Injection posture.** Principal is runtime-injected; tool args validated;
   egress gated; tool content is data; explicit citations required for
   finalization; per-tenant concurrency semaphore.

`ChatLLM` is a sibling protocol to `LLMClient`. `EchoChatLLM` is the hermetic
stub (same role as `HashEmbedder`).

## Consequences

- p50 latency and cost stay where they are for factual queries.
- Agent regressions are gated on average steps and cost in CI, not only recall.
- Web/graph tools can be added behind the existing `Tool` protocol without
  changing the loop; egress grant must stay explicit.
- Production refuses `agent_enabled` with an echo chat backend and refuses
  `agent_allow_egress` until egress tools are audited.
