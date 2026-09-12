# ADR 0005 — Refusal is a normal return value

**Status:** accepted · **Date:** 2026-01

## Context

An LLM given retrieved context will answer even when the context does not support
an answer. That is the single most expensive failure mode in RAG: a fluent,
confident, wrong answer with a plausible citation. Users cannot detect it, and
one instance costs more trust than ten "I don't know"s.

So refusal has to be a first-class outcome. The question is how to represent it.
Two obvious options are wrong:

- **An exception.** Refusal is not an error — the system worked correctly. Callers
  would wrap every call in `try/except` and be tempted to swallow it.
- **A 4xx status.** Same problem, plus it leaks information: a 403-style response
  tells the caller that content exists but is inaccessible.

## Decision

Refusal is an `Answer` with `refused=True`, an empty `citations` list, a
`refusal_reason`, and HTTP **200**.

Refusal is checked at every stage where confidence can fail:

| Reason | Checked |
|---|---|
| `empty_query` | routing, before any retrieval |
| `no_retrieved_context` | after retrieval — nothing visible matched |
| `low_confidence_score` | best fused score below the threshold, before any token is spent |
| `zero_resolvable_citations` | after generation — the model cited nothing verifiable |
| `llm_unavailable` | every model and retry exhausted |

Two supporting rules:

- Citations are **resolved**, not trusted: an emitted id must match a retrieved
  chunk. An unresolvable citation is not a citation.
- Refusals are **never cached**. They are often transient (an index mid-build, a
  provider outage), and caching one extends an incident for a full TTL.

## Consequences

Good:

- Callers branch on one boolean. No exception handling, no status-code mapping.
- Refusal for "no access" is indistinguishable from "not in the corpus", so the
  API never discloses the existence of inaccessible content.
- Two refusal paths short-circuit before the LLM call, so the cheap failure is
  also the fast one.
- `refusals{reason}` in metrics turns "quality got worse" into a specific,
  actionable signal.

Costs, accepted:

- Clients that only check the HTTP status will treat a refusal as a successful
  answer. Documented prominently in [../api.md](../api.md).
- The refusal threshold is tied to RRF score scale, which is small and unintuitive
  (see [0002](0002-hybrid-retrieval-rrf.md)) and must be re-derived from evaluation
  when fusion parameters change.
- Over-refusal is possible and is a real quality bug. `RefusalRate` in evaluation
  exists to catch it in both directions.

## Alternatives considered

**Return a low-confidence answer with a warning field.** Users read the answer and
ignore the warning.

**Always answer, let the client judge.** The client has strictly less information
than the pipeline does.

**Raise an exception.** Turns a correct outcome into an error path, and error paths
get swallowed.
