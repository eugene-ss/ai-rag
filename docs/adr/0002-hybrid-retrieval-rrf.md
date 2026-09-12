# ADR 0002 — Hybrid retrieval fused with Reciprocal Rank Fusion

**Status:** accepted · **Date:** 2026-01

## Context

Dense retrieval is the default choice and it has a specific, repeatable failure:
exact tokens. Error codes, SKUs, version strings, and proper nouns are exactly
what users search for and exactly what embeddings blur. BM25 has the mirror
failure — it cannot match a question phrased entirely differently from the source.

Combining them requires merging two ranked lists whose scores are not comparable:
cosine similarity lives in `[-1, 1]`, BM25 is unbounded and corpus-dependent.
Score normalization requires knowing each backend's distribution, which changes
whenever the corpus changes.

## Decision

Run both legs concurrently and fuse by **rank**, not score:

```
score(chunk) = Σ_legs weight_leg / (k + rank_leg(chunk))
```

with `k = 60` by default. Multi-query and HyDE are the same mechanism with more
legs. Reranking runs after fusion, over the surviving candidates only.

## Consequences

Good:

- No score calibration, ever. Ranks are always comparable.
- Adding a leg (a third retriever, more query variants) requires no retuning.
- Consensus is rewarded: a chunk both legs rank highly beats a chunk one leg loves
  and the other misses.
- Concurrency makes hybrid retrieval cost roughly the slower leg, not the sum.

Costs, accepted:

- Absolute magnitudes are lost. A chunk retrieved with cosine `0.95` and one with
  `0.55` fuse identically if both rank first.
- **Fused scores are small and unintuitive** — two legs at rank 1 give ≈ `0.033`.
  This is the source of a real bug: a refusal threshold of `0.15`, which looks
  reasonable, refuses every query. Hence `refusal_score_threshold = 0.01`, and the
  requirement to re-derive it from evaluation whenever `rrf_k`, the weights, or the
  number of legs changes.
- Two indexes to keep in sync. Mitigated by writing both in one stage from one
  chunk list.

## Alternatives considered

**Dense only.** Simpler and fails on exact-match queries, which are a large share
of real traffic.

**Score normalization (min-max or z-score).** Requires per-corpus calibration that
drifts as the corpus grows, and silently degrades rather than failing loudly.

**Learned fusion.** Better ceiling, needs training data and a model to maintain.
RRF is a strong parameter-free baseline; revisit if evaluation shows fusion is the
bottleneck.
