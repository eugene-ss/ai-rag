# ADR 0004 — Index versions with alias-based promotion

**Status:** accepted · **Date:** 2026-01

## Context

Reindexing is not rare. It is triggered by a new embedding model, a chunking
change, a schema change, or simply new documents. Writing into the live index
means:

- readers observe a partially built index and get worse answers for the duration
- a failed job leaves the index in an unknown state
- there is no rollback — the previous state is gone

The job also takes minutes to hours, so "briefly inconsistent" means a long time.

## Decision

Build into a versioned collection, then flip an alias.

- Physical name: `{collection_prefix}__{index_version}`, e.g. `docs__v3`.
- Readers resolve the alias `docs_live` **on every request**; no reader knows a
  version name.
- `configs/index_versions.yaml` records each version's status, chunk count, and
  embedding model, and which one is active.
- Promotion and rollback are the same operation: `rag activate --index-version vN`.
- `--no-activate` separates "built" from "serving", so evaluation can run against a
  version before traffic does.

Chunk ids are deterministic in the version:
`short_id(doc_id, ordinal, chunker_version, index_version)`, which makes a rerun
idempotent rather than duplicating content.

## Consequences

Good:

- Readers never see a half-built index.
- Rollback is an alias flip: seconds, no redeploy, no restart.
- A failed job is a no-op for users.
- `rag versions` shows which embedding model built each version, so a
  half-migrated index is detectable.
- Evaluating before promoting is the natural workflow rather than an extra step.

Costs, accepted:

- Storage for at least two versions. Necessary for rollback to mean anything.
- Old versions must be pruned deliberately; nothing deletes them automatically,
  because automatic deletion of the rollback target is a bad trade.
- Promotion invalidates every cache entry, since the index version is part of the
  cache key. Correct, but it means a cold-cache latency spike after promotion.
- Two sources of truth for "active": the store alias and the registry file. The
  alias wins at query time; the file exists for history and rollback, and degrades
  to a warning on a read-only filesystem.

## Alternatives considered

**Write in place.** Simple, and every failure mode above applies.

**Blue/green whole deployments.** Works, but couples an index rebuild to an
application deploy — the two have completely different cadences.

**Delete and rebuild.** Guarantees downtime and no rollback path.
