# ADR 0003 — Access control lives on the chunk

**Status:** accepted · **Date:** 2026-01

## Context

RAG makes a retriever into a confused deputy: it reads the whole corpus and
answers everyone. A leak does not look like a leak — it looks like a helpful
answer quoting a document the user was never allowed to see.

Three places could own access control:

1. **One index per permission set.** Clean isolation, but the index count explodes
   with group combinations, and every document belonging to two groups is stored
   twice.
2. **Document level.** Cannot express "this paragraph is restricted", and
   retrieval does not return documents — it returns passages.
3. **Chunk level.** Matches the unit that is actually retrieved.

## Decision

Every `Chunk` carries `AclTags(tenant, allow_groups, classification)`, stamped at
ingest and inherited from its `Document`. There is no code path that produces a
chunk without them.

Enforcement is layered, each layer assuming the others may fail:

1. ACLs stamped at ingest.
2. `Principal` derived from a verified token — never from request headers.
3. Tenant and group filters pushed **into** Qdrant and OpenSearch.
4. `is_allowed()` re-checked after RRF fusion.
5. ACL fingerprint folded into every cache key and semantic-cache bucket.
6. The prompt can only contain chunks that survived 1–5.

Semantics: tenant mismatch is unrecoverable; empty `allow_groups` means
tenant-wide, not public.

## Consequences

Good:

- Paragraph-level restrictions are expressible.
- One index serves all tenants; no combinatorial index explosion.
- Filtering inside the backend means `top_k` is `top_k` *among authorized chunks*,
  so a restricted user still gets a full candidate set.
- A filter regression in one adapter costs recall, not confidentiality, because
  layer 4 catches it — and records `acl_post_filtered` so it is visible.
- Cross-tenant cache hits are structurally impossible, not merely checked for.

Costs, accepted:

- Every retrieval backend must implement ACL filtering natively. This is the main
  burden on anyone adding one, and it is not optional.
- ACL changes in the source system are not reflected until reindex. A permission
  revocation is not immediate.
- Layer 4 is redundant work on every request. Cheap (≤ `top_k` items) and worth it.
- ACL granularity is currently per-ingest-run, so mixed-permission corpora need
  multiple runs until connectors derive ACLs from the source system.

## Alternatives considered

**Filter after retrieval.** Rejected twice over: it silently shrinks the candidate
set, and one forgotten call site is a leak.

**Trust the LLM to respect an instruction like "only use documents the user may
see".** Not a security control.

**Per-tenant collections.** Reasonable for a handful of large tenants; does not
survive group-level permissions within a tenant.
