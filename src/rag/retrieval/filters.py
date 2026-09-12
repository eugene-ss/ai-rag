"""ACL pushdown filters.

Every backend must exclude unauthorized chunks *during* search, not after, so a
`top_k` is never silently consumed by documents the caller cannot see. Each
store translates this description into its own native filter language:

- Qdrant   -> `models.Filter` on the `tenant` / `allow_groups` payload
- OpenSearch -> a `bool.filter` with `term` / `terms` clauses
- memory   -> a direct `is_allowed` predicate
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.schemas import Principal
from rag.security.acl import acl_fingerprint


@dataclass(frozen=True)
class AclFilter:
    """Backend-neutral description of what a principal may retrieve."""

    tenant: str
    groups: frozenset[str]
    fingerprint: str
    index_version: str | None = None

    @classmethod
    def for_principal(
        cls,
        principal: Principal,
        *,
        index_version: str | None = None,
    ) -> AclFilter:
        return cls(
            tenant=principal.tenant,
            groups=principal.groups,
            fingerprint=acl_fingerprint(principal),
            index_version=index_version,
        )

    def as_dict(self) -> dict[str, object]:
        """Loggable, cache-key-safe representation. Contains no document text."""
        return {
            "tenant": self.tenant,
            "groups": sorted(self.groups),
            "fingerprint": self.fingerprint,
            "index_version": self.index_version,
        }
