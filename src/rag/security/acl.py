from __future__ import annotations

from rag.schemas import AclTags, Principal


def is_allowed(principal: Principal, acl: AclTags) -> bool:
    """Return True if principal may access a document/chunk with the given ACL."""
    if principal.tenant != acl.tenant:
        return False
    if not acl.allow_groups:
        # Empty allow_groups means tenant-wide access within the tenant.
        return True
    return bool(principal.groups & acl.allow_groups)


def acl_fingerprint(principal: Principal) -> str:
    """Stable cache-scope fingerprint: identical reach must produce one key.

    The fingerprint must contain exactly what `is_allowed` reads — tenant and
    groups — and nothing else. Both halves of that matter:

    - Include less and the cache leaks across an ACL boundary.
    - Include more and the cache fragments without adding isolation. `subject`
      used to be here, which gave every user a private cache of results they
      were all equally entitled to see. In a tenant of a thousand engineers
      asking overlapping questions the hit rate approaches zero, so the cache
      costs memory and returns nothing.

    If ACL evaluation ever becomes subject-dependent (per-user ownership, say),
    this function must change in the same commit, or the cache will serve one
    user's documents to another.
    """
    groups = ",".join(sorted(principal.groups))
    return f"{principal.tenant}|{groups}"
