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
    """Stable fingerprint for cache keys — must differ across tenants/groups."""
    groups = ",".join(sorted(principal.groups))
    return f"{principal.tenant}|{groups}|{principal.subject}"
