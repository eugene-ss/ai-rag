from __future__ import annotations

from rag.schemas import Principal
from rag.security.acl import acl_fingerprint


def acl_filter_dict(principal: Principal) -> dict[str, object]:
    """Pushdown filter representation passed into vector/lexical backends."""
    return {
        "tenant": principal.tenant,
        "groups": sorted(principal.groups),
        "fingerprint": acl_fingerprint(principal),
    }
