from __future__ import annotations

import hashlib
import json
import re

from rag.schemas import Principal
from rag.security.acl import acl_fingerprint


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def cache_key(
    *,
    query: str,
    principal: Principal,
    index_version: str,
    prompt_version: str,
    retrieval_params: dict[str, object] | None = None,
) -> str:
    """Hash of (normalized query, ACL fingerprint, index_version, prompt_version, params).

    ACL fingerprint is what stops one tenant's answers leaking to another.
    """
    payload = {
        "q": normalize_query(query),
        "acl": acl_fingerprint(principal),
        "index_version": index_version,
        "prompt_version": prompt_version,
        "params": retrieval_params or {},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()
