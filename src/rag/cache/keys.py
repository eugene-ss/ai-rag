from __future__ import annotations

import hashlib
import json
import re

from rag.schemas import Principal
from rag.security.acl import acl_fingerprint


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def cache_scope(
    *,
    principal: Principal,
    index_version: str,
    prompt_version: str,
    retrieval_params: dict[str, object] | None = None,
) -> str:
    """Everything a cache entry depends on except the query text itself.

    The semantic cache buckets entries by scope so a nearest-neighbour match can
    never cross an ACL, index version, or prompt version boundary.
    """
    payload = {
        "acl": acl_fingerprint(principal),
        "index_version": index_version,
        "prompt_version": prompt_version,
        "params": retrieval_params or {},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


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
    scope = cache_scope(
        principal=principal,
        index_version=index_version,
        prompt_version=prompt_version,
        retrieval_params=retrieval_params,
    )
    raw = f"{scope}|{normalize_query(query)}"
    return hashlib.sha256(raw.encode()).hexdigest()


def agent_cache_params(
    *,
    mode: str,
    allow_egress: bool,
    tool_names: list[str],
    budget: dict[str, object],
) -> dict[str, object]:
    """Params that must enter the answer-cache scope for agent turns.

    Without these, two agent configurations (different tools or budget caps)
    would collide on the same query key.
    """
    return {
        "mode": mode,
        "allow_egress": allow_egress,
        "tools": sorted(tool_names),
        "budget": budget,
    }
