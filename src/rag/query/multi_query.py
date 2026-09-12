from __future__ import annotations

from rag.query.rewrite import rewrite_query


def expand_multi_query(query: str, *, n: int = 3) -> list[str]:
    """Produce up to n query variants for multi-query retrieval."""
    base = rewrite_query(query)
    if not base:
        return []
    primary = base[0]
    variants = list(base)
    extras = [
        f"information about {primary}",
        f"details on {primary}",
        f"explain {primary}",
    ]
    for e in extras:
        if e not in variants:
            variants.append(e)
        if len(variants) >= n:
            break
    return variants[:n]
