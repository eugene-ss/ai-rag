from __future__ import annotations

import re

from rag.utils.text import normalize_whitespace


def rewrite_query(query: str) -> list[str]:
    """Normalize and lightly expand a user query.

    Production systems swap this for an LLM rewrite prompt; the contract is a
    list of rewritten strings (primary first).
    """
    cleaned = normalize_whitespace(query)
    if not cleaned:
        return []
    variants = [cleaned]
    # Drop trailing question marks / filler for a secondary form
    alt = re.sub(r"[?!.]+$", "", cleaned).strip()
    if alt and alt.lower() != cleaned.lower():
        variants.append(alt)
    return variants
