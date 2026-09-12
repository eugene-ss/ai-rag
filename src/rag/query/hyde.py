from __future__ import annotations


def hyde_document(query: str) -> str:
    """Hypothetical Document Embedding (HyDE) stub.

    Returns a synthetic passage that can be embedded instead of the raw query.
    Swap for an LLM-generated passage in production.
    """
    cleaned = query.strip()
    if not cleaned:
        return ""
    return (
        f"This document answers the question: {cleaned}. "
        f"It provides factual background, definitions, and supporting details "
        f"relevant to: {cleaned}."
    )
