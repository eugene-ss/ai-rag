from __future__ import annotations

from enum import StrEnum


class QueryRoute(StrEnum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CHITCHAT = "chitchat"
    REFUSE = "refuse"


def route_query(query: str) -> QueryRoute:
    """Cheap heuristic router. Replace with classifier / LLM in production."""
    q = query.lower().strip()
    if not q:
        return QueryRoute.REFUSE
    greetings = ("hi", "hello", "hey", "thanks", "thank you")
    if q in greetings or q.startswith(("hi ", "hello ", "hey ")):
        return QueryRoute.CHITCHAT
    if any(w in q for w in ("how do i", "how to", "steps to", "procedure")):
        return QueryRoute.PROCEDURAL
    return QueryRoute.FACTUAL
