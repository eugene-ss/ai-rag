from __future__ import annotations

from enum import StrEnum
from typing import Literal

QueryMode = Literal["auto", "fast", "agent"]


class QueryRoute(StrEnum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPLEX = "complex"
    CHITCHAT = "chitchat"
    REFUSE = "refuse"


_COMPLEX_MARKERS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "vs.",
    "difference between",
    "differences between",
    "both ",
    "relate",
    "relationship between",
    "how does",
    "how do ",
    "multi-hop",
    "and also",
    "on the other hand",
    "pros and cons",
    "trade-off",
    "tradeoff",
)


def route_query(query: str) -> QueryRoute:
    """Cheap heuristic router. Replace with classifier / LLM in production."""
    q = query.lower().strip()
    if not q:
        return QueryRoute.REFUSE
    greetings = ("hi", "hello", "hey", "thanks", "thank you")
    if q in greetings or q.startswith(("hi ", "hello ", "hey ")):
        return QueryRoute.CHITCHAT
    if any(marker in q for marker in _COMPLEX_MARKERS):
        return QueryRoute.COMPLEX
    if any(w in q for w in ("how do i", "how to", "steps to", "procedure")):
        return QueryRoute.PROCEDURAL
    return QueryRoute.FACTUAL


def should_use_agent(
    query: str,
    *,
    mode: QueryMode = "auto",
    agent_enabled: bool = True,
) -> bool:
    """Decide whether this request should enter the agent runtime.

    `fast` always stays on the deterministic online pipeline. `agent` forces the
    agent when it is enabled. `auto` only elevates COMPLEX routes so p50 cost
    and latency stay where they are today.
    """
    if not agent_enabled:
        return False
    if mode == "fast":
        return False
    if mode == "agent":
        return True
    return route_query(query) is QueryRoute.COMPLEX
