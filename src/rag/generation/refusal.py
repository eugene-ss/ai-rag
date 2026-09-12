from __future__ import annotations

from rag.schemas import Answer, Citation, ScoredChunk, Usage


def should_refuse(
    scored: list[ScoredChunk],
    citations: list[Citation],
    *,
    score_threshold: float,
) -> tuple[bool, str | None]:
    """Refuse when top score is too low or no citations resolve."""
    if not scored:
        return True, "no_retrieved_context"
    if scored[0].score < score_threshold:
        return True, "low_confidence_score"
    if not citations:
        return True, "zero_resolvable_citations"
    return False, None


def refused_answer(
    *,
    reason: str,
    usage: Usage,
    trace_id: str,
) -> Answer:
    """Refusal is a normal return value, not an exception."""
    return Answer(
        text="I don't know based on the available documents.",
        citations=[],
        refused=True,
        refusal_reason=reason,
        usage=usage,
        trace_id=trace_id,
    )
