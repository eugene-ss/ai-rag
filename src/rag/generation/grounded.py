from __future__ import annotations

from rag.generation.citations import resolve_citations
from rag.generation.refusal import refused_answer, should_refuse
from rag.llm.base import LLMClient
from rag.observability.metrics import METRICS
from rag.observability.tracing import current_trace_id, span
from rag.prompts import get as get_prompt
from rag.schemas import Answer, ScoredChunk, Usage
from rag.security.pii import redact_pii


def format_context(scored: list[ScoredChunk]) -> str:
    """Render retrieved chunks with ids the model can cite."""
    parts: list[str] = []
    for s in scored:
        parts.append(
            f"[chunk_id={s.chunk.chunk_id}] doc={s.chunk.doc_id} score={s.score:.4f}\n"
            f"{s.chunk.text}"
        )
    return "\n\n".join(parts)


def generate_grounded(
    *,
    question: str,
    scored: list[ScoredChunk],
    llm: LLMClient,
    score_threshold: float = 0.01,
    prompt_name: str = "answer_grounded",
    prompt_version: str = "v1",
    redact: bool = True,
) -> Answer:
    """Grounded generation with citations and explicit refusal.

    Raises whatever the LLM layer raises when generation is impossible; the
    caller decides whether that becomes a refusal or an error response.
    """
    trace_id = current_trace_id()
    with span("generate", prompt=f"{prompt_name}@{prompt_version}") as attrs:
        # Refuse before spending a token when retrieval found nothing usable.
        early_refuse, early_reason = should_refuse(
            scored, citations=[], score_threshold=score_threshold
        )
        if early_refuse and early_reason in {"no_retrieved_context", "low_confidence_score"}:
            attrs["refused"] = True
            attrs["reason"] = early_reason
            METRICS.incr("refusals", reason=early_reason or "unknown")
            return refused_answer(
                reason=early_reason or "unknown",
                usage=Usage(model=llm.model_name, prompt_version=prompt_version),
                trace_id=trace_id,
            )

        template = get_prompt(prompt_name, prompt_version)
        prompt = template.render(
            question=question,
            context=format_context(scored),
            prompt_version=prompt_version,
        )
        raw_text, usage = llm.complete(prompt)
        usage.prompt_version = prompt_version

        text = redact_pii(raw_text) if redact else raw_text
        citations = resolve_citations(text, scored)
        refuse, reason = should_refuse(scored, citations, score_threshold=score_threshold)
        if refuse:
            attrs["refused"] = True
            attrs["reason"] = reason
            METRICS.incr("refusals", reason=reason or "unknown")
            return refused_answer(
                reason=reason or "unknown",
                usage=usage,
                trace_id=trace_id,
            )

        METRICS.observe("answer_citations", len(citations))
        METRICS.observe("llm_cost_usd", usage.cost_usd)
        METRICS.observe("llm_total_tokens", usage.total_tokens)
        attrs["citations"] = len(citations)
        attrs["cost_usd"] = usage.cost_usd
        return Answer(
            text=text,
            citations=citations,
            refused=False,
            refusal_reason=None,
            usage=usage,
            trace_id=trace_id,
        )
