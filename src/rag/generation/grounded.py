from __future__ import annotations

from rag.generation.base import LLMClient
from rag.generation.citations import resolve_citations
from rag.generation.refusal import refused_answer, should_refuse
from rag.observability.tracing import current_trace_id, span
from rag.prompts import get as get_prompt
from rag.schemas import Answer, ScoredChunk, Usage
from rag.security.pii import redact_pii


def format_context(scored: list[ScoredChunk]) -> str:
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
    score_threshold: float = 0.15,
    prompt_name: str = "answer_grounded",
    prompt_version: str = "v1",
) -> Answer:
    """Grounded generation with citations and explicit refusal."""
    trace_id = current_trace_id()
    with span("generate", prompt=f"{prompt_name}@{prompt_version}") as attrs:
        # Early refusal on empty / low score before calling the LLM
        early_refuse, early_reason = should_refuse(
            scored, citations=[], score_threshold=score_threshold
        )
        if early_refuse and early_reason in {"no_retrieved_context", "low_confidence_score"}:
            attrs["refused"] = True
            attrs["reason"] = early_reason
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
        usage.model = llm.model_name

        text = redact_pii(raw_text)
        citations = resolve_citations(text, scored)
        refuse, reason = should_refuse(scored, citations, score_threshold=score_threshold)
        if refuse:
            attrs["refused"] = True
            attrs["reason"] = reason
            return refused_answer(
                reason=reason or "unknown",
                usage=usage,
                trace_id=trace_id,
            )

        attrs["citations"] = len(citations)
        return Answer(
            text=text,
            citations=citations,
            refused=False,
            refusal_reason=None,
            usage=usage,
            trace_id=trace_id,
        )
