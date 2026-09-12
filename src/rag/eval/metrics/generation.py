from __future__ import annotations


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Token-overlap faithfulness proxy (swap for NLI judge in production)."""
    if not answer.strip() or not contexts:
        return 0.0
    answer_tokens = set(answer.lower().split())
    ctx_tokens: set[str] = set()
    for c in contexts:
        ctx_tokens.update(c.lower().split())
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & ctx_tokens) / len(answer_tokens)


def answer_relevance(answer: str, question: str) -> float:
    """Simple lexical relevance between answer and question."""
    if not answer.strip() or not question.strip():
        return 0.0
    a = set(answer.lower().split())
    q = set(question.lower().split())
    if not q:
        return 0.0
    return len(a & q) / len(q)
