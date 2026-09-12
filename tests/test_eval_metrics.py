from __future__ import annotations

from rag.eval.metrics import answer_relevance, faithfulness, mrr, ndcg_at_k, recall_at_k


def test_recall_at_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = ["c", "e"]
    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert recall_at_k(retrieved, relevant, 3) == 0.5


def test_mrr() -> None:
    assert mrr(["a", "b", "c"], ["c"]) == 1 / 3
    assert mrr(["a", "b"], ["z"]) == 0.0


def test_ndcg_at_k() -> None:
    score = ndcg_at_k(["a", "b", "c"], ["a", "c"], 3)
    assert 0.0 < score <= 1.0


def test_faithfulness_and_relevance() -> None:
    answer = "hybrid retrieval uses dense and lexical search"
    ctx = ["Hybrid retrieval combines dense vector search with lexical BM25."]
    assert faithfulness(answer, ctx) > 0.3
    assert answer_relevance(answer, "what is hybrid retrieval") > 0.2
