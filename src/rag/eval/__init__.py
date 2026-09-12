"""Golden datasets, faithfulness, and Recall@K as first-class components."""

from rag.eval.datasets import GoldenExample, load_golden
from rag.eval.metrics import answer_relevance, faithfulness, mrr, ndcg_at_k, recall_at_k
from rag.eval.report import EvalReport, format_report
from rag.eval.runner import EvalRunner

__all__ = [
    "EvalReport",
    "EvalRunner",
    "GoldenExample",
    "answer_relevance",
    "faithfulness",
    "format_report",
    "load_golden",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]
