from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.eval.datasets import GoldenExample, load_golden
from rag.eval.metrics import answer_relevance, faithfulness, mrr, ndcg_at_k, recall_at_k
from rag.eval.report import EvalCaseResult, EvalReport
from rag.observability.tracing import reset_trace
from rag.schemas import Answer, QueryResult


@dataclass
class EvalRunner:
    """Runs golden-set evaluation against retrieve/generate callables."""

    retrieve_fn: Any
    generate_fn: Any
    k_values: tuple[int, ...] = (5, 10)

    def run(self, dataset: list[GoldenExample] | Path | str) -> EvalReport:
        examples = load_golden(dataset) if isinstance(dataset, (str, Path)) else dataset
        cases: list[EvalCaseResult] = []
        for ex in examples:
            # One trace per case so a bad example can be traced in isolation.
            reset_trace()
            qr: QueryResult = self.retrieve_fn(ex.question)
            retrieved_chunk_ids = [s.chunk.chunk_id for s in qr.results]
            retrieved_doc_ids = [s.chunk.doc_id for s in qr.results]
            relevant = ex.relevant_chunk_ids or ex.relevant_doc_ids
            retrieved = retrieved_chunk_ids if ex.relevant_chunk_ids else retrieved_doc_ids
            answer: Answer = self.generate_fn(ex.question, qr)
            contexts = [s.chunk.text for s in qr.results]
            cases.append(
                EvalCaseResult(
                    example_id=ex.id,
                    recall_at_5=recall_at_k(retrieved, relevant, 5),
                    recall_at_10=recall_at_k(retrieved, relevant, 10),
                    mrr=mrr(retrieved, relevant),
                    ndcg_at_10=ndcg_at_k(retrieved, relevant, 10),
                    faithfulness=faithfulness(answer.text, contexts),
                    answer_relevance=answer_relevance(answer.text, ex.question),
                    refused=answer.refused,
                )
            )
        return EvalReport.from_cases(cases)
