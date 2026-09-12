from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCaseResult:
    example_id: str
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    faithfulness: float
    answer_relevance: float
    refused: bool


@dataclass
class EvalReport:
    n: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    faithfulness: float
    answer_relevance: float
    refusal_rate: float
    cases: list[EvalCaseResult] = field(default_factory=list)

    @classmethod
    def from_cases(cls, cases: list[EvalCaseResult]) -> EvalReport:
        n = len(cases) or 1
        return cls(
            n=len(cases),
            recall_at_5=sum(c.recall_at_5 for c in cases) / n,
            recall_at_10=sum(c.recall_at_10 for c in cases) / n,
            mrr=sum(c.mrr for c in cases) / n,
            ndcg_at_10=sum(c.ndcg_at_10 for c in cases) / n,
            faithfulness=sum(c.faithfulness for c in cases) / n,
            answer_relevance=sum(c.answer_relevance for c in cases) / n,
            refusal_rate=sum(1 for c in cases if c.refused) / n,
            cases=cases,
        )


def format_report(report: EvalReport) -> str:
    return "\n".join(
        [
            f"EvalReport n={report.n}",
            f"  Recall@5={report.recall_at_5:.3f}  Recall@10={report.recall_at_10:.3f}",
            f"  MRR={report.mrr:.3f}  nDCG@10={report.ndcg_at_10:.3f}",
            f"  Faithfulness={report.faithfulness:.3f}  AnswerRel={report.answer_relevance:.3f}",
            f"  RefusalRate={report.refusal_rate:.3f}",
        ]
    )
