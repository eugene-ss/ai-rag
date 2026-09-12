from __future__ import annotations

from pathlib import Path

from rag.backends import BackendContext
from rag.eval.report import EvalReport, format_report
from rag.eval.runner import EvalRunner
from rag.generation.grounded import generate_grounded
from rag.observability.logging import get_logger
from rag.pipelines import online as online_pipeline
from rag.schemas import Answer, Principal, QueryResult

log = get_logger("jobs.eval")


def scheduled_eval(
    dataset: Path | str,
    *,
    context: BackendContext | None = None,
    principal: Principal | None = None,
) -> EvalReport:
    """Run golden-set evaluation against the live index.

    Runs as a job, not in the request path, so it can be scheduled nightly and
    gate a version promotion.
    """
    ctx = context or BackendContext.from_settings()
    pipeline = online_pipeline.from_context(ctx)
    who = principal or Principal(
        subject="scheduled-eval",
        tenant=ctx.settings.default_tenant,
        groups=frozenset({"public"}),
    )

    def retrieve_fn(question: str) -> QueryResult:
        return pipeline.retrieve_only(question, principal=who)

    def generate_fn(question: str, result: QueryResult) -> Answer:
        return generate_grounded(
            question=question,
            scored=result.results,
            llm=pipeline.llm,
            score_threshold=ctx.settings.refusal_score_threshold,
            redact=ctx.settings.pii_redaction_enabled,
        )

    report = EvalRunner(retrieve_fn=retrieve_fn, generate_fn=generate_fn).run(dataset)
    log.info(
        "eval complete n=%s recall@5=%.3f mrr=%.3f faithfulness=%.3f refusal_rate=%.3f",
        report.n,
        report.recall_at_5,
        report.mrr,
        report.faithfulness,
        report.refusal_rate,
    )
    return report


def format_eval(report: EvalReport) -> str:
    return format_report(report)
