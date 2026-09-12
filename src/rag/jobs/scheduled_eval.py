from __future__ import annotations

from pathlib import Path

from rag.cache.memory import MemoryCache
from rag.eval.report import format_report
from rag.eval.runner import EvalRunner
from rag.generation.echo import EchoLLM
from rag.generation.grounded import generate_grounded
from rag.jobs.reindex import EMBEDDER, LEXICAL_INDEX, VECTOR_STORE
from rag.observability.logging import get_logger
from rag.pipelines.online import OnlinePipeline
from rag.rerank.identity import IdentityReranker
from rag.retrieval.hybrid import HybridRetriever
from rag.schemas import Principal
from rag.settings import get_settings

log = get_logger("jobs.eval")


def scheduled_eval(dataset: Path | str) -> str:
    """Run golden-set evaluation against the current in-process indexes."""
    settings = get_settings()
    retriever = HybridRetriever(
        vector_store=VECTOR_STORE,
        lexical_index=LEXICAL_INDEX,
        embedder=EMBEDDER,
        rrf_k=settings.rrf_k,
    )
    pipeline = OnlinePipeline(
        retriever=retriever,
        reranker=IdentityReranker(),
        llm=EchoLLM(),
        cache=MemoryCache(),
        settings=settings,
    )
    principal = Principal(
        subject="eval",
        tenant="default",
        groups=frozenset({"public"}),
    )

    def retrieve_fn(question: str):  # type: ignore[no-untyped-def]
        return pipeline.retrieve_only(question, principal=principal)

    def generate_fn(question: str, qr):  # type: ignore[no-untyped-def]
        return generate_grounded(
            question=question,
            scored=qr.results,
            llm=pipeline.llm,
            score_threshold=settings.refusal_score_threshold,
        )

    runner = EvalRunner(retrieve_fn=retrieve_fn, generate_fn=generate_fn)
    report = runner.run(dataset)
    text = format_report(report)
    log.info("\n%s", text)
    return text
