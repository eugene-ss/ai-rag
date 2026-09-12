from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rag.backends import BackendContext
from rag.cache.base import Cache
from rag.cache.keys import cache_key, cache_scope
from rag.cache.semantic import SemanticCache
from rag.generation.grounded import generate_grounded
from rag.llm.base import LLMClient
from rag.llm.errors import LLMError
from rag.observability.logging import get_logger
from rag.observability.metrics import METRICS
from rag.observability.tracing import current_trace_id, span, start_trace
from rag.query.hyde import hyde_document
from rag.query.multi_query import expand_multi_query
from rag.query.rewrite import rewrite_query
from rag.query.route import QueryRoute, route_query
from rag.rerank.base import Reranker
from rag.retrieval.hybrid import HybridRetriever
from rag.schemas import Answer, Principal, QueryResult, Usage
from rag.settings import RetrievalConfig, Settings, get_settings

log = get_logger("pipelines.online")

REFUSAL_TEXT = "I don't know based on the available documents."


@dataclass
class OnlinePipeline:
    """Query Rewrite → Hybrid Retrieve → Rerank → Generate → Cite → Refuse.

    Read-only with respect to the index: it never writes, so it can never turn
    into an indexer running inside a request handler.
    """

    retriever: HybridRetriever
    reranker: Reranker
    llm: LLMClient
    cache: Cache
    settings: Settings = field(default_factory=get_settings)
    prompt_version: str = "v1"
    semantic_cache: SemanticCache | None = None
    fallback_index_version: str | None = None

    # --- public API --------------------------------------------------------

    def answer(self, query: str, *, principal: Principal) -> Answer:
        trace_id = start_trace()
        start = time.perf_counter()
        cfg = self.settings.retrieval_config()
        index_version = self.active_index_version()
        params = cfg.cache_params()

        key = cache_key(
            query=query,
            principal=principal,
            index_version=index_version,
            prompt_version=self.prompt_version,
            retrieval_params=params,
        )
        scope = cache_scope(
            principal=principal,
            index_version=index_version,
            prompt_version=self.prompt_version,
            retrieval_params=params,
        )

        cached = self._lookup_cache(key, scope, query)
        if cached is not None:
            METRICS.observe("online_latency_ms", (time.perf_counter() - start) * 1000)
            return cached.model_copy(update={"trace_id": trace_id, "cached": True})

        with span("online_pipeline", trace_id=trace_id) as attrs:
            route = route_query(query)
            attrs["route"] = route.value
            if route is QueryRoute.REFUSE:
                return self._refuse("empty_query", trace_id)
            if route is QueryRoute.CHITCHAT:
                return Answer(
                    text="Hello. Ask me a question about the indexed documents.",
                    trace_id=trace_id,
                    usage=Usage(model=self.llm.model_name, prompt_version=self.prompt_version),
                )

            result = self._retrieve(
                query, principal=principal, cfg=cfg, index_version=index_version
            )

            try:
                answer = generate_grounded(
                    question=query,
                    scored=result.results,
                    llm=self.llm,
                    score_threshold=self.settings.refusal_score_threshold,
                    prompt_version=self.prompt_version,
                    redact=self.settings.pii_redaction_enabled,
                )
            except LLMError as exc:
                # Every model and retry is spent. Refusing beats inventing.
                METRICS.incr("generation_unavailable")
                log.error("generation unavailable: %s", exc)
                attrs["error"] = "llm_unavailable"
                return self._refuse("llm_unavailable", trace_id)

            answer = answer.model_copy(
                update={"trace_id": current_trace_id(), "index_version": index_version}
            )

            if self._should_cache(answer):
                payload = answer.model_dump_json()
                self.cache.set(key, payload, ttl_seconds=self.settings.cache_ttl_seconds)
                if self.semantic_cache is not None:
                    self.semantic_cache.set(query, payload, scope=scope)

            latency = (time.perf_counter() - start) * 1000
            METRICS.observe("online_latency_ms", latency)
            METRICS.incr("answers", refused=str(answer.refused).lower())
            attrs["refused"] = answer.refused
            attrs["citations"] = len(answer.citations)
            return answer.model_copy(update={"latency_ms": latency})

    def retrieve_only(self, query: str, *, principal: Principal) -> QueryResult:
        """Retrieval without generation. Used by evaluation and debugging."""
        start_trace()
        cfg = self.settings.retrieval_config()
        return self._retrieve(
            query,
            principal=principal,
            cfg=cfg,
            index_version=self.active_index_version(),
        )

    def active_index_version(self) -> str:
        """Resolve the live index through the alias, never a hardcoded name."""
        resolved = self.retriever.vector_store.resolve_alias(self.settings.collection_alias)
        return resolved or self.fallback_index_version or self.settings.index_version

    def readiness(self) -> dict[str, Any]:
        """Report whether this replica can actually answer a query.

        An unresolved alias or an empty index means no: serving traffic then
        would return refusals that look like a corpus gap rather than a deploy
        that has not finished.
        """
        checks: dict[str, bool] = {}
        details: dict[str, Any] = {"collection_alias": self.settings.collection_alias}

        store = self.retriever.vector_store
        try:
            alias_version = store.resolve_alias(self.settings.collection_alias)
            checks["index_alias"] = bool(alias_version)
            details["index_version"] = alias_version or ""
        except Exception as exc:
            checks["index_alias"] = False
            details["index_error"] = str(exc)

        try:
            count = store.count()
            checks["index_populated"] = count > 0
            details["chunk_count"] = count
        except Exception as exc:
            checks["index_populated"] = False
            details["index_error"] = str(exc)

        ping = getattr(self.cache, "ping", None)
        if callable(ping):
            checks["cache"] = bool(ping())

        return {"ready": all(checks.values()), "checks": checks, **details}

    # --- internals ---------------------------------------------------------

    def _retrieve(
        self,
        query: str,
        *,
        principal: Principal,
        cfg: RetrievalConfig,
        index_version: str,
    ) -> QueryResult:
        start = time.perf_counter()
        trace_id = current_trace_id()

        variants = self._query_variants(query, cfg)
        dense_variants = list(variants)
        if cfg.hyde_enabled:
            # HyDE: embed a hypothetical answer, keep lexical on the real query.
            dense_variants = [hyde_document(v) or v for v in variants]

        with span("retrieve", variants=len(variants)):
            scored = self.retriever.retrieve_multi(
                variants,
                principal=principal,
                top_k=cfg.top_k,
                index_version=index_version,
                dense_queries=dense_variants,
            )

        with span("rerank", candidates=len(scored)):
            scored = self.reranker.rerank(
                variants[0],
                scored,
                top_k=cfg.rerank_top_k,
            )

        return QueryResult(
            query=query,
            rewritten=variants,
            results=scored,
            index_version=index_version,
            trace_id=trace_id,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def _query_variants(self, query: str, cfg: RetrievalConfig) -> list[str]:
        if cfg.multi_query_enabled:
            variants = expand_multi_query(query, n=cfg.multi_query_count)
        elif cfg.rewrite_enabled:
            variants = rewrite_query(query)
        else:
            variants = [query]
        return variants or [query]

    def _lookup_cache(self, key: str, scope: str, query: str) -> Answer | None:
        if not self.settings.cache_enabled:
            return None
        exact = self.cache.get(key)
        if exact is not None:
            METRICS.incr("cache_hit", kind="exact")
            return Answer.model_validate_json(exact)
        if self.semantic_cache is not None:
            near = self.semantic_cache.get(query, scope=scope)
            if near is not None:
                METRICS.incr("cache_hit", kind="semantic")
                return Answer.model_validate_json(near)
        METRICS.incr("cache_miss")
        return None

    def _should_cache(self, answer: Answer) -> bool:
        # Refusals are cheap to recompute and often transient (an index still
        # building, a model outage), so caching them would prolong an incident.
        return self.settings.cache_enabled and not answer.refused

    def _refuse(self, reason: str, trace_id: str) -> Answer:
        METRICS.incr("answers", refused="true")
        METRICS.incr("refusals", reason=reason)
        return Answer(
            text=REFUSAL_TEXT,
            refused=True,
            refusal_reason=reason,
            trace_id=trace_id,
            usage=Usage(model=self.llm.model_name, prompt_version=self.prompt_version),
        )


def default_online_pipeline(settings: Settings | None = None) -> OnlinePipeline:
    """Build the online pipeline from configuration."""
    settings = settings or get_settings()
    return from_context(BackendContext.from_settings(settings))


def from_context(context: BackendContext) -> OnlinePipeline:
    """Build the online pipeline from already-constructed backends."""
    settings = context.settings
    retriever = HybridRetriever(
        vector_store=context.vector_store,
        lexical_index=context.lexical_index,
        embedder=context.embedder,
        rrf_k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        lexical_weight=settings.lexical_weight,
    )
    return OnlinePipeline(
        retriever=retriever,
        reranker=context.reranker,
        llm=context.llm,
        cache=context.cache,
        settings=settings,
        semantic_cache=context.semantic_cache,
    )
