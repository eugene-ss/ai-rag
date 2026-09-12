from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag.cache.base import Cache
from rag.cache.keys import cache_key
from rag.cache.memory import MemoryCache
from rag.embedding.base import Embedder
from rag.embedding.hash_embedder import HashEmbedder
from rag.generation.base import LLMClient
from rag.generation.echo import EchoLLM
from rag.generation.grounded import generate_grounded
from rag.lexical.base import LexicalIndex
from rag.lexical.bm25_memory import BM25MemoryIndex
from rag.observability.metrics import METRICS
from rag.observability.tracing import current_trace_id, reset_trace, span
from rag.query.rewrite import rewrite_query
from rag.query.route import QueryRoute, route_query
from rag.rerank.base import Reranker
from rag.rerank.identity import IdentityReranker
from rag.retrieval.hybrid import HybridRetriever
from rag.schemas import Answer, Principal, QueryResult
from rag.settings import Settings, get_settings
from rag.vectordb.base import VectorStore
from rag.vectordb.memory import MemoryVectorStore


@dataclass
class OnlinePipeline:
    """Query Rewrite → Hybrid Retrieve → Rerank → Generate → Cite → Refuse."""

    retriever: HybridRetriever
    reranker: Reranker
    llm: LLMClient
    cache: Cache
    settings: Settings = field(default_factory=get_settings)
    prompt_version: str = "v1"

    def answer(self, query: str, *, principal: Principal) -> Answer:
        trace_id = reset_trace()
        start = time.perf_counter()
        cfg = self.settings.retrieval_config()
        index_version = (
            self.retriever.vector_store.resolve_alias(self.settings.collection_alias)
            or self.settings.index_version
        )

        key = cache_key(
            query=query,
            principal=principal,
            index_version=index_version,
            prompt_version=self.prompt_version,
            retrieval_params={
                "top_k": cfg.get("top_k", self.settings.top_k),
                "rerank_top_k": cfg.get("rerank_top_k", self.settings.rerank_top_k),
            },
        )

        if self.settings.cache_enabled:
            cached = self.cache.get(key)
            if cached is not None:
                METRICS.incr("cache_hit")
                answer = Answer.model_validate_json(cached)
                return answer.model_copy(update={"trace_id": trace_id})

        with span("online_pipeline", trace_id=trace_id) as attrs:
            route = route_query(query)
            attrs["route"] = route.value
            if route == QueryRoute.REFUSE:
                return Answer(
                    text="I don't know based on the available documents.",
                    refused=True,
                    refusal_reason="empty_query",
                    trace_id=trace_id,
                )
            if route == QueryRoute.CHITCHAT:
                return Answer(
                    text="Hello! Ask me a question about the indexed documents.",
                    refused=False,
                    trace_id=trace_id,
                )

            rewritten = rewrite_query(query) if cfg.get("rewrite", True) else [query]
            primary = rewritten[0] if rewritten else query

            with span("retrieve"):
                scored = self.retriever.retrieve(
                    primary,
                    principal=principal,
                    top_k=int(cfg.get("top_k", self.settings.top_k)),
                    index_version=index_version,
                )

            with span("rerank"):
                scored = self.reranker.rerank(
                    primary,
                    scored,
                    top_k=int(cfg.get("rerank_top_k", self.settings.rerank_top_k)),
                )

            qr = QueryResult(
                query=query,
                rewritten=rewritten,
                results=scored,
                index_version=index_version,
                trace_id=trace_id,
                latency_ms=(time.perf_counter() - start) * 1000,
                cache_hit=False,
            )

            answer = generate_grounded(
                question=query,
                scored=qr.results,
                llm=self.llm,
                score_threshold=self.settings.refusal_score_threshold,
                prompt_version=self.prompt_version,
            )
            answer = answer.model_copy(update={"trace_id": current_trace_id()})

            if self.settings.cache_enabled and not answer.refused:
                self.cache.set(key, answer.model_dump_json(), ttl_seconds=3600)

            METRICS.observe("online_latency_ms", (time.perf_counter() - start) * 1000)
            attrs["refused"] = answer.refused
            attrs["citations"] = len(answer.citations)
            return answer

    def retrieve_only(self, query: str, *, principal: Principal) -> QueryResult:
        """Expose retrieval for eval without generation."""
        trace_id = reset_trace()
        start = time.perf_counter()
        cfg = self.settings.retrieval_config()
        index_version = (
            self.retriever.vector_store.resolve_alias(self.settings.collection_alias)
            or self.settings.index_version
        )
        rewritten = rewrite_query(query)
        primary = rewritten[0] if rewritten else query
        scored = self.retriever.retrieve(
            primary,
            principal=principal,
            top_k=int(cfg.get("top_k", self.settings.top_k)),
            index_version=index_version,
        )
        scored = self.reranker.rerank(
            primary,
            scored,
            top_k=int(cfg.get("rerank_top_k", self.settings.rerank_top_k)),
        )
        return QueryResult(
            query=query,
            rewritten=rewritten,
            results=scored,
            index_version=index_version,
            trace_id=trace_id,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def default_online_pipeline(
    *,
    vector_store: VectorStore | None = None,
    lexical_index: LexicalIndex | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> OnlinePipeline:
    settings = settings or get_settings()
    vs = vector_store or MemoryVectorStore()
    lx = lexical_index or BM25MemoryIndex()
    emb = embedder or HashEmbedder()
    retriever = HybridRetriever(
        vector_store=vs,
        lexical_index=lx,
        embedder=emb,
        rrf_k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        lexical_weight=settings.lexical_weight,
    )
    return OnlinePipeline(
        retriever=retriever,
        reranker=IdentityReranker(),
        llm=llm or EchoLLM(),
        cache=MemoryCache(),
        settings=settings,
    )
