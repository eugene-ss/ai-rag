"""Settings-driven construction of every pluggable backend.

One place decides which concrete implementation each Protocol gets, so the API
process and the offline jobs are wired identically from the same configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag.cache.base import Cache
from rag.cache.memory import MemoryCache
from rag.cache.semantic import SemanticCache
from rag.embedding.base import Embedder
from rag.embedding.hash_embedder import HashEmbedder
from rag.lexical.base import LexicalIndex
from rag.lexical.bm25_memory import BM25MemoryIndex
from rag.llm.base import LLMClient
from rag.llm.echo import EchoLLM
from rag.llm.policy import RetryPolicy
from rag.llm.resilient import ResilientLLM
from rag.observability.logging import get_logger
from rag.rerank.base import Reranker
from rag.rerank.identity import IdentityReranker
from rag.settings import Settings, get_settings
from rag.vectordb.base import VectorStore
from rag.vectordb.memory import MemoryVectorStore

if TYPE_CHECKING:
    from rag.agent.runtime import AgentRuntime
    from rag.agent.tools import ToolRegistry
    from rag.llm.chat import ChatLLM
    from rag.pipelines.online import OnlinePipeline

log = get_logger("backends")


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "openai":
        from rag.embedding.openai import OpenAIEmbedder

        return OpenAIEmbedder(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            api_key=settings.openai_api_key,
        )
    return HashEmbedder(dimensions=settings.hash_embedding_dimensions)


def build_vector_store(settings: Settings, *, index_version: str | None = None) -> VectorStore:
    if settings.vector_backend == "qdrant":
        from rag.vectordb.qdrant import QdrantVectorStore

        return QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=settings.collection_for(index_version),
            dimensions=_vector_dimensions(settings),
        )
    return MemoryVectorStore()


def build_lexical_index(settings: Settings, *, index_version: str | None = None) -> LexicalIndex:
    if settings.lexical_backend == "opensearch":
        from rag.lexical.opensearch import OpenSearchLexicalIndex

        return OpenSearchLexicalIndex(
            url=settings.opensearch_url,
            index=settings.collection_for(index_version),
        )
    return BM25MemoryIndex()


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "cross_encoder":
        from rag.rerank.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker()
    return IdentityReranker()


def build_llm(settings: Settings) -> LLMClient:
    """Primary client plus fallbacks, wrapped in the shared resilience policy."""
    policy = RetryPolicy(
        max_attempts=settings.llm_max_attempts,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    if settings.llm_backend == "openai":
        from rag.llm.openai import OpenAIChatClient

        def make(model: str) -> LLMClient:
            return OpenAIChatClient(
                model=model,
                api_key=settings.openai_api_key,
                timeout=settings.llm_timeout_seconds,
                max_output_tokens=settings.llm_max_output_tokens,
                temperature=settings.llm_temperature,
            )

        return ResilientLLM(
            make(settings.llm_model),
            fallbacks=[make(m) for m in settings.llm_fallback_models],
            policy=policy,
        )
    return ResilientLLM(EchoLLM(), policy=policy)


def build_chat_llm(settings: Settings) -> ChatLLM:
    """Async chat model used by the agent planner and critic."""
    from rag.llm.echo_chat import EchoChatLLM
    from rag.llm.resilient_chat import ResilientChatLLM

    policy = RetryPolicy(
        max_attempts=settings.llm_max_attempts,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    if settings.chat_llm_backend == "openai":
        from rag.llm.openai_chat import OpenAIChatLLM

        def make(model: str) -> ChatLLM:
            return OpenAIChatLLM(
                model=model,
                api_key=settings.openai_api_key,
                timeout=settings.llm_timeout_seconds,
                max_output_tokens=settings.llm_max_output_tokens,
                temperature=settings.llm_temperature,
            )

        return ResilientChatLLM(
            make(settings.llm_model),
            fallbacks=[make(m) for m in settings.llm_fallback_models],
            policy=policy,
        )
    return ResilientChatLLM(EchoChatLLM(), policy=policy)


def build_tool_registry(
    settings: Settings,
    pipeline: OnlinePipeline,
    *,
    cache: Cache | None = None,
) -> ToolRegistry:
    from rag.agent.tools import (
        GraphQueryTool,
        RetrievalTool,
        ToolRegistry,
        ToolResultCache,
        WebSearchTool,
    )

    result_cache = None
    if cache is not None and settings.agent_tool_cache_enabled:
        result_cache = ToolResultCache(
            cache,
            ttl_seconds=settings.agent_tool_cache_ttl_seconds,
            enabled=True,
        )
    return ToolRegistry(
        [RetrievalTool(pipeline), WebSearchTool(), GraphQueryTool()],
        result_cache=result_cache,
    )


def build_agent_runtime(
    settings: Settings,
    pipeline: OnlinePipeline,
    *,
    chat_llm: ChatLLM | None = None,
    tools: ToolRegistry | None = None,
    cache: Cache | None = None,
) -> AgentRuntime | None:
    """Build the agent subsystem, or None when disabled."""
    if not settings.agent_enabled:
        return None
    from rag.agent import AgentRuntime, Critic

    chat = chat_llm or build_chat_llm(settings)
    registry = tools or build_tool_registry(settings, pipeline, cache=cache or pipeline.cache)
    return AgentRuntime(
        chat_llm=chat,
        tools=registry,
        pipeline=pipeline,
        critic=Critic(chat),
        budget=settings.agent_budget(),
        allow_egress=settings.agent_allow_egress,
        include_trace=settings.agent_include_trace,
    )


def build_cache(settings: Settings) -> Cache:
    if settings.cache_backend == "redis":
        from rag.cache.redis import RedisCache

        return RedisCache(url=settings.redis_url)
    return MemoryCache()


def build_semantic_cache(settings: Settings, embedder: Embedder) -> SemanticCache | None:
    if not (settings.cache_enabled and settings.semantic_cache_enabled):
        return None
    return SemanticCache(embedder=embedder, threshold=settings.semantic_cache_threshold)


def _vector_dimensions(settings: Settings) -> int:
    if settings.embedding_backend == "openai":
        return settings.embedding_dimensions
    return settings.hash_embedding_dimensions


@dataclass
class BackendContext:
    """All backends for one process, built once and shared.

    Holding these together matters for the in-memory backends: a single process
    must reuse the same store instances or an index built by `ingest` would be
    invisible to `eval`.
    """

    settings: Settings
    embedder: Embedder
    vector_store: VectorStore
    lexical_index: LexicalIndex
    reranker: Reranker
    llm: LLMClient
    cache: Cache
    semantic_cache: SemanticCache | None

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        index_version: str | None = None,
    ) -> BackendContext:
        settings = settings or get_settings()
        embedder = build_embedder(settings)
        context = cls(
            settings=settings,
            embedder=embedder,
            vector_store=build_vector_store(settings, index_version=index_version),
            lexical_index=build_lexical_index(settings, index_version=index_version),
            reranker=build_reranker(settings),
            llm=build_llm(settings),
            cache=build_cache(settings),
            semantic_cache=build_semantic_cache(settings, embedder),
        )
        log.info(
            "backends: vector=%s lexical=%s embedding=%s llm=%s cache=%s rerank=%s",
            settings.vector_backend,
            settings.lexical_backend,
            settings.embedding_backend,
            settings.llm_backend,
            settings.cache_backend,
            settings.reranker,
        )
        return context
