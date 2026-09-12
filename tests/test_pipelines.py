from __future__ import annotations

from pathlib import Path

from rag.cache.memory import MemoryCache
from rag.embedding.hash_embedder import HashEmbedder
from rag.generation.echo import EchoLLM
from rag.lexical.bm25_memory import BM25MemoryIndex
from rag.pipelines.offline import OfflinePipeline
from rag.pipelines.online import OnlinePipeline
from rag.rerank.identity import IdentityReranker
from rag.retrieval.hybrid import HybridRetriever
from rag.schemas import Principal
from rag.settings import Settings
from rag.vectordb.memory import MemoryVectorStore

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


def test_offline_then_online_smoke() -> None:
    vs = MemoryVectorStore()
    lx = BM25MemoryIndex()
    emb = HashEmbedder()
    settings = Settings(
        index_version="v1",
        collection_alias="docs_live",
        cache_enabled=False,
        refusal_score_threshold=0.01,
    )

    offline = OfflinePipeline(vector_store=vs, lexical_index=lx, embedder=emb, settings=settings)
    chunks = offline.run(FIXTURES, index_version="v1")
    assert len(chunks) >= 3
    assert vs.resolve_alias("docs_live") == "v1"

    online = OnlinePipeline(
        retriever=HybridRetriever(vector_store=vs, lexical_index=lx, embedder=emb),
        reranker=IdentityReranker(),
        llm=EchoLLM(),
        cache=MemoryCache(),
        settings=settings,
    )
    principal = Principal(subject="tester", tenant="default", groups=frozenset({"public"}))
    answer = online.answer("What is hybrid retrieval?", principal=principal)
    assert answer.trace_id
    assert answer.usage.prompt_version == "v1"
    if not answer.refused:
        assert answer.citations
        assert "hybrid" in answer.text.lower() or answer.citations


def test_refusal_on_empty_index() -> None:
    vs = MemoryVectorStore()
    lx = BM25MemoryIndex()
    emb = HashEmbedder()
    settings = Settings(cache_enabled=False, refusal_score_threshold=0.01)
    online = OnlinePipeline(
        retriever=HybridRetriever(vector_store=vs, lexical_index=lx, embedder=emb),
        reranker=IdentityReranker(),
        llm=EchoLLM(),
        cache=MemoryCache(),
        settings=settings,
    )
    principal = Principal(subject="tester", tenant="default", groups=frozenset({"public"}))
    answer = online.answer("totally unknown topic xyzzy", principal=principal)
    assert answer.refused is True
