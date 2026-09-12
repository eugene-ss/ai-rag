from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.api.deps import init_state
from rag.cache.memory import MemoryCache
from rag.embedding.hash_embedder import HashEmbedder
from rag.lexical.bm25_memory import BM25MemoryIndex
from rag.llm.echo import EchoLLM
from rag.pipelines.offline import OfflinePipeline
from rag.pipelines.online import OnlinePipeline
from rag.rerank.identity import IdentityReranker
from rag.retrieval.hybrid import HybridRetriever
from rag.schemas import AclTags
from rag.settings import Settings
from rag.vectordb.memory import MemoryVectorStore

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture
def client() -> TestClient:
    """API wired to a pre-built index, mirroring production: serve, never index."""
    vs = MemoryVectorStore()
    lx = BM25MemoryIndex()
    emb = HashEmbedder()
    settings = Settings(cache_enabled=False, refusal_score_threshold=0.01)

    OfflinePipeline(
        vector_store=vs,
        lexical_index=lx,
        embedder=emb,
        settings=settings,
    ).run(FIXTURES, index_version="v1", acl=AclTags(tenant="acme", allow_groups=frozenset({"eng"})))

    pipeline = OnlinePipeline(
        retriever=HybridRetriever(vector_store=vs, lexical_index=lx, embedder=emb),
        reranker=IdentityReranker(),
        llm=EchoLLM(),
        cache=MemoryCache(),
        settings=settings,
    )
    init_state(pipeline=pipeline, settings=settings)
    return TestClient(create_app())


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_returns_grounded_answer(client: TestClient) -> None:
    resp = client.post(
        "/query",
        json={"query": "What is hybrid retrieval?"},
        headers={"x-tenant": "acme", "x-groups": "eng", "x-subject": "alice"},
    )
    assert resp.status_code == 200
    answer = resp.json()["answer"]
    assert answer["refused"] is False
    assert answer["citations"]
    assert answer["trace_id"]


def test_query_refuses_for_principal_without_access(client: TestClient) -> None:
    """Wrong tenant sees no chunks at all, so the only honest answer is refusal."""
    resp = client.post(
        "/query",
        json={"query": "What is hybrid retrieval?"},
        headers={"x-tenant": "other-corp", "x-groups": "eng", "x-subject": "bob"},
    )
    assert resp.status_code == 200
    answer = resp.json()["answer"]
    assert answer["refused"] is True
    assert answer["refusal_reason"] == "no_retrieved_context"
    assert answer["citations"] == []


def test_query_rejects_empty_body(client: TestClient) -> None:
    resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 422
