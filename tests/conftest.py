"""Shared fixtures.

Tests never touch real backends: every fixture wires in-memory implementations,
so the suite is hermetic and fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.backends import BackendContext
from rag.cache.memory import MemoryCache
from rag.cache.semantic import SemanticCache
from rag.embedding.hash_embedder import HashEmbedder
from rag.lexical.bm25_memory import BM25MemoryIndex
from rag.llm.echo import EchoLLM
from rag.rerank.identity import IdentityReranker
from rag.schemas import AclTags, Principal
from rag.settings import Settings
from rag.vectordb.memory import MemoryVectorStore

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
GOLDEN = Path(__file__).parent / "fixtures" / "golden.jsonl"

TENANT = "acme"
GROUP = "engineering"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Dev-mode settings with an isolated index registry per test."""
    return Settings(
        env="test",
        configs_root=Path("configs"),
        index_version="v1",
        collection_alias="docs_live",
        cache_enabled=False,
        semantic_cache_enabled=False,
        refusal_score_threshold=0.01,
        auth_trust_headers=True,
        default_tenant=TENANT,
        log_dir=None,
        index_registry_file=tmp_path / "index_versions.yaml",
    )


@pytest.fixture
def acl() -> AclTags:
    return AclTags(tenant=TENANT, allow_groups=frozenset({GROUP}))


@pytest.fixture
def principal() -> Principal:
    return Principal(subject="alice", tenant=TENANT, groups=frozenset({GROUP}))


@pytest.fixture
def outsider() -> Principal:
    return Principal(subject="bob", tenant="other-corp", groups=frozenset({GROUP}))


@pytest.fixture
def context(settings: Settings) -> BackendContext:
    """In-memory backend context shared by offline and online pipelines."""
    embedder = HashEmbedder(dimensions=settings.hash_embedding_dimensions)
    return BackendContext(
        settings=settings,
        embedder=embedder,
        vector_store=MemoryVectorStore(),
        lexical_index=BM25MemoryIndex(),
        reranker=IdentityReranker(),
        llm=EchoLLM(),
        cache=MemoryCache(),
        semantic_cache=SemanticCache(embedder=embedder),
    )
