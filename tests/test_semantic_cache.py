from __future__ import annotations

from rag.cache.keys import cache_scope
from rag.cache.semantic import SemanticCache
from rag.embedding.hash_embedder import HashEmbedder
from rag.schemas import Principal


def _scope(principal: Principal, *, index_version: str = "v1") -> str:
    return cache_scope(
        principal=principal,
        index_version=index_version,
        prompt_version="v1",
    )


def test_exact_repeat_hits_semantic_cache() -> None:
    cache = SemanticCache(embedder=HashEmbedder(), threshold=0.95)
    alice = Principal(subject="alice", tenant="acme", groups=frozenset({"eng"}))
    scope = _scope(alice)

    assert cache.get("what is hybrid retrieval?", scope=scope) is None
    cache.set("what is hybrid retrieval?", "cached-answer", scope=scope)
    # Whitespace/case differences normalize to the same entry.
    assert cache.get("  What is Hybrid Retrieval?  ", scope=scope) == "cached-answer"


def test_semantic_cache_never_crosses_tenants() -> None:
    cache = SemanticCache(embedder=HashEmbedder(), threshold=0.0)
    alice = Principal(subject="alice", tenant="acme", groups=frozenset({"eng"}))
    bob = Principal(subject="bob", tenant="beta", groups=frozenset({"eng"}))

    cache.set("payroll numbers", "acme-secret", scope=_scope(alice))

    # Threshold 0.0 would match anything *within* a scope; a different tenant
    # lands in a different bucket, so there is nothing to match against.
    assert cache.get("payroll numbers", scope=_scope(bob)) is None
    assert cache.get("payroll numbers", scope=_scope(alice)) == "acme-secret"


def test_semantic_cache_never_crosses_groups_or_index_versions() -> None:
    cache = SemanticCache(embedder=HashEmbedder(), threshold=0.0)
    eng = Principal(subject="alice", tenant="acme", groups=frozenset({"eng"}))
    hr = Principal(subject="alice", tenant="acme", groups=frozenset({"hr"}))

    cache.set("q", "eng-answer", scope=_scope(eng))
    assert cache.get("q", scope=_scope(hr)) is None
    assert cache.get("q", scope=_scope(eng, index_version="v2")) is None


def test_below_threshold_is_a_miss() -> None:
    cache = SemanticCache(embedder=HashEmbedder(), threshold=0.999)
    alice = Principal(subject="alice", tenant="acme", groups=frozenset({"eng"}))
    scope = _scope(alice)
    cache.set("hybrid retrieval", "answer", scope=scope)
    # HashEmbedder gives unrelated strings near-orthogonal vectors.
    assert cache.get("completely different question", scope=scope) is None


def test_eviction_caps_bucket_size() -> None:
    cache = SemanticCache(embedder=HashEmbedder(), threshold=0.95, max_entries_per_scope=2)
    alice = Principal(subject="alice", tenant="acme", groups=frozenset({"eng"}))
    scope = _scope(alice)
    for i in range(5):
        cache.set(f"query {i}", f"answer {i}", scope=scope)
    assert cache.size(scope) <= 2
