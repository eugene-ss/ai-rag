"""Blue/green read routing and ACL filter construction for the remote backends.

These are the two defects that a memory-backed test suite cannot see:

1. Promoting `v2` used to leave reads pointed at the physical `v1` collection
   while filtering for `index_version=v2`. Nothing matches, so a healthy cluster
   reports an empty corpus and every answer degrades to a refusal.
2. Qdrant's `min_should` is its own clause taking `conditions` + `min_count`,
   not an int paired with `should`. Passing an int raised `ValidationError` on
   every single search, so ACL filtering never ran at all.

Both are exercised through an injected fake client: no live cluster needed, and
the filter objects are still built by the real `qdrant_client` models, so their
validation runs for real.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from rag.retrieval.filters import AclFilter
from rag.schemas import Principal

TENANT = "acme"


def _require(module: str) -> None:
    """Import `module`, skipping locally but failing where CI expects the extra.

    A skip is a reasonable default on a laptop without the extras installed. In
    CI it is a blind spot: the suite goes green while the code paths that hold
    ACL enforcement and alias routing are never executed.
    """
    if os.environ.get("RAG_REQUIRE_BACKEND_EXTRAS") == "1":
        __import__(module)
        return
    pytest.importorskip(module)


@pytest.fixture
def principal() -> Principal:
    return Principal(subject="alice", tenant=TENANT, groups=frozenset({"engineering"}))


class _FakeQdrantClient:
    """Records the collection each call targeted."""

    def __init__(self) -> None:
        self.queried: list[str] = []
        self.counted: list[str] = []
        self.alias_ops: list[Any] = []

    def query_points(self, *, collection_name: str, **_: Any) -> Any:
        self.queried.append(collection_name)
        return type("Response", (), {"points": []})()

    def count(self, *, collection_name: str, exact: bool) -> Any:
        self.counted.append(collection_name)
        return type("Count", (), {"count": 7})()

    def update_collection_aliases(self, *, change_aliases_operations: list[Any]) -> None:
        self.alias_ops = change_aliases_operations


def _qdrant_store(client: Any, *, collection: str = "docs__v1") -> Any:
    _require("qdrant_client")
    from rag.vectordb.qdrant import QdrantVectorStore

    return QdrantVectorStore(collection=collection, dimensions=4, client=client)


def test_qdrant_reads_follow_the_promoted_version(principal: Principal) -> None:
    client = _FakeQdrantClient()
    store = _qdrant_store(client)

    # The store was built against v1, but the caller resolved v2 from the alias.
    store.search([0.0, 0.0, 0.0, 0.0], top_k=3, principal=principal, index_version="v2")
    store.count("v2")

    assert client.queried == ["docs__v2"]
    assert client.counted == ["docs__v2"]


def test_qdrant_writes_stay_on_the_version_being_built() -> None:
    client = _FakeQdrantClient()
    store = _qdrant_store(client)

    # No index_version means "the version this store owns" — the write target.
    assert store.collection_for() == "docs__v1"
    assert store.collection_for(None) == "docs__v1"
    assert store.collection_for("v9") == "docs__v9"


def test_qdrant_alias_swap_deletes_before_creating() -> None:
    client = _FakeQdrantClient()
    store = _qdrant_store(client)

    store.set_alias("docs_live", "v2")

    # A create-only swap fails when the alias already exists, which is exactly
    # the rollback case. Both operations must ship in one batch.
    assert len(client.alias_ops) == 2
    assert client.alias_ops[0].delete_alias.alias_name == "docs_live"
    assert client.alias_ops[1].create_alias.alias_name == "docs_live"
    assert client.alias_ops[1].create_alias.collection_name == "docs__v2"
    assert store.resolve_alias("docs_live") == "v2"


def test_qdrant_acl_filter_is_a_valid_min_should_clause(principal: Principal) -> None:
    store = _qdrant_store(_FakeQdrantClient())
    acl = AclFilter.for_principal(principal, index_version="v2")

    query_filter = store._acl_filter(acl)

    # Constructing the model is the regression: an int `min_should` raises here.
    assert query_filter.min_should.min_count == 1
    # Tenant-wide chunks (no groups) or a group the principal holds.
    assert len(query_filter.min_should.conditions) == 2
    must_keys = {condition.key for condition in query_filter.must}
    assert must_keys == {"tenant", "index_version"}


def test_qdrant_acl_filter_without_groups_still_matches_tenant_wide_chunks() -> None:
    store = _qdrant_store(_FakeQdrantClient())
    groupless = Principal(subject="bob", tenant=TENANT, groups=frozenset())
    acl = AclFilter.for_principal(groupless, index_version=None)

    query_filter = store._acl_filter(acl)

    # Only the is-empty alternative: a principal with no groups must not be
    # widened into a MatchAny over an empty list, which matches nothing.
    assert len(query_filter.min_should.conditions) == 1
    assert {condition.key for condition in query_filter.must} == {"tenant"}


class _FakeOpenSearchClient:
    def __init__(self) -> None:
        self.searched: list[str] = []
        self.counted: list[str] = []

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.searched.append(index)
        return {"hits": {"hits": []}}

    def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.counted.append(index)
        return {"count": 7}


def test_opensearch_reads_follow_the_promoted_version(principal: Principal) -> None:
    _require("opensearchpy")
    from rag.lexical.opensearch import OpenSearchLexicalIndex

    client = _FakeOpenSearchClient()
    index = OpenSearchLexicalIndex(index="docs__v1", client=client)

    index.search("hybrid retrieval", top_k=3, principal=principal, index_version="v2")

    assert client.searched == ["docs__v2"]
    assert index.index_for() == "docs__v1"
    assert index.index_for("v3") == "docs__v3"
