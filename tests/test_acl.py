from __future__ import annotations

from rag.embedding.hash_embedder import HashEmbedder
from rag.lexical.bm25_memory import BM25MemoryIndex
from rag.retrieval.hybrid import HybridRetriever
from rag.schemas import AclTags, Chunk, Principal
from rag.security.acl import acl_fingerprint, is_allowed
from rag.vectordb.memory import MemoryVectorStore


def _chunk(doc_id: str, text: str, tenant: str, groups: set[str]) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-0",
        doc_id=doc_id,
        ordinal=0,
        text=text,
        span=(0, len(text)),
        acl=AclTags(tenant=tenant, allow_groups=frozenset(groups)),
        index_version="v1",
    )


def test_is_allowed_tenant_and_groups() -> None:
    principal = Principal(subject="u", tenant="acme", groups=frozenset({"eng"}))
    ok = AclTags(tenant="acme", allow_groups=frozenset({"eng", "ops"}))
    bad_tenant = AclTags(tenant="other", allow_groups=frozenset({"eng"}))
    bad_group = AclTags(tenant="acme", allow_groups=frozenset({"finance"}))
    assert is_allowed(principal, ok)
    assert not is_allowed(principal, bad_tenant)
    assert not is_allowed(principal, bad_group)


def test_cache_fingerprint_matches_what_the_acl_rule_actually_reads() -> None:
    """The cache scope must contain exactly the inputs `is_allowed` consults.

    Include less and the cache leaks across an ACL boundary. Include more and it
    fragments for no safety benefit: `subject` used to be part of the key, so a
    thousand engineers in one tenant kept a thousand private copies of results
    they were all equally entitled to, and the hit rate approached zero.

    If this test fails because ACL evaluation became subject-dependent, the
    fingerprint has to grow in the same change — not this assertion.
    """
    acl = AclTags(tenant="acme", allow_groups=frozenset({"eng"}))
    alice = Principal(subject="alice", tenant="acme", groups=frozenset({"eng"}))
    bob = Principal(subject="bob", tenant="acme", groups=frozenset({"eng"}))

    # Same reach, so they must share a cache entry.
    assert is_allowed(alice, acl) == is_allowed(bob, acl)
    assert acl_fingerprint(alice) == acl_fingerprint(bob)

    # Anything that changes reach must change the scope.
    other_tenant = Principal(subject="alice", tenant="beta", groups=frozenset({"eng"}))
    other_groups = Principal(subject="alice", tenant="acme", groups=frozenset({"finance"}))
    extra_group = Principal(subject="alice", tenant="acme", groups=frozenset({"eng", "hr"}))

    assert acl_fingerprint(other_tenant) != acl_fingerprint(alice)
    assert acl_fingerprint(other_groups) != acl_fingerprint(alice)
    assert acl_fingerprint(extra_group) != acl_fingerprint(alice)

    # Group order is not reach.
    unordered = Principal(subject="alice", tenant="acme", groups=frozenset({"hr", "eng"}))
    assert acl_fingerprint(unordered) == acl_fingerprint(extra_group)


def test_acl_isolation_on_both_retrieval_paths() -> None:
    vs = MemoryVectorStore()
    lx = BM25MemoryIndex()
    emb = HashEmbedder()

    public = _chunk("pub", "public payroll policy overview", "acme", {"public"})
    secret = _chunk("sec", "secret payroll compensation numbers", "acme", {"hr"})
    other = _chunk("oth", "other tenant payroll notes", "beta", {"public"})

    chunks = [public, secret, other]
    vectors = emb.embed([c.text for c in chunks])
    vs.upsert(chunks, vectors)
    lx.upsert(chunks)

    principal = Principal(subject="u", tenant="acme", groups=frozenset({"public"}))
    retriever = HybridRetriever(vector_store=vs, lexical_index=lx, embedder=emb)

    query_vec = emb.embed_query("payroll")
    dense = vs.search(query_vec, top_k=10, principal=principal, index_version="v1")
    lexical = lx.search("payroll", top_k=10, principal=principal, index_version="v1")
    fused = retriever.retrieve("payroll", principal=principal, top_k=10, index_version="v1")

    dense_ids = {s.chunk.chunk_id for s in dense}
    lex_ids = {s.chunk.chunk_id for s in lexical}
    fused_ids = {s.chunk.chunk_id for s in fused}

    assert "pub-0" in dense_ids
    assert "sec-0" not in dense_ids
    assert "oth-0" not in dense_ids

    assert "pub-0" in lex_ids
    assert "sec-0" not in lex_ids
    assert "oth-0" not in lex_ids

    assert "sec-0" not in fused_ids
    assert "oth-0" not in fused_ids
