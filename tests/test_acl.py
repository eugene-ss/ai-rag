from __future__ import annotations

from rag.embedding.hash_embedder import HashEmbedder
from rag.lexical.bm25_memory import BM25MemoryIndex
from rag.retrieval.hybrid import HybridRetriever
from rag.schemas import AclTags, Chunk, Principal
from rag.security.acl import is_allowed
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
