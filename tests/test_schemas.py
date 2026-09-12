from __future__ import annotations

from rag.schemas import AclTags, Chunk, Document, Principal, derive_chunk_id


def test_derive_chunk_id_is_deterministic() -> None:
    a = derive_chunk_id("doc1", 0, "1.0", "v1")
    b = derive_chunk_id("doc1", 0, "1.0", "v1")
    c = derive_chunk_id("doc1", 1, "1.0", "v1")
    assert a == b
    assert a != c
    assert len(a) == 24


def test_document_and_chunk_roundtrip() -> None:
    acl = AclTags(tenant="acme", allow_groups=frozenset({"eng"}))
    doc = Document(
        doc_id="d1",
        source_uri="file:///tmp/d1.md",
        mime_type="text/markdown",
        checksum="abc",
        text="hello",
        acl=acl,
    )
    chunk = Chunk(
        chunk_id=derive_chunk_id(doc.doc_id, 0, "1.0", "v1"),
        doc_id=doc.doc_id,
        ordinal=0,
        text="hello",
        span=(0, 5),
        acl=acl,
        index_version="v1",
    )
    assert chunk.acl.tenant == "acme"
    principal = Principal(subject="u", tenant="acme", groups=frozenset({"eng"}))
    assert principal.tenant == chunk.acl.tenant
