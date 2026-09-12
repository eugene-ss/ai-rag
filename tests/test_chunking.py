from __future__ import annotations

from rag.chunking import chunk_document
from rag.chunking.fixed import FixedChunker
from rag.schemas import AclTags, Document


def _doc(text: str, mime: str = "text/plain") -> Document:
    return Document(
        doc_id="doc",
        source_uri="file:///doc.txt",
        mime_type=mime,
        checksum="x",
        text=text,
        acl=AclTags(tenant="default", allow_groups=frozenset({"public"})),
    )


def test_fixed_chunker_respects_size_and_overlap() -> None:
    text = "a" * 1000
    chunks = FixedChunker().chunk(_doc(text), index_version="v1", size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    again = FixedChunker().chunk(_doc(text), index_version="v1", size=200, overlap=50)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]


def test_registry_selects_markdown_aware() -> None:
    doc = _doc("# Title\n\nBody paragraph.\n\n## Sec\n\nMore.", mime="text/markdown")
    chunks = chunk_document(
        doc,
        index_version="v1",
        config={
            "strategies": {
                "text/markdown": {
                    "strategy": "markdown_aware",
                    "size": 800,
                    "overlap": 80,
                }
            }
        },
    )
    assert chunks
    assert chunks[0].metadata["chunker"] == "markdown_aware"
