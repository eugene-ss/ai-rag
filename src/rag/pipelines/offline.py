from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking import chunk_document
from rag.embedding.base import Embedder
from rag.embedding.hash_embedder import HashEmbedder
from rag.ingestion.local_fs import LocalFSConnector
from rag.lexical.base import LexicalIndex
from rag.lexical.bm25_memory import BM25MemoryIndex
from rag.observability.logging import get_logger
from rag.observability.tracing import reset_trace, span
from rag.parsing import parse_document
from rag.schemas import AclTags, Chunk, Document
from rag.security.pii import redact_pii
from rag.settings import Settings, get_settings
from rag.vectordb.base import VectorStore
from rag.vectordb.memory import MemoryVectorStore

log = get_logger("pipelines.offline")


@dataclass
class OfflinePipeline:
    """Ingest → Parse → Chunk → Embed → Index (never called from API handlers)."""

    vector_store: VectorStore
    lexical_index: LexicalIndex
    embedder: Embedder
    settings: Settings = field(default_factory=get_settings)

    def run(
        self,
        source: Path | str,
        *,
        index_version: str | None = None,
        acl: AclTags | None = None,
    ) -> list[Chunk]:
        trace_id = reset_trace()
        version = index_version or self.settings.index_version
        connector = LocalFSConnector(source)
        all_chunks: list[Chunk] = []
        chunking_cfg = self.settings.chunking_config()

        with span("offline_pipeline", index_version=version, trace_id=trace_id) as attrs:
            docs: list[Document] = []
            with span("ingest"):
                for uri in connector.list_uris():
                    doc = connector.fetch(uri, acl=acl)
                    docs.append(doc)
            attrs["documents"] = len(docs)

            parsed: list[Document] = []
            with span("parse"):
                for doc in docs:
                    parsed_doc = parse_document(doc)
                    if parsed_doc.text:
                        parsed_doc = parsed_doc.model_copy(
                            update={"text": redact_pii(parsed_doc.text)}
                        )
                    parsed.append(parsed_doc)

            with span("chunk"):
                for doc in parsed:
                    chunks = chunk_document(doc, index_version=version, config=chunking_cfg)
                    all_chunks.extend(chunks)
            attrs["chunks"] = len(all_chunks)

            with span("embed_index"):
                if all_chunks:
                    vectors = self.embedder.embed([c.text for c in all_chunks])
                    self.vector_store.upsert(all_chunks, vectors)
                    self.lexical_index.upsert(all_chunks)
                    self.vector_store.set_alias(self.settings.collection_alias, version)

            log.info(
                "offline complete docs=%s chunks=%s version=%s",
                len(docs),
                len(all_chunks),
                version,
            )
        return all_chunks


def default_offline_pipeline(
    *,
    vector_store: VectorStore | None = None,
    lexical_index: LexicalIndex | None = None,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> OfflinePipeline:
    return OfflinePipeline(
        vector_store=vector_store or MemoryVectorStore(),
        lexical_index=lexical_index or BM25MemoryIndex(),
        embedder=embedder or HashEmbedder(),
        settings=settings or get_settings(),
    )
