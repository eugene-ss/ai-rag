from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag.backends import BackendContext
from rag.chunking import chunk_document
from rag.embedding.base import Embedder
from rag.index_registry import IndexRegistry, IndexVersion
from rag.ingestion.local_fs import LocalFSConnector
from rag.lexical.base import LexicalIndex
from rag.observability.logging import get_logger
from rag.observability.metrics import METRICS
from rag.observability.tracing import span, start_trace
from rag.parsing import parse_document
from rag.schemas import AclTags, Chunk, Document
from rag.security.pii import redact_pii
from rag.settings import Settings, get_settings
from rag.vectordb.base import VectorStore

log = get_logger("pipelines.offline")


@dataclass
class OfflinePipeline:
    """Ingest → Parse → Chunk → Embed → Index.

    Runs only from `jobs/`. Nothing in `api/` may import or call it.
    """

    vector_store: VectorStore
    lexical_index: LexicalIndex
    embedder: Embedder
    settings: Settings = field(default_factory=get_settings)
    registry: IndexRegistry | None = None

    def run(
        self,
        source: Path | str,
        *,
        index_version: str | None = None,
        acl: AclTags | None = None,
        activate: bool = True,
    ) -> list[Chunk]:
        """Build one index version end to end, then publish it via the alias."""
        trace_id = start_trace()
        version = index_version or self.settings.index_version
        registry = self.registry or IndexRegistry(self.settings.index_registry_path())
        connector = LocalFSConnector(source)
        chunking_cfg = self.settings.chunking_config()
        all_chunks: list[Chunk] = []

        with span("offline_pipeline", index_version=version, trace_id=trace_id) as attrs:
            registry.record(
                IndexVersion(
                    version=version,
                    collection=self.settings.collection_for(version),
                    alias=self.settings.collection_alias,
                    status="building",
                    embedding_model=self.embedder.model_name,
                )
            )

            docs = self._ingest(connector, acl)
            attrs["documents"] = len(docs)

            parsed = self._parse(docs)
            attrs["parsed"] = len(parsed)

            with span("chunk"):
                for doc in parsed:
                    all_chunks.extend(
                        chunk_document(doc, index_version=version, config=chunking_cfg)
                    )
            attrs["chunks"] = len(all_chunks)

            if all_chunks:
                all_chunks = self._embed_and_index(all_chunks, version)

            registry.record(
                IndexVersion(
                    version=version,
                    collection=self.settings.collection_for(version),
                    alias=self.settings.collection_alias,
                    status="built",
                    chunk_count=len(all_chunks),
                    embedding_model=self.embedder.model_name,
                    chunker_version=str(chunking_cfg.get("chunker_version", "1.0")),
                )
            )

            if activate and all_chunks:
                self.activate(version, registry=registry)

            METRICS.incr("indexed_documents", value=len(docs))
            METRICS.incr("indexed_chunks", value=len(all_chunks))
            log.info(
                "offline complete docs=%s chunks=%s version=%s activated=%s",
                len(docs),
                len(all_chunks),
                version,
                activate and bool(all_chunks),
            )
        return all_chunks

    def activate(self, version: str, *, registry: IndexRegistry | None = None) -> None:
        """Point the alias at a built version. This is the atomic cutover."""
        registry = registry or IndexRegistry(self.settings.index_registry_path())
        self.vector_store.set_alias(self.settings.collection_alias, version)
        try:
            registry.activate(version)
        except KeyError:
            log.warning("version %s missing from registry; alias flipped anyway", version)
        log.info("alias %s -> %s", self.settings.collection_alias, version)

    # --- stages ------------------------------------------------------------

    def _ingest(self, connector: LocalFSConnector, acl: AclTags | None) -> list[Document]:
        docs: list[Document] = []
        with span("ingest", source=str(connector.root)):
            for uri in connector.list_uris():
                docs.append(connector.fetch(uri, acl=acl))
        return docs

    def _parse(self, docs: list[Document]) -> list[Document]:
        parsed: list[Document] = []
        with span("parse"):
            for doc in docs:
                try:
                    parsed_doc = parse_document(doc)
                except (KeyError, ValueError) as exc:
                    # One malformed document must not abort a whole reindex.
                    METRICS.incr("parse_failures")
                    log.warning("skipping %s: %s", doc.source_uri, exc)
                    continue
                if parsed_doc.text and self.settings.pii_redaction_enabled:
                    # Redact before anything is embedded or stored.
                    parsed_doc = parsed_doc.model_copy(update={"text": redact_pii(parsed_doc.text)})
                parsed.append(parsed_doc)
        return parsed

    def _embed_and_index(self, chunks: list[Chunk], version: str) -> list[Chunk]:
        with span("embed_index", embedding_model=self.embedder.model_name, chunks=len(chunks)):
            # Stamp the model so swapping embedders invalidates the index.
            stamped = [
                c.model_copy(update={"embedding_model": self.embedder.model_name}) for c in chunks
            ]
            vectors = self.embedder.embed([c.text for c in stamped])
            self.vector_store.upsert(stamped, vectors)
            self.lexical_index.upsert(stamped)
        _ = version
        return stamped


def default_offline_pipeline(settings: Settings | None = None) -> OfflinePipeline:
    settings = settings or get_settings()
    return from_context(BackendContext.from_settings(settings))


def from_context(context: BackendContext) -> OfflinePipeline:
    return OfflinePipeline(
        vector_store=context.vector_store,
        lexical_index=context.lexical_index,
        embedder=context.embedder,
        settings=context.settings,
    )
