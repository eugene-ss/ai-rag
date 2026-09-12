from __future__ import annotations

from pathlib import Path

from rag.embedding.hash_embedder import HashEmbedder
from rag.lexical.bm25_memory import BM25MemoryIndex
from rag.observability.logging import get_logger
from rag.pipelines.offline import OfflinePipeline
from rag.settings import get_settings
from rag.vectordb.memory import MemoryVectorStore

log = get_logger("jobs.reindex")

# Process-local stores shared with CLI eval when using memory backends.
VECTOR_STORE = MemoryVectorStore()
LEXICAL_INDEX = BM25MemoryIndex()
EMBEDDER = HashEmbedder()


def reindex(
    source: Path | str,
    *,
    index_version: str,
    activate: bool = True,
) -> int:
    """Build into {collection}__{index_version}, then flip alias if activate."""
    settings = get_settings()
    pipeline = OfflinePipeline(
        vector_store=VECTOR_STORE,
        lexical_index=LEXICAL_INDEX,
        embedder=EMBEDDER,
        settings=settings,
    )
    chunks = pipeline.run(source, index_version=index_version)
    if activate:
        VECTOR_STORE.set_alias(settings.collection_alias, index_version)
        log.info("alias %s -> %s", settings.collection_alias, index_version)
    return len(chunks)
