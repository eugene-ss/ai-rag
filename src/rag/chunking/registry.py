from __future__ import annotations

from typing import Any

from rag.chunking.base import Chunker
from rag.chunking.fixed import FixedChunker
from rag.chunking.markdown_aware import MarkdownAwareChunker
from rag.chunking.recursive import RecursiveChunker
from rag.chunking.semantic import SemanticChunker
from rag.embedding.base import Embedder
from rag.schemas import Chunk, Document

REGISTRY: dict[str, Chunker] = {
    "fixed": FixedChunker(),
    "recursive": RecursiveChunker(),
    "markdown_aware": MarkdownAwareChunker(),
    "semantic": SemanticChunker(),
}

# Strategy per document type, not one splitter for everything.
MIME_DEFAULTS: dict[str, str] = {
    "text/plain": "fixed",
    "text/markdown": "markdown_aware",
    "application/pdf": "recursive",
    "text/html": "recursive",
    "application/json": "recursive",
}


def get_chunker(name: str) -> Chunker:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        msg = f"Unknown chunker: {name}. Available: {sorted(REGISTRY)}"
        raise KeyError(msg) from exc


def register(name: str, chunker: Chunker) -> None:
    REGISTRY[name] = chunker


def enable_semantic_chunking(embedder: Embedder) -> None:
    """Upgrade the semantic strategy from fallback to real embedding-based splits."""
    REGISTRY["semantic"] = SemanticChunker(embedder)


def chunk_document(
    document: Document,
    *,
    index_version: str,
    config: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Chunk a document using the strategy configured for its mime type."""
    cfg = config or {}
    strategies = cfg.get("strategies", {})
    mime_cfg = strategies.get(document.mime_type, {})
    strategy = mime_cfg.get("strategy") or MIME_DEFAULTS.get(
        document.mime_type, cfg.get("default_strategy", "recursive")
    )
    size = int(mime_cfg.get("size", 512))
    overlap = int(mime_cfg.get("overlap", 64))
    if overlap >= size:
        msg = f"overlap ({overlap}) must be smaller than size ({size}) for {document.mime_type}"
        raise ValueError(msg)
    chunker = get_chunker(strategy)
    return chunker.chunk(document, index_version=index_version, size=size, overlap=overlap)
