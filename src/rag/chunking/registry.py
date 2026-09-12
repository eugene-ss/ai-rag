from __future__ import annotations

from typing import Any

from rag.chunking.base import Chunker
from rag.chunking.fixed import FixedChunker
from rag.chunking.markdown_aware import MarkdownAwareChunker
from rag.chunking.recursive import RecursiveChunker
from rag.chunking.semantic import SemanticChunker
from rag.schemas import Chunk, Document

REGISTRY: dict[str, Chunker] = {
    "fixed": FixedChunker(),
    "recursive": RecursiveChunker(),
    "markdown_aware": MarkdownAwareChunker(),
    "semantic": SemanticChunker(),
}

MIME_DEFAULTS: dict[str, str] = {
    "text/plain": "fixed",
    "text/markdown": "markdown_aware",
    "application/pdf": "recursive",
    "text/html": "recursive",
}


def get_chunker(name: str) -> Chunker:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        msg = f"Unknown chunker: {name}. Available: {sorted(REGISTRY)}"
        raise KeyError(msg) from exc


def register(name: str, chunker: Chunker) -> None:
    REGISTRY[name] = chunker


def chunk_document(
    document: Document,
    *,
    index_version: str,
    config: dict[str, Any] | None = None,
) -> list[Chunk]:
    cfg = config or {}
    strategies = cfg.get("strategies", {})
    mime_cfg = strategies.get(document.mime_type, {})
    strategy = mime_cfg.get("strategy") or MIME_DEFAULTS.get(
        document.mime_type, cfg.get("default_strategy", "recursive")
    )
    size = int(mime_cfg.get("size", 512))
    overlap = int(mime_cfg.get("overlap", 64))
    chunker = get_chunker(strategy)
    return chunker.chunk(document, index_version=index_version, size=size, overlap=overlap)
