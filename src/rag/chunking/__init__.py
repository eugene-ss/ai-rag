"""Document-type-aware chunking strategies."""

from rag.chunking.base import Chunker
from rag.chunking.fixed import FixedChunker
from rag.chunking.markdown_aware import MarkdownAwareChunker
from rag.chunking.recursive import RecursiveChunker
from rag.chunking.registry import (
    chunk_document,
    enable_semantic_chunking,
    get_chunker,
    register,
)
from rag.chunking.semantic import SemanticChunker

__all__ = [
    "Chunker",
    "FixedChunker",
    "MarkdownAwareChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "chunk_document",
    "enable_semantic_chunking",
    "get_chunker",
    "register",
]
