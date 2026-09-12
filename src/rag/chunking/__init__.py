"""Document-type-aware chunking strategies."""

from rag.chunking.base import Chunker
from rag.chunking.registry import chunk_document, get_chunker, register

__all__ = ["Chunker", "chunk_document", "get_chunker", "register"]
