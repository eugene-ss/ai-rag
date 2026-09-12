from __future__ import annotations

from rag.exceptions import MissingBackendError
from rag.schemas import ScoredChunk


class CrossEncoderReranker:
    """Lazy cross-encoder reranker. Requires the 'rerank' extra."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise MissingBackendError("CrossEncoderReranker", "rerank") from exc
        self.model_name = model_name

    def rerank(self, query: str, candidates: list[ScoredChunk], *, top_k: int) -> list[ScoredChunk]:
        raise MissingBackendError("CrossEncoderReranker", "rerank")
