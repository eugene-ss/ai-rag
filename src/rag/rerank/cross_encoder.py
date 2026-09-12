from __future__ import annotations

from rag.exceptions import MissingBackendError
from rag.observability.tracing import span
from rag.schemas import ScoredChunk


class CrossEncoderReranker:
    """Cross-encoder reranker over the retrieved candidate set.

    Requires the 'rerank' extra. Scores every (query, chunk) pair jointly, which
    is far more accurate than the fused retrieval score but too slow to run over
    a whole corpus — hence rerank-after-retrieve.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise MissingBackendError("CrossEncoderReranker", "rerank") from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, candidates: list[ScoredChunk], *, top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        with span("rerank_cross_encoder", candidates=len(candidates)) as attrs:
            pairs = [(query, c.chunk.text) for c in candidates]
            scores = self._model.predict(pairs, batch_size=self.batch_size)
            ordered = sorted(
                zip(candidates, scores, strict=True),
                key=lambda pair: float(pair[1]),
                reverse=True,
            )
            results = [
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=float(score),
                    rank=rank,
                    retriever="rerank",
                )
                for rank, (candidate, score) in enumerate(ordered[:top_k], start=1)
            ]
            attrs["returned"] = len(results)
            return results
