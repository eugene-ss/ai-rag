from __future__ import annotations

from rag.exceptions import MissingBackendError
from rag.observability.logging import get_logger

log = get_logger("embedding.openai")

_MAX_BATCH = 128


class OpenAIEmbedder:
    """OpenAI embeddings adapter.

    Requires the 'openai' extra. Batches inputs and preserves ordering so the
    returned vectors line up with the chunks passed in.
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise MissingBackendError("OpenAIEmbedder", "openai") from exc
        if not api_key:
            msg = "OPENAI_API_KEY is required for OpenAIEmbedder"
            raise ValueError(msg)

        self.model_name = model
        self.dimensions = dimensions
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            batch = texts[start : start + _MAX_BATCH]
            response = self._client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=self.dimensions,
            )
            # Sort by index: the API does not guarantee response ordering.
            for item in sorted(response.data, key=lambda d: d.index):
                vectors.append(list(item.embedding))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed([text])
        if not vectors:
            msg = "embedding request returned no vectors"
            raise RuntimeError(msg)
        return vectors[0]
