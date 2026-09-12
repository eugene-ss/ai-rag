from __future__ import annotations

from rag.exceptions import MissingBackendError


class OpenAIEmbedder:
    """Lazy OpenAI embeddings adapter."""

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise MissingBackendError("OpenAIEmbedder", "openai") from exc
        self.model_name = model
        self.dimensions = dimensions
        self._client = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise MissingBackendError("OpenAIEmbedder", "openai")

    def embed_query(self, text: str) -> list[float]:
        raise MissingBackendError("OpenAIEmbedder", "openai")
