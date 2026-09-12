from __future__ import annotations

from rag.exceptions import MissingBackendError
from rag.schemas import Usage


class OpenAIChatClient:
    """Lazy OpenAI chat adapter. Requires the 'openai' extra.

    Wrap with ResilientLLM for retries, timeouts, and model fallback.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise MissingBackendError("OpenAIChatClient", "openai") from exc
        self.model_name = model

    def complete(self, prompt: str) -> tuple[str, Usage]:
        raise MissingBackendError("OpenAIChatClient", "openai")
