from __future__ import annotations


class LLMError(RuntimeError):
    """Base class for LLM transport failures."""


class LLMTimeoutError(LLMError):
    """A single attempt exceeded its timeout budget."""

    def __init__(self, model: str, timeout_seconds: float) -> None:
        super().__init__(f"{model} timed out after {timeout_seconds}s")
        self.model = model
        self.timeout_seconds = timeout_seconds


class LLMUnavailableError(LLMError):
    """All models (primary and fallbacks) failed."""

    def __init__(self, models: list[str], last_error: str) -> None:
        super().__init__(
            f"all LLM models exhausted ({', '.join(models)}); last error: {last_error}"
        )
        self.models = models
        self.last_error = last_error
