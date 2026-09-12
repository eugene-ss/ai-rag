from __future__ import annotations

from rag.exceptions import MissingBackendError
from rag.observability.cost import estimate_cost_usd
from rag.schemas import Usage


class OpenAIChatClient:
    """OpenAI chat-completions adapter.

    Requires the 'openai' extra. Deliberately has no retry logic of its own:
    wrap it in `ResilientLLM` so retries, timeouts, and fallback are handled
    uniformly across providers.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout: float = 30.0,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise MissingBackendError("OpenAIChatClient", "openai") from exc
        if not api_key:
            msg = "OPENAI_API_KEY is required for OpenAIChatClient"
            raise ValueError(msg)

        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        # max_retries=0: ResilientLLM owns the retry policy.
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    def complete(self, prompt: str) -> tuple[str, Usage]:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        raw_usage = response.usage
        prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(raw_usage, "completion_tokens", 0) or 0
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=getattr(raw_usage, "total_tokens", prompt_tokens + completion_tokens) or 0,
            cost_usd=estimate_cost_usd(self.model_name, prompt_tokens, completion_tokens),
            model=self.model_name,
        )
        return text, usage
