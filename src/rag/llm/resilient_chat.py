"""Async resilience wrapper for ChatLLM: retries, timeouts, fallbacks."""

from __future__ import annotations

import asyncio

from rag.llm.chat import ChatCompletion, ChatLLM, Message, ToolSpec
from rag.llm.errors import LLMTimeoutError, LLMUnavailableError
from rag.llm.policy import RetryPolicy
from rag.observability.logging import get_logger
from rag.observability.metrics import METRICS
from rag.observability.tracing import span

log = get_logger("llm.resilient_chat")


class ResilientChatLLM:
    """Retries each model with asyncio.wait_for, then falls through the chain."""

    def __init__(
        self,
        primary: ChatLLM,
        *,
        fallbacks: list[ChatLLM] | None = None,
        policy: RetryPolicy | None = None,
    ) -> None:
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.policy = policy or RetryPolicy()
        self.model_name = primary.model_name

    @property
    def chain(self) -> list[ChatLLM]:
        return [self.primary, *self.fallbacks]

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ChatCompletion:
        last_error = "none"
        with span("chat_llm", policy_attempts=self.policy.max_attempts) as attrs:
            for client in self.chain:
                for attempt in range(1, self.policy.max_attempts + 1):
                    try:
                        result = await self._attempt(client, messages, tools=tools)
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        METRICS.incr("llm_attempt_failed", model=client.model_name)
                        log.warning(
                            "chat llm attempt failed model=%s attempt=%s/%s error=%s",
                            client.model_name,
                            attempt,
                            self.policy.max_attempts,
                            last_error,
                        )
                        if attempt < self.policy.max_attempts:
                            await asyncio.sleep(self.policy.delay_for(attempt))
                        continue
                    METRICS.incr("llm_attempt_ok", model=client.model_name)
                    attrs["model"] = client.model_name
                    attrs["attempts"] = attempt
                    attrs["fell_back"] = client is not self.primary
                    return result
            attrs["exhausted"] = True
            raise LLMUnavailableError([c.model_name for c in self.chain], last_error)

    async def _attempt(
        self,
        client: ChatLLM,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None,
    ) -> ChatCompletion:
        timeout = self.policy.timeout_seconds
        if timeout <= 0:
            return await client.chat(messages, tools=tools)
        try:
            return await asyncio.wait_for(
                client.chat(messages, tools=tools),
                timeout=timeout,
            )
        except TimeoutError as exc:
            raise LLMTimeoutError(client.model_name, timeout) from exc
