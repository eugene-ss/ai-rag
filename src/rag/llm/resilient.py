from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from rag.llm.base import LLMClient
from rag.llm.errors import LLMTimeoutError, LLMUnavailableError
from rag.llm.policy import RetryPolicy
from rag.observability.logging import get_logger
from rag.observability.metrics import METRICS
from rag.observability.tracing import span
from rag.schemas import Usage

log = get_logger("llm.resilient")


class ResilientLLM:
    """Wraps a primary client with retries, per-attempt timeout, and fallbacks.

    Retries are attempted against each model in turn; only when a model
    exhausts its attempts do we move to the next fallback.
    """

    def __init__(
        self,
        primary: LLMClient,
        *,
        fallbacks: list[LLMClient] | None = None,
        policy: RetryPolicy | None = None,
    ) -> None:
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.policy = policy or RetryPolicy()
        self.model_name = primary.model_name

    @property
    def chain(self) -> list[LLMClient]:
        return [self.primary, *self.fallbacks]

    def complete(self, prompt: str) -> tuple[str, Usage]:
        last_error = "none"
        with span("llm_complete", policy_attempts=self.policy.max_attempts) as attrs:
            for client in self.chain:
                for attempt in range(1, self.policy.max_attempts + 1):
                    try:
                        text, usage = self._attempt(client, prompt)
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        METRICS.incr("llm_attempt_failed", model=client.model_name)
                        log.warning(
                            "llm attempt failed model=%s attempt=%s/%s error=%s",
                            client.model_name,
                            attempt,
                            self.policy.max_attempts,
                            last_error,
                        )
                        if attempt < self.policy.max_attempts:
                            time.sleep(self.policy.delay_for(attempt))
                        continue
                    METRICS.incr("llm_attempt_ok", model=client.model_name)
                    attrs["model"] = client.model_name
                    attrs["attempts"] = attempt
                    attrs["fell_back"] = client is not self.primary
                    return text, usage
            attrs["exhausted"] = True
            raise LLMUnavailableError([c.model_name for c in self.chain], last_error)

    def _attempt(self, client: LLMClient, prompt: str) -> tuple[str, Usage]:
        timeout = self.policy.timeout_seconds
        if timeout <= 0:
            return client.complete(prompt)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(client.complete, prompt)
            try:
                return future.result(timeout=timeout)
            except FutureTimeout as exc:
                future.cancel()
                raise LLMTimeoutError(client.model_name, timeout) from exc
