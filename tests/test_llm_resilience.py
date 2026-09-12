from __future__ import annotations

import pytest

from rag.llm.echo import EchoLLM, FlakyLLM
from rag.llm.errors import LLMTimeoutError, LLMUnavailableError
from rag.llm.policy import RetryPolicy
from rag.llm.resilient import ResilientLLM
from rag.schemas import Usage

FAST = RetryPolicy(max_attempts=3, base_delay_seconds=0.0, jitter=False, timeout_seconds=5.0)


def test_retries_until_success() -> None:
    flaky = FlakyLLM(fail_times=2)
    llm = ResilientLLM(flaky, policy=FAST)
    text, usage = llm.complete("prompt")
    assert text == "ok from flaky"
    assert usage.model == "flaky"
    assert flaky.calls == 3


def test_falls_back_to_next_model() -> None:
    primary = FlakyLLM(fail_times=99, model_name="primary")
    backup = EchoLLM()
    llm = ResilientLLM(primary, fallbacks=[backup], policy=FAST)
    text, usage = llm.complete("## Context\n[chunk_id=abc123]\nhello world\n")
    assert usage.model == "echo"
    assert text
    assert primary.calls == FAST.max_attempts


def test_raises_when_all_models_exhausted() -> None:
    a = FlakyLLM(fail_times=99, model_name="a")
    b = FlakyLLM(fail_times=99, model_name="b")
    llm = ResilientLLM(a, fallbacks=[b], policy=FAST)
    with pytest.raises(LLMUnavailableError) as exc:
        llm.complete("prompt")
    assert exc.value.models == ["a", "b"]


def test_timeout_is_enforced_per_attempt() -> None:
    class SlowLLM:
        model_name = "slow"

        def complete(self, prompt: str) -> tuple[str, Usage]:
            import time

            time.sleep(2.0)
            return "too late", Usage(model=self.model_name)

    policy = RetryPolicy(max_attempts=1, base_delay_seconds=0.0, jitter=False, timeout_seconds=0.05)
    llm = ResilientLLM(SlowLLM(), policy=policy)
    with pytest.raises(LLMUnavailableError) as exc:
        llm.complete("prompt")
    assert "LLMTimeoutError" in exc.value.last_error


def test_timeout_error_message() -> None:
    err = LLMTimeoutError("gpt-4o-mini", 30.0)
    assert "gpt-4o-mini" in str(err)
    assert err.timeout_seconds == 30.0


def test_backoff_is_exponential_and_capped() -> None:
    policy = RetryPolicy(
        base_delay_seconds=1.0, max_delay_seconds=4.0, jitter=False, max_attempts=5
    )
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0
    assert policy.delay_for(3) == 4.0
    assert policy.delay_for(4) == 4.0
