from __future__ import annotations

from rag.observability.cost import estimate_cost_usd
from rag.schemas import Usage


class EchoLLM:
    """Deterministic stub LLM for offline tests — echoes grounded context."""

    model_name = "echo"

    def complete(self, prompt: str) -> tuple[str, Usage]:
        text = prompt
        if "## Context" in prompt:
            after = prompt.split("## Context", 1)[1]
            if "## Instructions" in after:
                after = after.split("## Instructions", 1)[0]
            lines = [ln.strip() for ln in after.strip().splitlines() if ln.strip()]
            body_lines = [ln for ln in lines if not ln.startswith("[") and not ln.startswith("#")]
            if body_lines:
                text = " ".join(body_lines[:4])
                markers = [ln for ln in lines if ln.startswith("[chunk_id=")]
                if markers:
                    cites = " ".join(f"[{m.split('=', 1)[1].rstrip(']')}]" for m in markers[:3])
                    text = f"{text} {cites}".strip()
        prompt_tokens = max(len(prompt.split()), 1)
        completion_tokens = max(len(text.split()), 1)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=estimate_cost_usd(self.model_name, prompt_tokens, completion_tokens),
            model=self.model_name,
        )
        return text, usage


class FlakyLLM:
    """Test double that fails a fixed number of times before succeeding."""

    def __init__(self, *, fail_times: int, model_name: str = "flaky") -> None:
        self.model_name = model_name
        self.fail_times = fail_times
        self.calls = 0

    def complete(self, prompt: str) -> tuple[str, Usage]:
        self.calls += 1
        if self.calls <= self.fail_times:
            msg = f"{self.model_name} transient failure {self.calls}"
            raise RuntimeError(msg)
        return f"ok from {self.model_name}", Usage(model=self.model_name)
