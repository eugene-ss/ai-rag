from __future__ import annotations

# Rough USD per 1M tokens — illustrative defaults for cost accounting.
_DEFAULT_RATES: dict[str, tuple[float, float]] = {
    "echo": (0.0, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int = 0,
) -> float:
    prompt_rate, completion_rate = _DEFAULT_RATES.get(model, (1.0, 3.0))
    return (prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000
