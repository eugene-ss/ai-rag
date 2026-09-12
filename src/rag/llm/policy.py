from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with optional jitter, plus a per-attempt timeout."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 5.0
    timeout_seconds: float = 30.0
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        """Delay before retrying; attempt is 1-based and counts the failed try."""
        if attempt < 1:
            return 0.0
        raw = self.base_delay_seconds * float(2 ** (attempt - 1))
        capped = float(min(raw, self.max_delay_seconds))
        if not self.jitter:
            return capped
        return capped * (0.5 + random.random() / 2)
