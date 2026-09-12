from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class MetricsRegistry:
    """In-process counters and histograms for latency / quality / cost."""

    counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    histograms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: Lock = field(default_factory=Lock)

    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = _key(name, labels)
        with self._lock:
            self.counters[key] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = _key(name, labels)
        with self._lock:
            self.histograms[key].append(value)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "histograms": {
                    k: {"count": len(v), "sum": sum(v), "avg": (sum(v) / len(v) if v else 0.0)}
                    for k, v in self.histograms.items()
                },
            }


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    parts = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{parts}}}"


METRICS = MetricsRegistry()
