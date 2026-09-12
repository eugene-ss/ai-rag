from __future__ import annotations

import math
from dataclasses import dataclass, field

from rag.cache.keys import normalize_query
from rag.embedding.base import Embedder
from rag.observability.metrics import METRICS


@dataclass
class _Entry:
    query: str
    vector: list[float]
    value: str


@dataclass
class SemanticCache:
    """Nearest-neighbour cache for paraphrased queries.

    Entries are bucketed by an opaque scope string (see `cache.keys.cache_scope`),
    so a match can only ever be found within the same ACL, index version, and
    prompt version. Cross-scope hits are structurally impossible rather than
    merely unlikely.
    """

    embedder: Embedder
    threshold: float = 0.95
    max_entries_per_scope: int = 256
    _buckets: dict[str, list[_Entry]] = field(default_factory=dict)

    def get(self, query: str, *, scope: str) -> str | None:
        entries = self._buckets.get(scope)
        if not entries:
            return None
        normalized = normalize_query(query)
        vector = self.embedder.embed_query(normalized)
        best: tuple[float, _Entry] | None = None
        for entry in entries:
            score = _cosine(vector, entry.vector)
            if best is None or score > best[0]:
                best = (score, entry)
        if best is None or best[0] < self.threshold:
            METRICS.incr("semantic_cache_miss")
            return None
        METRICS.incr("semantic_cache_hit")
        return best[1].value

    def set(self, query: str, value: str, *, scope: str) -> None:
        normalized = normalize_query(query)
        vector = self.embedder.embed_query(normalized)
        bucket = self._buckets.setdefault(scope, [])
        for entry in bucket:
            if entry.query == normalized:
                entry.value = value
                entry.vector = vector
                return
        bucket.append(_Entry(query=normalized, vector=vector, value=value))
        if len(bucket) > self.max_entries_per_scope:
            del bucket[0]

    def clear(self) -> None:
        self._buckets.clear()

    def size(self, scope: str | None = None) -> int:
        if scope is None:
            return sum(len(b) for b in self._buckets.values())
        return len(self._buckets.get(scope, []))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
