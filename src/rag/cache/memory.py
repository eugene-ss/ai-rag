from __future__ import annotations

import time


class MemoryCache:
    """In-process TTL cache."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float | None]] = {}

    def get(self, key: str) -> str | None:
        item = self._store.get(key)
        if item is None:
            return None
        value, expires = item
        if expires is not None and time.time() > expires:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        expires = time.time() + ttl_seconds if ttl_seconds else None
        self._store[key] = (value, expires)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
