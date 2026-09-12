from __future__ import annotations

from rag.exceptions import MissingBackendError
from rag.observability.logging import get_logger

log = get_logger("cache.redis")


class RedisCache:
    """Shared answer cache backed by Redis.

    Requires the 'redis' extra. Necessary once more than one API replica is
    running, since MemoryCache is per-process and would give inconsistent hits.
    """

    def __init__(
        self,
        *,
        url: str = "redis://localhost:6379/0",
        namespace: str = "rag:answer",
        socket_timeout: float = 2.0,
    ) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise MissingBackendError("RedisCache", "redis") from exc

        self.namespace = namespace
        self._client = redis.Redis.from_url(
            url,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            decode_responses=True,
        )

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> str | None:
        # A cache outage must degrade to a miss, never fail the request.
        try:
            value = self._client.get(self._key(key))
        except Exception as exc:  # pragma: no cover - transport dependent
            log.warning("redis get failed, treating as miss: %s", exc)
            return None
        return str(value) if value is not None else None

    def set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        try:
            if ttl_seconds:
                self._client.setex(self._key(key), ttl_seconds, value)
            else:
                self._client.set(self._key(key), value)
        except Exception as exc:  # pragma: no cover - transport dependent
            log.warning("redis set failed, continuing without caching: %s", exc)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(self._key(key))
        except Exception as exc:  # pragma: no cover - transport dependent
            log.warning("redis delete failed: %s", exc)

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:  # pragma: no cover - transport dependent
            return False
