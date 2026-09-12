"""Tool-result cache: keyed by (tool, canonical args, ACL scope).

Agents re-retrieve similar things within and across turns; caching tool
outputs is where the cost savings live, not answer caching alone.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rag.cache.base import Cache
from rag.schemas import Principal
from rag.schemas.agent import ToolResult
from rag.security.acl import acl_fingerprint


def tool_result_cache_key(
    *,
    tool: str,
    arguments: dict[str, Any],
    principal: Principal,
) -> str:
    payload = {
        "tool": tool,
        "args": arguments,
        "acl": acl_fingerprint(principal),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "tool:" + hashlib.sha256(raw.encode()).hexdigest()


class ToolResultCache:
    """Thin wrapper around the shared Cache protocol for tool results."""

    def __init__(self, cache: Cache, *, ttl_seconds: int = 600, enabled: bool = True) -> None:
        self._cache = cache
        self._ttl = ttl_seconds
        self._enabled = enabled

    def get(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        principal: Principal,
    ) -> ToolResult | None:
        if not self._enabled:
            return None
        key = tool_result_cache_key(tool=tool, arguments=arguments, principal=principal)
        raw = self._cache.get(key)
        if raw is None:
            return None
        try:
            return ToolResult.model_validate_json(raw)
        except Exception:
            return None

    def set(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        principal: Principal,
        result: ToolResult,
    ) -> None:
        if not self._enabled or not result.ok:
            return
        key = tool_result_cache_key(tool=tool, arguments=arguments, principal=principal)
        # Drop call_id from the cached payload; the registry rebinds it per call.
        cached = result.model_copy(update={"call_id": ""})
        self._cache.set(key, cached.model_dump_json(), ttl_seconds=self._ttl)
