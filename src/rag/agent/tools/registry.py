"""Tool registry with argument validation and egress gating."""

from __future__ import annotations

import json
from typing import Any

from rag.agent.budget import BudgetTracker
from rag.agent.tools.base import Tool
from rag.agent.tools.cache import ToolResultCache
from rag.llm.chat import ToolSpec
from rag.observability.logging import get_logger
from rag.observability.metrics import METRICS
from rag.schemas import Principal
from rag.schemas.agent import ToolCall, ToolResult

log = get_logger("agent.tools")


class ToolRegistry:
    """Name → Tool map with allow-lists and pre-execution validation."""

    def __init__(
        self,
        tools: list[Tool] | None = None,
        *,
        result_cache: ToolResultCache | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._result_cache = result_cache
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            msg = f"duplicate tool registration: {tool.name}"
            raise ValueError(msg)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self, *, principal: Principal, allow_egress: bool = False) -> list[ToolSpec]:
        return [t.spec() for t in self.allowed_for(principal, allow_egress=allow_egress)]

    def allowed_for(self, principal: Principal, *, allow_egress: bool = False) -> list[Tool]:
        """Drop egress tools unless the tenant holds the grant.

        `principal` is accepted for future per-tenant allow-lists; today the
        gate is the boolean `allow_egress` flag from settings.
        """
        _ = principal
        return [tool for tool in self._tools.values() if allow_egress or not tool.requires_egress]

    async def execute(
        self,
        call: ToolCall,
        *,
        principal: Principal,
        budget: BudgetTracker,
        allow_egress: bool = False,
        index_version: str = "",
    ) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            METRICS.incr("agent_tool_calls", tool=call.name, ok="false")
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                error=f"unknown_tool:{call.name}",
            )
        if tool.requires_egress and not allow_egress:
            METRICS.incr("agent_tool_calls", tool=call.name, ok="false")
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                error="egress_denied",
            )
        try:
            arguments = validate_arguments(call.arguments, tool.parameters)
        except ValueError as exc:
            METRICS.incr("agent_tool_calls", tool=call.name, ok="false")
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                error=f"invalid_arguments:{exc}",
            )
        if self._result_cache is not None:
            cached = self._result_cache.get(
                tool=call.name,
                arguments=arguments,
                principal=principal,
                index_version=index_version,
            )
            if cached is not None:
                METRICS.incr("agent_tool_calls", tool=call.name, ok="true", cached="true")
                return cached.model_copy(update={"call_id": call.id, "name": call.name})
        try:
            result = await tool.run(arguments, principal=principal, budget=budget)
        except Exception as exc:
            log.warning("tool %s raised: %s", call.name, exc)
            METRICS.incr("agent_tool_calls", tool=call.name, ok="false")
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        # Force identity fields from the call, never from the tool.
        result = result.model_copy(update={"call_id": call.id, "name": call.name})
        if self._result_cache is not None and result.ok:
            self._result_cache.set(
                tool=call.name,
                arguments=arguments,
                principal=principal,
                result=result,
                index_version=index_version,
            )
        METRICS.incr("agent_tool_calls", tool=call.name, ok=str(result.ok).lower())
        return result


def validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Validate tool arguments against a JSON-schema-ish parameters object.

    Rejects unexpected keys so a poisoned document cannot smuggle fields the
    tool did not declare. Required fields must be present.
    """
    if not isinstance(arguments, dict):
        msg = "arguments must be an object"
        raise ValueError(msg)

    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        properties = {}
    additional = schema.get("additionalProperties", False)
    required = schema.get("required") or []

    if additional is False:
        unexpected = set(arguments) - set(properties)
        if unexpected:
            msg = f"unexpected fields: {sorted(unexpected)}"
            raise ValueError(msg)

    missing = [key for key in required if key not in arguments]
    if missing:
        msg = f"missing required fields: {missing}"
        raise ValueError(msg)

    # Shallow type checks for the common JSON-schema primitives.
    for key, value in arguments.items():
        prop = properties.get(key) or {}
        expected = prop.get("type")
        if expected and not _matches_type(value, expected):
            msg = f"field {key!r} expected {expected}, got {type(value).__name__}"
            raise ValueError(msg)

    # Round-trip through JSON to reject non-serialisable payloads early.
    try:
        json.dumps(arguments)
    except (TypeError, ValueError) as exc:
        msg = f"arguments not JSON-serialisable: {exc}"
        raise ValueError(msg) from exc

    return dict(arguments)


def _matches_type(value: Any, expected: str | list[str]) -> bool:
    kinds = {expected} if isinstance(expected, str) else set(expected)
    if "string" in kinds and isinstance(value, str):
        return True
    if "number" in kinds and isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if "integer" in kinds and isinstance(value, int) and not isinstance(value, bool):
        return True
    if "boolean" in kinds and isinstance(value, bool):
        return True
    if "array" in kinds and isinstance(value, list):
        return True
    if "object" in kinds and isinstance(value, dict):
        return True
    return bool("null" in kinds and value is None)
