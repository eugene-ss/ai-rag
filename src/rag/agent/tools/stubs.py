"""Declared but unimplemented egress tools.

They exist so the registry, egress gate, and planner prompt surface are real
before the backends are. Calling them raises MissingBackendError.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rag.agent.budget import BudgetTracker
from rag.agent.tools.mixin import ToolMixin
from rag.exceptions import MissingBackendError
from rag.schemas import Principal
from rag.schemas.agent import ToolResult


class WebSearchTool(ToolMixin):
    """External web search. Requires egress grant; not implemented in this phase."""

    name = "web_search"
    description = (
        "Search the public web for up-to-date information not in the corpus. "
        "Requires egress permission. Currently unavailable."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Web search query."},
            "max_results": {"type": "integer", "description": "Max results (1-10)."},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    requires_egress = True

    async def run(
        self,
        arguments: dict[str, Any],
        *,
        principal: Principal,
        budget: BudgetTracker,
    ) -> ToolResult:
        _ = arguments, principal, budget
        raise MissingBackendError("WebSearchTool", "web")


class GraphQueryTool(ToolMixin):
    """Knowledge-graph query. Declared only; needs offline extraction + store."""

    name = "graph_query"
    description = "Query the knowledge graph for entities and relations. Currently unavailable."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language or structured graph query.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    requires_egress = False

    async def run(
        self,
        arguments: dict[str, Any],
        *,
        principal: Principal,
        budget: BudgetTracker,
    ) -> ToolResult:
        _ = arguments, principal, budget
        raise MissingBackendError("GraphQueryTool", "graph")
