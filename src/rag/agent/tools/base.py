"""Tool protocol for the agent runtime.

Two non-negotiable properties:
- `principal` is injected by the runtime, never chosen by the model.
- Arguments are validated against the tool's JSON schema before execution.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from rag.agent.budget import BudgetTracker
from rag.llm.chat import ToolSpec
from rag.schemas import Principal
from rag.schemas.agent import ToolResult


@runtime_checkable
class Tool(Protocol):
    """One capability the agent may invoke."""

    name: str
    description: str
    parameters: ClassVar[dict[str, Any]]
    requires_egress: bool

    async def run(
        self,
        arguments: dict[str, Any],
        *,
        principal: Principal,
        budget: BudgetTracker,
    ) -> ToolResult: ...

    def spec(self) -> ToolSpec:
        """JSON-schema description exposed to the planning model."""
        ...
