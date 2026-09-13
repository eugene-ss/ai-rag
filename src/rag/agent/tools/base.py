"""Tool protocol for the agent runtime.

Three non-negotiable properties:
- `principal` is injected by the runtime, never chosen by the model.
- Arguments are validated against the tool's JSON schema before execution.
- A tool that cannot run is never advertised. See `available`.
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

    @property
    def available(self) -> bool:
        """Whether this tool can actually execute right now.

        The registry refuses to advertise a tool that answers False, because a
        planner can only avoid a tool it was never offered. Relying on prose in
        `description` to warn the model off does not work: a call still costs a
        step and a tool call before it fails, so an unimplemented tool silently
        eats the turn's budget.

        Two distinct uses: a declared-but-unimplemented tool answers False
        permanently, and a tool whose backend is down can answer False for as
        long as that lasts — a runtime state no description can express.
        """
        ...

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
