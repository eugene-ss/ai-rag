"""Shared helpers for Tool implementations."""

from __future__ import annotations

from typing import Any, ClassVar

from rag.llm.chat import ToolSpec


class ToolMixin:
    """Concrete defaults for name/description/parameters/available/spec()."""

    name: str
    description: str
    parameters: ClassVar[dict[str, Any]]
    requires_egress: bool = False

    @property
    def available(self) -> bool:
        """Available by default; override to withhold the tool from the planner."""
        return True

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
