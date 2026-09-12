"""Shared helpers for Tool implementations."""

from __future__ import annotations

from typing import Any, ClassVar

from rag.llm.chat import ToolSpec


class ToolMixin:
    """Concrete defaults for name/description/parameters/spec()."""

    name: str
    description: str
    parameters: ClassVar[dict[str, Any]]
    requires_egress: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
