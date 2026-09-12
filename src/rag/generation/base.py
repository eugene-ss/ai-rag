from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.schemas import Usage


@runtime_checkable
class LLMClient(Protocol):
    model_name: str

    def complete(self, prompt: str) -> tuple[str, Usage]: ...
