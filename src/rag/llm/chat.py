"""Multi-turn chat protocol with structured tool calls.

Sibling to `LLMClient` (prompt → text). An agent needs messages and tool calls;
forcing that through `complete()` would bury the structure in string parsing.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from rag.schemas import Usage
from rag.schemas.agent import ToolCall


class Message(BaseModel):
    """One turn in a chat conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolSpec(BaseModel):
    """JSON-schema description of a tool the model may call."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ChatCompletion(BaseModel):
    """Result of one chat turn: text and/or tool calls, plus usage."""

    message: Message
    usage: Usage = Field(default_factory=Usage)
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"] = "stop"

    @property
    def text(self) -> str:
        return self.message.content

    @property
    def tool_calls(self) -> list[ToolCall]:
        return list(self.message.tool_calls)


@runtime_checkable
class ChatLLM(Protocol):
    """Async multi-turn chat backend with optional tool calling."""

    model_name: str

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ChatCompletion: ...
