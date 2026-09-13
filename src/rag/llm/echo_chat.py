"""Scripted ChatLLM for hermetic agent tests.

Same role HashEmbedder plays for embeddings: deterministic, no network, and
capable of emitting pre-planned tool-call sequences so the agent loop can be
exercised without a real model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from rag.llm.chat import ChatCompletion, ChatLLM, Message, ToolSpec
from rag.observability.cost import estimate_cost_usd
from rag.schemas import Usage
from rag.schemas.agent import ToolCall


@dataclass
class ScriptedTurn:
    """One canned response the EchoChatLLM will emit in order."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class EchoChatLLM:
    """Deterministic chat stub that walks a script of planned turns.

    When the script is exhausted it returns a stop turn with empty content so
    the agent loop can terminate cleanly rather than hang.
    """

    def __init__(
        self,
        script: list[ScriptedTurn] | None = None,
        *,
        model_name: str = "echo-chat",
    ) -> None:
        self.model_name = model_name
        self._script = list(script or [])
        self._cursor = 0
        self.calls: list[list[Message]] = []

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ChatCompletion:
        self.calls.append(copy.deepcopy(messages))
        _ = tools
        if self._cursor < len(self._script):
            turn = self._script[self._cursor]
            self._cursor += 1
        else:
            turn = ScriptedTurn(content="", finish_reason="stop")

        finish: str = "tool_calls" if turn.tool_calls else turn.finish_reason
        if finish not in {"stop", "tool_calls", "length", "content_filter"}:
            finish = "stop"

        content = turn.content
        prompt_tokens = max(sum(len(m.content.split()) for m in messages), 1)
        completion_tokens = max(len(content.split()) + 2 * len(turn.tool_calls), 1)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=estimate_cost_usd(self.model_name, prompt_tokens, completion_tokens),
            model=self.model_name,
        )
        message = Message(
            role="assistant",
            content=content,
            tool_calls=list(turn.tool_calls),
        )
        return ChatCompletion(
            message=message,
            usage=usage,
            finish_reason=finish,  # type: ignore[arg-type]
        )

    def reset(self) -> None:
        self._cursor = 0
        self.calls.clear()


# Protocol structural check — EchoChatLLM must satisfy ChatLLM.
_: type[ChatLLM] = EchoChatLLM
