"""Async OpenAI chat adapter with native function calling."""

from __future__ import annotations

import json
from typing import Any

from rag.exceptions import MissingBackendError
from rag.llm.chat import ChatCompletion, Message, ToolSpec
from rag.observability.cost import estimate_cost_usd
from rag.schemas import Usage
from rag.schemas.agent import ToolCall


class OpenAIChatLLM:
    """Async OpenAI chat-completions client with tool calling.

    Requires the 'openai' extra. No retries of its own — wrap in
    `ResilientChatLLM` so policy is uniform across providers.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout: float = 30.0,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise MissingBackendError("OpenAIChatLLM", "openai") from exc
        if not api_key:
            msg = "OPENAI_API_KEY is required for OpenAIChatLLM"
            raise ValueError(msg)

        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ChatCompletion:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [_to_openai_message(m) for m in messages],
            "max_completion_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        raw_msg = choice.message
        tool_calls = [_from_openai_tool_call(tc) for tc in (raw_msg.tool_calls or [])]
        content = raw_msg.content or ""
        finish = choice.finish_reason or ("tool_calls" if tool_calls else "stop")
        if finish == "tool_calls" or tool_calls:
            finish_reason = "tool_calls"
        elif finish in {"stop", "length", "content_filter"}:
            finish_reason = finish
        else:
            finish_reason = "stop"

        raw_usage = response.usage
        prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(raw_usage, "completion_tokens", 0) or 0
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=getattr(raw_usage, "total_tokens", prompt_tokens + completion_tokens)
            or 0,
            cost_usd=estimate_cost_usd(self.model_name, prompt_tokens, completion_tokens),
            model=self.model_name,
        )
        return ChatCompletion(
            message=Message(role="assistant", content=content, tool_calls=tool_calls),
            usage=usage,
            finish_reason=finish_reason,  # type: ignore[arg-type]
        )


def _to_openai_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content or ""}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in message.tool_calls
        ]
    return payload


def _to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters
            or {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


def _from_openai_tool_call(raw: Any) -> ToolCall:
    arguments: dict[str, Any]
    try:
        arguments = json.loads(raw.function.arguments or "{}")
    except json.JSONDecodeError:
        arguments = {"_raw": raw.function.arguments}
    if not isinstance(arguments, dict):
        arguments = {"_raw": arguments}
    return ToolCall(id=str(raw.id), name=str(raw.function.name), arguments=arguments)
