"""LLM clients with retries, timeouts, and model fallback."""

from rag.llm.base import LLMClient
from rag.llm.chat import ChatCompletion, ChatLLM, Message, ToolSpec
from rag.llm.echo import EchoLLM, FlakyLLM
from rag.llm.echo_chat import EchoChatLLM, ScriptedTurn
from rag.llm.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from rag.llm.openai import OpenAIChatClient
from rag.llm.openai_chat import OpenAIChatLLM
from rag.llm.policy import RetryPolicy
from rag.llm.resilient import ResilientLLM
from rag.llm.resilient_chat import ResilientChatLLM

__all__ = [
    "ChatCompletion",
    "ChatLLM",
    "EchoChatLLM",
    "EchoLLM",
    "FlakyLLM",
    "LLMClient",
    "LLMError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "Message",
    "OpenAIChatClient",
    "OpenAIChatLLM",
    "ResilientChatLLM",
    "ResilientLLM",
    "RetryPolicy",
    "ScriptedTurn",
    "ToolSpec",
]
