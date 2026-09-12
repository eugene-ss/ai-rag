"""Independent critic for agent drafts.

Sees only the draft and its sources — never the agent's chain of thought — so it
cannot rubber-stamp the planner's own reasoning. Cannot call tools.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rag.llm.chat import ChatLLM, Message
from rag.observability.logging import get_logger
from rag.prompts import get as get_prompt
from rag.schemas.agent import Verdict
from rag.schemas.answer import Usage
from rag.schemas.source import Source

log = get_logger("agent.critic")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def format_sources_for_prompt(sources: list[Source]) -> str:
    """Render sources for planner/critic prompts."""
    if not sources:
        return "(none)"
    parts: list[str] = []
    for src in sources:
        header = f"[{src.ref}] kind={src.kind.value} doc={src.doc_id} score={src.score:.4f}"
        if src.title:
            header += f" title={src.title}"
        parts.append(f"{header}\n{src.quote}")
    return "\n\n".join(parts)


class Critic:
    """LLM-as-judge over a draft answer and its sources."""

    def __init__(self, llm: ChatLLM, *, prompt_version: str = "v1") -> None:
        self._llm = llm
        self._prompt_version = prompt_version

    async def evaluate(
        self,
        *,
        question: str,
        draft: str,
        sources: list[Source],
    ) -> tuple[Verdict, Usage]:
        template = get_prompt("agent_critique", self._prompt_version)
        prompt = template.render(
            question=question,
            draft=draft,
            sources=format_sources_for_prompt(sources),
        )
        completion = await self._llm.chat([Message(role="user", content=prompt)], tools=None)
        verdict = _parse_verdict(completion.text)
        return verdict, completion.usage


def _parse_verdict(text: str) -> Verdict:
    raw: dict[str, Any]
    match = _JSON_RE.search(text or "")
    if not match:
        log.warning("critic returned non-JSON; treating as insufficient")
        return Verdict(
            sufficient=False,
            grounded=False,
            reasons=["critic_parse_error"],
            missing=["retry with clearer evidence"],
            confidence=0.0,
        )
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(
            sufficient=False,
            grounded=False,
            reasons=["critic_parse_error"],
            missing=["retry with clearer evidence"],
            confidence=0.0,
        )
    if not isinstance(parsed, dict):
        return Verdict(sufficient=False, grounded=False, reasons=["critic_parse_error"])
    raw = parsed
    return Verdict(
        sufficient=bool(raw.get("sufficient", False)),
        grounded=bool(raw.get("grounded", False)),
        reasons=[str(r) for r in (raw.get("reasons") or [])],
        missing=[str(m) for m in (raw.get("missing") or [])],
        confidence=float(raw.get("confidence") or 0.0),
    )
