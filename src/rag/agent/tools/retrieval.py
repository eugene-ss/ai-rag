"""Retrieval tool — wraps OnlinePipeline.retrieve_only as an agent capability."""

from __future__ import annotations

import time
from typing import Any, ClassVar

from rag.agent.budget import BudgetTracker
from rag.agent.tools.mixin import ToolMixin
from rag.generation.citations import source_from_scored
from rag.pipelines.online import OnlinePipeline
from rag.schemas import Principal
from rag.schemas.agent import ToolResult
from rag.security.acl import is_allowed


class RetrievalTool(ToolMixin):
    """Dense+lexical hybrid retrieval under the caller's ACL.

    Calls `retrieve_only`, never `answer` — nested generation would double-spend
    the agent budget and hide the critic behind a nested refusal.
    """

    name = "retrieval_search"
    description = (
        "Search the indexed corpus with hybrid dense+lexical retrieval. "
        "Use for factual questions about internal documents. "
        "Returns passages the caller is allowed to see."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query; rewrite/expand as needed for multi-hop.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum passages to return (1-20).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    requires_egress = False

    def __init__(self, pipeline: OnlinePipeline) -> None:
        self._pipeline = pipeline

    async def run(
        self,
        arguments: dict[str, Any],
        *,
        principal: Principal,
        budget: BudgetTracker,
    ) -> ToolResult:
        _ = budget
        query = str(arguments["query"])
        top_k = int(arguments.get("top_k") or self._pipeline.settings.top_k)
        top_k = max(1, min(top_k, 20))

        started = time.perf_counter()
        # retrieve_only is sync; run it off the event loop.
        import asyncio

        result = await asyncio.to_thread(
            self._pipeline.retrieve_only, query, principal=principal
        )
        scored = result.results[:top_k]
        # Defense in depth: re-check ACL even though retrieval already filtered.
        sources = [
            source_from_scored(s)
            for s in scored
            if is_allowed(principal, s.chunk.acl)
        ]
        lines = [
            f"[{src.ref}] score={src.score:.4f} doc={src.doc_id}\n{src.quote}"
            for src in sources
        ]
        content = "\n\n".join(lines) if lines else "No authorised passages found."
        return ToolResult(
            call_id="",  # filled by registry
            name=self.name,
            ok=True,
            content=content,
            sources=sources,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
