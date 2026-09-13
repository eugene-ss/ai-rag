"""Hermetic agent evaluation job for CI.

Scripts EchoChatLLM from each example's expected_tools so the gate stays offline
and deterministic. Real-model agent eval belongs in a separate nightly job.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

from rag.agent import AgentRuntime, Critic, RetrievalTool, ToolRegistry
from rag.agent.tools.base import Tool
from rag.backends import BackendContext
from rag.eval.agent_runner import AgentEvalReport, AgentEvalRunner, format_agent_report
from rag.eval.datasets import load_golden
from rag.jobs.reindex import reindex
from rag.llm.echo_chat import EchoChatLLM, ScriptedTurn
from rag.pipelines.online import from_context as online_from_context
from rag.schemas import AclTags, Principal
from rag.schemas.agent import ToolCall


def run_agent_eval(
    dataset: Path | str,
    *,
    context: BackendContext,
    index_source: Path | str | None = None,
    max_avg_steps: float | None = None,
    max_avg_cost: float | None = None,
    min_success_rate: float | None = None,
) -> AgentEvalReport:
    settings = context.settings
    if index_source is not None:
        reindex(
            Path(index_source),
            index_version=settings.index_version,
            activate=True,
            context=context,
            acl=AclTags(
                tenant=settings.default_tenant,
                allow_groups=frozenset({"public"}),
            ),
        )

    pipeline = online_from_context(context)
    principal = Principal(
        subject="eval-agent",
        tenant=settings.default_tenant,
        groups=frozenset({"public"}),
    )
    examples = load_golden(dataset)

    # One retrieval hit gives a citable chunk id for the scripted final answer.
    probe = pipeline.retrieve_only(
        examples[0].question if examples else "overview", principal=principal
    )
    chunk_id = probe.results[0].chunk.chunk_id if probe.results else "missing"

    script: list[ScriptedTurn] = []
    for ex in examples:
        tools = ex.expected_tools or ["retrieval_search"]
        script.append(
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id=f"{ex.id}-{i}", name=name, arguments={"query": ex.question})
                    for i, name in enumerate(tools)
                    if name == "retrieval_search"
                ]
            )
        )
        script.append(
            ScriptedTurn(
                content=(
                    (ex.reference_answer or "Grounded answer from the corpus.") + f" [{chunk_id}]"
                )
            )
        )

    critic_script = [
        ScriptedTurn(
            content=(
                '{"sufficient": true, "grounded": true, "reasons": ["cited"], '
                '"missing": [], "confidence": 0.9}'
            )
        )
        for _ in examples
    ]

    runtime = AgentRuntime(
        chat_llm=EchoChatLLM(script=script),
        tools=ToolRegistry([cast(Tool, RetrievalTool(pipeline))]),
        pipeline=pipeline,
        critic=Critic(EchoChatLLM(script=critic_script)),
        budget=settings.agent_budget(),
        allow_egress=False,
        include_trace=True,
    )
    report = asyncio.run(
        AgentEvalRunner(
            runtime=runtime,
            principal=principal,
            budget_steps=settings.agent_max_steps,
        ).run(examples)
    )

    if min_success_rate is not None and report.success_rate < min_success_rate:
        msg = f"agent success_rate {report.success_rate:.3f} below threshold {min_success_rate:.3f}"
        raise SystemExit(msg)
    if max_avg_steps is not None and report.avg_steps > max_avg_steps:
        msg = f"agent avg_steps {report.avg_steps:.2f} above threshold {max_avg_steps:.2f}"
        raise SystemExit(msg)
    if max_avg_cost is not None and report.avg_cost_usd > max_avg_cost:
        msg = f"agent avg_cost_usd {report.avg_cost_usd:.4f} above threshold {max_avg_cost:.4f}"
        raise SystemExit(msg)
    return report


def format_report(report: AgentEvalReport) -> str:
    return format_agent_report(report)
