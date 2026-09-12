"""Deterministic agent evaluation against a golden set.

Uses a caller-supplied AgentRuntime (typically EchoChatLLM-scripted) so CI stays
hermetic. Real-model agent eval belongs in a separate nightly job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.eval.datasets import GoldenExample, load_golden
from rag.eval.metrics.agent import (
    cost_per_correct,
    self_correction_rate,
    step_efficiency,
    task_success,
    tool_selection_precision,
)
from rag.observability.tracing import reset_trace
from rag.schemas import Principal
from rag.schemas.agent import AgentAnswer


@dataclass
class AgentEvalCaseResult:
    example_id: str
    success: float
    steps: int
    step_efficiency: float
    tool_precision: float
    self_correction_rate: float
    cost_usd: float
    cost_per_correct: float
    fallback_used: bool
    refused: bool


@dataclass
class AgentEvalReport:
    n: int
    success_rate: float
    avg_steps: float
    avg_step_efficiency: float
    avg_tool_precision: float
    avg_self_correction_rate: float
    avg_cost_usd: float
    avg_cost_per_correct: float
    fallback_rate: float
    refusal_rate: float
    cases: list[AgentEvalCaseResult] = field(default_factory=list)

    @classmethod
    def from_cases(cls, cases: list[AgentEvalCaseResult]) -> AgentEvalReport:
        n = len(cases) or 1
        return cls(
            n=len(cases),
            success_rate=sum(c.success for c in cases) / n,
            avg_steps=sum(c.steps for c in cases) / n,
            avg_step_efficiency=sum(c.step_efficiency for c in cases) / n,
            avg_tool_precision=sum(c.tool_precision for c in cases) / n,
            avg_self_correction_rate=sum(c.self_correction_rate for c in cases) / n,
            avg_cost_usd=sum(c.cost_usd for c in cases) / n,
            avg_cost_per_correct=sum(c.cost_per_correct for c in cases) / n,
            fallback_rate=sum(1 for c in cases if c.fallback_used) / n,
            refusal_rate=sum(1 for c in cases if c.refused) / n,
            cases=cases,
        )


def format_agent_report(report: AgentEvalReport) -> str:
    return "\n".join(
        [
            f"AgentEvalReport n={report.n}",
            f"  Success={report.success_rate:.3f}  AvgSteps={report.avg_steps:.2f}",
            f"  StepEff={report.avg_step_efficiency:.3f}  ToolPrec={report.avg_tool_precision:.3f}",
            f"  SelfCorr={report.avg_self_correction_rate:.3f}",
            f"  AvgCost=${report.avg_cost_usd:.4f}  Cost/Correct=${report.avg_cost_per_correct:.4f}",
            f"  FallbackRate={report.fallback_rate:.3f}  RefusalRate={report.refusal_rate:.3f}",
        ]
    )


@dataclass
class AgentEvalRunner:
    """Runs golden-set agent evaluation against an async AgentRuntime.run."""

    runtime: Any
    principal: Principal
    budget_steps: int = 6

    async def run(self, dataset: list[GoldenExample] | Path | str) -> AgentEvalReport:
        examples = load_golden(dataset) if isinstance(dataset, (str, Path)) else dataset
        cases: list[AgentEvalCaseResult] = []
        for ex in examples:
            reset_trace()
            answer: AgentAnswer = await self.runtime.run(
                ex.question, principal=self.principal
            )
            success = task_success(answer)
            predicted_tools = list(answer.tools_used) or _tools_from_answer(answer)
            cost = answer.usage.cost_usd
            cases.append(
                AgentEvalCaseResult(
                    example_id=ex.id,
                    success=success,
                    steps=answer.steps_taken,
                    step_efficiency=step_efficiency(
                        answer.steps_taken, budget_steps=self.budget_steps
                    ),
                    tool_precision=tool_selection_precision(
                        predicted_tools, ex.expected_tools
                    ),
                    self_correction_rate=self_correction_rate(
                        answer.self_corrections, answer.steps_taken
                    ),
                    cost_usd=cost,
                    cost_per_correct=cost_per_correct(cost, success),
                    fallback_used=answer.fallback_used,
                    refused=answer.refused,
                )
            )
        return AgentEvalReport.from_cases(cases)


def _tools_from_answer(answer: AgentAnswer) -> list[str]:
    if answer.trace is None:
        return []
    names: list[str] = []
    for step in answer.trace.steps:
        names.extend(call.name for call in step.tool_calls)
    return names
