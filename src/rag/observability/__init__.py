"""Traces, metrics, cost accounting, and structured logging."""

from rag.observability.cost import estimate_cost_usd
from rag.observability.logging import configure_logging, get_logger
from rag.observability.metrics import METRICS, MetricsRegistry
from rag.observability.tracing import (
    SpanRecord,
    current_trace_id,
    get_spans,
    reset_trace,
    set_trace_id,
    span,
    start_trace,
)

__all__ = [
    "METRICS",
    "MetricsRegistry",
    "SpanRecord",
    "configure_logging",
    "current_trace_id",
    "estimate_cost_usd",
    "get_logger",
    "get_spans",
    "reset_trace",
    "set_trace_id",
    "span",
    "start_trace",
]
