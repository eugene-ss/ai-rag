from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpanRecord:
    name: str
    latency_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_spans: ContextVar[list[SpanRecord] | None] = ContextVar("spans", default=None)


def current_trace_id() -> str:
    tid = _trace_id.get()
    if not tid:
        tid = uuid.uuid4().hex
        _trace_id.set(tid)
    return tid


def set_trace_id(trace_id: str) -> None:
    _trace_id.set(trace_id)


def get_spans() -> list[SpanRecord]:
    return list(_spans.get() or [])


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
    """Record stage latency / attrs. OTel exporter seam can wrap this later."""
    attrs: dict[str, Any] = dict(attributes)
    start = time.perf_counter()
    error: str | None = None
    try:
        yield attrs
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        record = SpanRecord(name=name, latency_ms=elapsed, attributes=attrs, error=error)
        existing = list(_spans.get() or [])
        existing.append(record)
        _spans.set(existing)


def reset_trace() -> str:
    """Start a fresh trace context; returns the new trace id."""
    tid = uuid.uuid4().hex
    _trace_id.set(tid)
    _spans.set([])
    return tid
