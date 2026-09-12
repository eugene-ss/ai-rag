"""Provenance for anything an agent (or the pipeline) may cite.

A citation used to be chunk-shaped. That breaks the moment a second source type
exists — a web result or a graph fact has neither `chunk_id` nor `doc_id`.
`Source` is the generalised form; `Citation` stays additive for HTTP compat.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from rag.schemas.principal import AclTags


class SourceKind(StrEnum):
    CHUNK = "chunk"
    WEB = "web"
    GRAPH = "graph"
    TOOL = "tool"


class Source(BaseModel):
    """One piece of evidence the agent may use or cite.

    Carrying `acl` is what lets every tool result be re-checked before it enters
    the model context — the model must never see a source the caller cannot.
    """

    kind: SourceKind
    ref: str
    doc_id: str = ""
    title: str = ""
    quote: str = ""
    score: float = 0.0
    url: str | None = None
    acl: AclTags | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
