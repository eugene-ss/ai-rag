from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from rag.schemas.principal import AclTags


class Document(BaseModel):
    """Parsed source document ready for chunking."""

    doc_id: str
    source_uri: str
    mime_type: str
    checksum: str
    text: str | None = None
    acl: AclTags
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
