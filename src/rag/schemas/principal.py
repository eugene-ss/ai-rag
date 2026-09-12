from __future__ import annotations

from pydantic import BaseModel, Field


class Principal(BaseModel):
    """Authenticated caller identity used for ACL filtering."""

    subject: str
    groups: frozenset[str] = Field(default_factory=frozenset)
    tenant: str


class AclTags(BaseModel):
    """Access-control tags attached to documents and chunks."""

    tenant: str
    allow_groups: frozenset[str] = Field(default_factory=frozenset)
    classification: str = "internal"

    def fingerprint(self) -> str:
        groups = ",".join(sorted(self.allow_groups))
        return f"{self.tenant}|{groups}|{self.classification}"
