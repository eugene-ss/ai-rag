from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from rag.schemas import AclTags, Document
from rag.utils.hashing import sha256_hex


class SourceConnector(Protocol):
    """Discovers and yields documents from a source system."""

    def list_uris(self) -> Iterator[str]: ...

    def fetch(self, uri: str, *, acl: AclTags | None = None) -> Document: ...


DEFAULT_ACL = AclTags(tenant="default", allow_groups=frozenset({"public"}))


def checksum_bytes(data: bytes) -> str:
    return sha256_hex(data)


def guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".pdf": "application/pdf",
        ".html": "text/html",
        ".htm": "text/html",
        ".json": "application/json",
    }.get(suffix, "application/octet-stream")
