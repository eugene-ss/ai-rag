from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from rag.ingestion.base import DEFAULT_ACL, SourceConnector, checksum_bytes, guess_mime
from rag.schemas import AclTags, Document


class LocalFSConnector:
    """Ingest documents from a local filesystem directory."""

    def __init__(self, root: Path | str, *, pattern: str = "**/*") -> None:
        self.root = Path(root)
        self.pattern = pattern

    def list_uris(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for path in sorted(self.root.glob(self.pattern)):
            if path.is_file() and path.name != ".gitkeep" and path.name != "README.md":
                yield path.as_uri()

    def fetch(self, uri: str, *, acl: AclTags | None = None) -> Document:
        path = Path(uri.removeprefix("file://"))
        data = path.read_bytes()
        text: str | None
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        return Document(
            doc_id=path.stem,
            source_uri=uri,
            mime_type=guess_mime(path),
            checksum=checksum_bytes(data),
            text=text,
            acl=acl or DEFAULT_ACL,
            metadata={"path": str(path)},
        )


def create_local_fs(root: Path | str) -> SourceConnector:
    return LocalFSConnector(root)
