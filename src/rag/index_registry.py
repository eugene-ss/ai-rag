"""Index version registry backed by configs/index_versions.yaml.

The registry records which physical index each version maps to and which one is
active. Reads are cheap and safe; writes degrade to a warning on a read-only
filesystem so a container with a mounted config never fails a job for it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from rag.observability.logging import get_logger

log = get_logger("index_registry")


class IndexVersion(BaseModel):
    version: str
    collection: str
    alias: str
    status: str = "building"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    chunk_count: int = 0
    embedding_model: str | None = None
    chunker_version: str | None = None


class IndexRegistry:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # --- reads -------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"active": None, "versions": {}}
        data = yaml.safe_load(self.path.read_text()) or {}
        if not isinstance(data, dict):
            return {"active": None, "versions": {}}
        data.setdefault("active", None)
        data.setdefault("versions", {})
        return data

    def active_version(self) -> str | None:
        active = self._load().get("active")
        return str(active) if active else None

    def get(self, version: str) -> IndexVersion | None:
        raw = self._load()["versions"].get(version)
        if not raw:
            return None
        return IndexVersion.model_validate({"version": version, **raw})

    def list_versions(self) -> list[IndexVersion]:
        raw = self._load()["versions"]
        return [
            IndexVersion.model_validate({"version": name, **body})
            for name, body in sorted(raw.items())
        ]

    # --- writes ------------------------------------------------------------

    def record(self, entry: IndexVersion) -> None:
        data = self._load()
        body = entry.model_dump(exclude={"version"})
        data["versions"][entry.version] = body
        self._save(data)

    def activate(self, version: str) -> None:
        data = self._load()
        versions = data["versions"]
        if version not in versions:
            msg = f"cannot activate unknown index version {version!r}"
            raise KeyError(msg)
        for name, body in versions.items():
            body["status"] = "active" if name == version else "retired"
        data["active"] = version
        self._save(data)
        log.info("index registry: active version -> %s", version)

    def _save(self, data: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(yaml.safe_dump(data, sort_keys=True))
        except OSError as exc:
            # Read-only config mount: the vector store alias remains the source
            # of truth at query time, so this is degraded, not fatal.
            log.warning("could not persist index registry at %s: %s", self.path, exc)
