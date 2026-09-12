from __future__ import annotations

from pathlib import Path

from rag.backends import BackendContext
from rag.index_registry import IndexRegistry
from rag.observability.logging import get_logger
from rag.pipelines import offline as offline_pipeline
from rag.schemas import AclTags

log = get_logger("jobs.reindex")


def reindex(
    source: Path | str,
    *,
    index_version: str,
    activate: bool = True,
    context: BackendContext | None = None,
    acl: AclTags | None = None,
) -> int:
    """Build index `index_version` from `source`, optionally flipping the alias.

    Build-then-swap: the live alias only moves after the new version is fully
    written, so readers never observe a half-built index.
    """
    ctx = context or BackendContext.from_settings(index_version=index_version)
    pipeline = offline_pipeline.from_context(ctx)
    pipeline.registry = IndexRegistry(ctx.settings.index_registry_path())
    chunks = pipeline.run(source, index_version=index_version, acl=acl, activate=activate)
    return len(chunks)


def activate_version(version: str, *, context: BackendContext | None = None) -> None:
    """Promote an already-built version, or roll back to a previous one."""
    ctx = context or BackendContext.from_settings(index_version=version)
    offline_pipeline.from_context(ctx).activate(version)


def list_versions(*, context: BackendContext | None = None) -> list[str]:
    ctx = context or BackendContext.from_settings()
    registry = IndexRegistry(ctx.settings.index_registry_path())
    active = registry.active_version()
    lines: list[str] = []
    for v in registry.list_versions():
        marker = " (active)" if v.version == active else ""
        lines.append(
            f"{v.version}\t{v.status}\tchunks={v.chunk_count}\t"
            f"embedding={v.embedding_model}{marker}"
        )
    return lines
