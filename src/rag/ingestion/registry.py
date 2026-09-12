from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag.ingestion.base import SourceConnector
from rag.ingestion.local_fs import LocalFSConnector

ConnectorFactory = Callable[..., SourceConnector]

REGISTRY: dict[str, ConnectorFactory] = {
    "local_fs": LocalFSConnector,
}


def get_connector(name: str, **kwargs: object) -> SourceConnector:
    try:
        factory = REGISTRY[name]
    except KeyError as exc:
        msg = f"Unknown connector: {name}. Available: {sorted(REGISTRY)}"
        raise KeyError(msg) from exc
    return factory(**kwargs)


def register(name: str, factory: ConnectorFactory) -> None:
    REGISTRY[name] = factory


def default_connector(root: Path | str = "data/raw") -> SourceConnector:
    return get_connector("local_fs", root=root)
